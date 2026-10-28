"""Fused device kernels of the fast R584 scorer (EXP586). They compute the function of shared_evict.SharedEvict
with fewer launches; shared_evict_fast.FastSharedEvict calls them. The original kernels stay unchanged.

Rounding rule: where the original runs one torch op per step, these kernels round each step the same way.
Products and sums use libdevice mul_rn/add_rn, which the compiler cannot contract into an FMA. exp, log,
log1p and erf are the libdevice functions (the accurate CUDA math functions), not the fast tl.exp/tl.log.
Equality with the torch kernels is a GPU test target (tests/test_gpu_exp586_fast.py), not a claim: the
torch reduction order of logsumexp's sum over 4 values, for example, is not documented.
Denormals: Triton links libdevice with flush-to-zero by default (enable_reflect_ftz), so mul_rn would
become mul.rn.ftz and an ema below 2^-126 would flush to 0, unlike torch. Every launch of these kernels
must pass enable_reflect_ftz=False (shared_evict_fast.NO_FTZ).

Kernels:
  state_own   F4: advance the eviction state of one layer (ema, age, routed; only when step > 0) and write
              the BF16 own features x = [routed, ema, agef, resident] in the w_layer input layout.
  qgemv2      F8/F6: qbank.qgemv with a head stride for X (0 = one input row shared by all heads), an
              optional GELU epilogue on the BF16 output, and an optional split of the K groups (FP32
              partials; splitk_reduce adds them in split order, adds the bias once and rounds once).
  readout     F3: the head tail and the mix of one layer: skip terms, softplus, cumulative horizons, P64,
              the log-domain mixture logit z, and a count of non-finite z.
  publish     F3: valid = (step > 0) & (no non-finite z); count the non-finite tokens; clear the count.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

SQRT1_2 = tl.constexpr(0.7071067811865476)  # torch GELU kAlpha = M_SQRT1_2, rounded to FP32 in the kernel
SOFTPLUS_THRESHOLD = tl.constexpr(20.0)  # torch F.softplus default (beta 1): x above it returns x


@triton.jit
def state_own(ROUTES, STEP, EMA, AGE, ROUTED, AGE_TABLE, RESIDENT, X, beta, one_minus_beta, far,
              E: tl.constexpr, K: tl.constexpr, K_POW2: tl.constexpr):
    """One program per layer l. ROUTES (L, K) int64; STEP (1,) int64; EMA/ROUTED/RESIDENT (L, E) FP32;
    AGE (L, E) int64 index of AGE_TABLE (far + 1,) FP32; X (L, 4 * E) BF16 out, X[l, f * E + e].

    As SharedEvict.advance then own: routed = 1 at the routed ids (a repeated id is still 1), ema = ema *
    beta + routed * (1 - beta) with three FP32 roundings, age = 0 if routed else min(age + 1, far); the new
    state is stored only when step > 0. The own features read the state after that update."""
    l = tl.program_id(0)
    e = tl.arange(0, E)
    k = tl.arange(0, K_POW2)
    r = tl.load(ROUTES + l * K + k, mask=k < K, other=-1)
    hit = tl.sum((r[None, :] == e[:, None]).to(tl.int32), 1) > 0
    new_routed = hit.to(tl.float32)
    go = tl.load(STEP) > 0
    ema = tl.load(EMA + l * E + e)
    age = tl.load(AGE + l * E + e)
    routed = tl.load(ROUTED + l * E + e)
    new_ema = libdevice.add_rn(libdevice.mul_rn(ema, beta), libdevice.mul_rn(new_routed, one_minus_beta))
    new_age = tl.where(hit, tl.zeros_like(age), tl.minimum(age + 1, far))
    ema = tl.where(go, new_ema, ema)
    age = tl.where(go, new_age, age)
    routed = tl.where(go, new_routed, routed)
    tl.store(EMA + l * E + e, ema)
    tl.store(AGE + l * E + e, age)
    tl.store(ROUTED + l * E + e, routed)
    agef = tl.load(AGE_TABLE + age)
    resident = tl.load(RESIDENT + l * E + e)
    row = X + l * 4 * E
    tl.store(row + e, routed.to(tl.bfloat16))
    tl.store(row + E + e, ema.to(tl.bfloat16))
    tl.store(row + 2 * E + e, agef.to(tl.bfloat16))
    tl.store(row + 3 * E + e, resident.to(tl.bfloat16))


@triton.jit
def _gelu_bf16(y):
    """torch F.gelu (exact erf form) of a BF16 value: FP32 x * 0.5 * (1 + erf(x * kAlpha)), BF16 out."""
    x = y.to(tl.float32)
    t = libdevice.add_rn(1.0, libdevice.erf(libdevice.mul_rn(x, SQRT1_2)))
    return libdevice.mul_rn(libdevice.mul_rn(x, 0.5), t).to(tl.bfloat16)


@triton.jit
def qgemv2(X, W, S, B, Y, P, N, K, KG, X_HEAD_STRIDE, G_PER_SPLIT, HAS_BIAS: tl.constexpr, GELU: tl.constexpr,
           SPLIT: tl.constexpr, GROUP: tl.constexpr, BLOCK_N: tl.constexpr):
    """Y[h, n] = BF16(sum_k W[h, n, k] * S[h, n, k // GROUP] * X[h * X_HEAD_STRIDE + k] (+ B[h, n])).

    The loop body is qbank.qgemv's. Grid (H, cdiv(N, BLOCK_N), SPLIT). SPLIT == 1: the qgemv epilogue
    (bias in FP32, BF16 rounding), then GELU when set. SPLIT > 1: program s sums the groups
    s * G_PER_SPLIT .. min((s + 1) * G_PER_SPLIT, KG) - 1 into P[s, h, n] (FP32), and splitk_reduce
    finishes; Y, B and GELU are then unused here."""
    h = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    s_id = tl.program_id(2)
    n_ok = n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    g0 = s_id * G_PER_SPLIT
    g1 = tl.minimum(g0 + G_PER_SPLIT, KG)
    for g in range(g0, g1):
        k = g * GROUP + tl.arange(0, GROUP)
        k_ok = k < K
        x = tl.load(X + h * X_HEAD_STRIDE + k, mask=k_ok, other=0.0).to(tl.float32)
        w = tl.load(W + (h * N + n[:, None]) * K + k[None, :], mask=n_ok[:, None] & k_ok[None, :],
                    other=0).to(tl.float32)
        s = tl.load(S + (h * N + n) * KG + g, mask=n_ok, other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1) * s
    if SPLIT == 1:
        if HAS_BIAS:
            acc += tl.load(B + h * N + n, mask=n_ok, other=0.0).to(tl.float32)
        y = acc.to(tl.bfloat16)
        if GELU:
            y = _gelu_bf16(y)
        tl.store(Y + h * N + n, y, mask=n_ok)
    else:
        H = tl.num_programs(0)
        tl.store(P + (s_id * H + h) * N + n, acc, mask=n_ok)


@triton.jit
def splitk_reduce(P, B, Y, HN, HAS_BIAS: tl.constexpr, GELU: tl.constexpr, SPLIT: tl.constexpr,
                  BLOCK: tl.constexpr):
    """Y[i] = BF16(((P[0, i] + P[1, i]) + ...) + B[i]) for i < HN = H * N, then GELU when set."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < HN
    acc = tl.load(P + i, mask=ok, other=0.0)
    for s in tl.static_range(1, SPLIT):
        acc = libdevice.add_rn(acc, tl.load(P + s * HN + i, mask=ok, other=0.0))
    if HAS_BIAS:
        acc = libdevice.add_rn(acc, tl.load(B + i, mask=ok, other=0.0).to(tl.float32))
    y = acc.to(tl.bfloat16)
    if GELU:
        y = _gelu_bf16(y)
    tl.store(Y + i, y, mask=ok)


@triton.jit
def _skip(o0, o1, o2, o3, W, stride):
    """BF16(o0 w0 + o1 w1 + o2 w2 + o3 w3) as skip(): FP32 products added in feature order, BF16-rounded,
    returned in FP32. W[f * stride] is feature f's BF16-rounded weight in FP32."""
    acc = libdevice.mul_rn(o0, tl.load(W))
    acc = libdevice.add_rn(acc, libdevice.mul_rn(o1, tl.load(W + stride)))
    acc = libdevice.add_rn(acc, libdevice.mul_rn(o2, tl.load(W + 2 * stride)))
    acc = libdevice.add_rn(acc, libdevice.mul_rn(o3, tl.load(W + 3 * stride)))
    return acc.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _softplus(x):
    """torch F.softplus, beta 1, threshold 20: x if x > 20 else log1p(exp(x))."""
    return tl.where(x > SOFTPLUS_THRESHOLD, x, libdevice.log1p(libdevice.exp(x)))


@triton.jit
def _logsigmoid(x):
    """torch F.logsigmoid on CUDA: min(0, x) - log1p(exp(-|x|))."""
    return libdevice.sub_rn(tl.minimum(0.0, x), libdevice.log1p(libdevice.exp(-tl.abs(x))))


@triton.jit
def _lse4(a0, a1, a2, a3):
    """torch.logsumexp over 4 values: m = max (0 if infinite), log(sum exp(a - m)) + m. The sum runs
    ((a0 + a1) + a2) + a3; whether torch's reduction uses this order is checked on the GPU."""
    m = tl.maximum(tl.maximum(a0, a1), tl.maximum(a2, a3))
    m = tl.where(tl.abs(m) == float('inf'), 0.0, m)
    s = libdevice.add_rn(libdevice.exp(libdevice.sub_rn(a0, m)), libdevice.exp(libdevice.sub_rn(a1, m)))
    s = libdevice.add_rn(s, libdevice.exp(libdevice.sub_rn(a2, m)))
    s = libdevice.add_rn(s, libdevice.exp(libdevice.sub_rn(a3, m)))
    return libdevice.add_rn(libdevice.log(s), m)


@triton.jit
def readout(WB, X, WIDE, WIDE64, ELL, Z, U, BAD, E: tl.constexpr, WRITE_U: tl.constexpr):
    """One program per layer l, all E experts. WB (L, 4 * E) BF16: the joint wb/wb64 qgemv output, wb at
    e * 3 + j and wb64 at 3 * E + e; X (L, 4 * E) BF16 own features (state_own); WIDE (L, 4, 3) and
    WIDE64 (L, 4) FP32 BF16-rounded skip weights; ELL (L, 4) FP32 log_softmax(mix_alpha).
    Z (L, E) FP32 keep score out; U (L, E, 4) FP32 cumulative logits out when WRITE_U; BAD (1,) int32
    += the number of non-finite z of the layer.

    As SharedEvict.heads then mix: raw_j = wb_j + skip_j (FP32), ev = cumsum(raw0, softplus(raw1),
    softplus(raw2)) in order, P64 = ev2 + softplus(wb64 + skip64), z = lse(ell + logsigmoid(u)) -
    lse(ell + logsigmoid(-u))."""
    l = tl.program_id(0)
    e = tl.arange(0, E)
    xrow = X + l * 4 * E
    o0 = tl.load(xrow + e).to(tl.float32)
    o1 = tl.load(xrow + E + e).to(tl.float32)
    o2 = tl.load(xrow + 2 * E + e).to(tl.float32)
    o3 = tl.load(xrow + 3 * E + e).to(tl.float32)
    wrow = WB + l * 4 * E
    raw0 = libdevice.add_rn(tl.load(wrow + e * 3).to(tl.float32), _skip(o0, o1, o2, o3, WIDE + l * 12, 3))
    raw1 = libdevice.add_rn(tl.load(wrow + e * 3 + 1).to(tl.float32), _skip(o0, o1, o2, o3, WIDE + l * 12 + 1, 3))
    raw2 = libdevice.add_rn(tl.load(wrow + e * 3 + 2).to(tl.float32), _skip(o0, o1, o2, o3, WIDE + l * 12 + 2, 3))
    r = libdevice.add_rn(tl.load(wrow + 3 * E + e).to(tl.float32), _skip(o0, o1, o2, o3, WIDE64 + l * 4, 1))
    u0 = raw0
    u1 = libdevice.add_rn(u0, _softplus(raw1))
    u2 = libdevice.add_rn(u1, _softplus(raw2))
    u3 = libdevice.add_rn(u2, _softplus(r))
    l0 = tl.load(ELL + l * 4)
    l1 = tl.load(ELL + l * 4 + 1)
    l2 = tl.load(ELL + l * 4 + 2)
    l3 = tl.load(ELL + l * 4 + 3)
    pos = _lse4(libdevice.add_rn(l0, _logsigmoid(u0)), libdevice.add_rn(l1, _logsigmoid(u1)),
                libdevice.add_rn(l2, _logsigmoid(u2)), libdevice.add_rn(l3, _logsigmoid(u3)))
    neg = _lse4(libdevice.add_rn(l0, _logsigmoid(-u0)), libdevice.add_rn(l1, _logsigmoid(-u1)),
                libdevice.add_rn(l2, _logsigmoid(-u2)), libdevice.add_rn(l3, _logsigmoid(-u3)))
    z = libdevice.sub_rn(pos, neg)
    tl.store(Z + l * E + e, z)
    if WRITE_U:
        urow = U + (l * E + e) * 4
        tl.store(urow, u0)
        tl.store(urow + 1, u1)
        tl.store(urow + 2, u2)
        tl.store(urow + 3, u3)
    # |z| < inf is false for +-inf and for NaN, as torch.isfinite.
    bad = tl.sum(tl.where(tl.abs(z) < float('inf'), 0, 1), 0).to(tl.int32)
    tl.atomic_add(BAD, bad)


@triton.jit
def publish(BAD, STEP, VALID, NONFINITE):
    """VALID = (step > 0) & (BAD == 0) as int32; NONFINITE += (step > 0) & (BAD > 0); BAD = 0 for the
    next token. The integer count makes the result independent of the order of readout's atomic adds."""
    bad = tl.load(BAD)
    scored = tl.load(STEP) > 0
    tl.store(VALID, (scored & (bad == 0)).to(tl.int32))
    tl.store(NONFINITE, tl.load(NONFINITE) + (scored & (bad > 0)).to(tl.int64))
    tl.store(BAD, tl.zeros_like(bad))
