"""EXP586: the fast R584 scorer (shared_evict_fast.FastSharedEvict) on the GPU.

Step B runs C0-C2 and B2 for both F8 variants; step C runs C0-C2, P1F, P2F and F1 for the F8 variant of
EXP586_F8, then a2 last. The runner (experiments/exp586_gpu_window.py) refuses any skip.

C0  Components on the EXP585 fixtures (the P0 checks, adapted to the fast call structure): after
    advance_and_own the state is bit-exact (fixture ema and age; routed equal to SharedEvict.advance) and
    x equals SharedEvict's own features in the w_layer layout bit for bit. Each stage gets the ORIGINAL
    path's input of that stage: every pre-activation is within the unchanged P0 tolerance of
    reference_linear (exp585_window_checks.qgemv_ratio <= 1); each GELU epilogue output is within one
    BF16 spacing of F.gelu of the same pre-activation; both wbx slices are within the P0 tolerance of the
    original wb and wb64 stages. Bit equality with qbank.qlinear is reported, not gated. The fast trunk's
    own assembly is checked too: its fc1 input must equal the bytes that the original heads code builds
    from the fast w_layer and h_proj outputs, and each later stage must read the previous fast output.
    The criteria are recorded, reported, then asserted by one pytest.fail(C0_REJECT ...), the only C0
    failure that step B treats as a numerical rejection of the variant.
C1  qgemv2 shapes outside the bundle's: N not a multiple of BLOCK_N, K not a multiple of GROUP, groups not
    divisible by the split count, bias and GELU on and off, head stride 0. The split-K scratch is filled
    with NaN before each launch, so an unwritten partial fails the finite check.
C2  readout + publish against the ORIGINAL torch code (SharedEvict.heads tail and SharedEvict.mix, run
    with the same wb and wb64 outputs) on fixture inputs and adversarial inputs (large signed values,
    cancellation, near ties, softplus threshold 20, exp underflow to subnormals, NaN and inf). Declared
    tolerances, fixed before any GPU run: |du_j| <= 2^-20 S_j, where S_j = |u_0| + (u_j - u_0) bounds
    the sum of the magnitudes of the terms added into u_j (softplus >= 0); |dz| <= 2^-20 (|pos| + |neg|)
    + 2 max_j tol_u_j, because logsigmoid and logsumexp pass an input error at most once each to pos
    and neg. The finite masks of z must be equal, the non-finite count must equal the reference count,
    and publish must give valid 0 and count the token exactly when a scored token has a non-finite z.
C3  state_own against SharedEvict.advance + own on seeded edge states (zero, subnormal and near-2^-126
    ema, ages near far, duplicate route ids, step 0 after a nonzero state, reset and replay): bit for bit.
P1F Graph against eager with the fast hooks (one-pass and split, J = 1), as P1: residency, keep, valid,
    state, probabilities and the victim decisions, plus the prior advanced exactly once per token.
    Memory: the reserve, the scorer's buffers, the split-K scratch and the capture are reported.
P2F The P2 gates, unchanged, with the fast scorer: bottom-16 overlap mean >= 0.99 against the
    emulator scores; the victim kernel chooses the same victim for >= 99% of evictions.
F1  Asynchronous lifetime (Codex c22e40f9): replay -> after_decode -> next token without any host
    synchronization, the post-ready raw copies delayed by a sleep kernel, inputs changing every token, two
    requests back to back; graph and eager, one-pass and split. Equal bit for bit to the same run with a
    synchronization after every step, and the bank inputs and probabilities equal to the original shared
    mode. A mutation that removes main's join of the probe stream must change the eager result.
B2  Component timings (step B): original against fast stages as captured graphs, cold L2, blocks
    original/fast/fast/original (F8 stages: original/i/ii/ii/i/original), 20 warm-ups and 200 timed
    replays per block, on the first valid fixture event (not a2's token). Every output must be finite.
    Each block's median/p90/max and the pooled statistics per arm are reported; the runner's F8 rule
    reads the pooled 'scorer' medians.
a2  The EXP585 gate, unchanged (the same graph and statistic: test_gpu_shared_evict._prefix_graph and
    _block, blocks base/cand/cand/base, 150 us on the candidate median), with the fast arm as candidate.
    Before the assertion, a separate stamped diagnostic graph reports scores-ready, probs written and the
    end after the prior update for both arms.

Every test writes its numbers as JSON lines to EXP585_GPU_REPORT (test_gpu_shared_evict._report; the runner
gives each run its own file), which must not exist when the session starts.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import exp585_window_checks as wc
from test_gpu_learned_evict import _shadow
from test_gpu_schedule import PUBLISHED, D, E, K, L, _cache, _capture, _check_resident_rows, _decode, _Inputs
from test_gpu_shared_evict import (A2_GATE_NS, BUNDLE, FLUSH_BYTES, OVERLAP_GATE, R36, START, STEPS, TIMED,
                                   VICTIM_GATE, WARMUP, _args, _block, _fixtures, _follow, _pinned,
                                   _prefix_graph, _release, _report, _rule, bottom_overlap)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

VARIANT = os.environ.get('EXP586_F8')   # the F8 variant of the step C tests
CUDA = torch.device('cuda')
BF = torch.bfloat16
TOL_REL = 2.0 ** -20     # C2: 16 FP32 units in the last place of the magnitude bound
DELAY_CYCLES = 50_000_000  # F1: about 20 ms at 2.5 GHz, longer than the 40 stand-in layers (about 4 ms)
VARIANTS = ('blockn', 'splitk')


def _variant() -> str:
    if VARIANT not in VARIANTS:
        pytest.skip(f'EXP586_F8={VARIANT!r}, expected one of {VARIANTS}')
    return VARIANT


def _scorers(record: list | None = None):
    """(original GPU scorer with qlinear, {variant: fast scorer}, GPU table). ``record`` collects every
    qlinear call of the original as (x, codes, scale, bias, group, y)."""
    from commitmoe_prefetch import shared_evict as se
    from commitmoe_prefetch.qbank import qlinear
    from commitmoe_prefetch.shared_evict_fast import FastSharedEvict
    release, table = _release()
    meta, t = se.load_bundle(BUNDLE, release, L, E, K, D, table, 256, 128)

    def recorded(x, codes, scale, bias, group):
        y = qlinear(x, codes, scale, bias, group)
        record.append((x, codes, scale, bias, group, y))
        return y
    orig = se.SharedEvict(meta, t, CUDA, recorded if record is not None else qlinear)
    fast = {v: FastSharedEvict(meta, t, CUDA, v) for v in VARIANTS}
    return orig, fast, table.to(CUDA)


def _x_of(own: torch.Tensor) -> torch.Tensor:
    """SharedEvict's w_layer input: own (L, E, 4) as x[l, f * E + e] in BF16."""
    return own.permute(0, 2, 1).reshape(own.shape[0], -1).to(BF).contiguous()


def _ratio(y, x, codes, scale, bias, group) -> torch.Tensor:
    from commitmoe_prefetch.shared_evict import reference_linear
    with wc.fp32_exact():
        ref = reference_linear(x, codes, scale, bias, group)
    return wc.qgemv_ratio(y, ref, x, codes, scale, group)  # raises on any nonfinite value


def _gelu_ratio(post: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """|post - F.gelu(pre)| in BF16 spacings of the reference; the epilogue must stay within one."""
    ref = F.gelu(pre)
    assert bool(torch.isfinite(post).all()) and bool(torch.isfinite(ref).all())
    return (post.float() - ref.float()).abs() / wc.bf16_spacing(ref)


# --- C0: components on the fixtures ---------------------------------------------------------------------

C0_REJECT = 'C0 numerical rejection: '  # exp586_gpu_window.C0_REJECT; the runner binds the two
C0_CRITERIA = ('state_exact', 'x_exact', 'assembly_exact', 'stage_ratio_le_1', 'gelu_spacings_le_1')


def _recording(s, log: list):
    """Make s.matvec append (name, x, y) for every call (FastSharedEvict.trunk calls self.matvec)."""
    real = type(s).matvec

    def matvec(x, name, gelu=False, shared_row=False):
        y = real(s, x, name, gelu=gelu, shared_row=shared_row)
        log.append((name, x, y))
        return y
    s.matvec = matvec


def _assembled_fc1_input(orig, own, args, table, z, hp) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(w_layer input, h_proj input, fc1 input) that the ORIGINAL SharedEvict.heads code builds when its
    w_layer and h_proj outputs are ``z`` and ``hp`` (the fast trunk's own outputs). The later stages
    return zeros: only the assembly before fc1 is read."""
    seen = []
    given = iter([z, hp])

    def lin(x, codes, scale, bias, group):
        seen.append(x)
        y = next(given, None)
        return torch.zeros(codes.shape[0], codes.shape[1], dtype=BF, device=CUDA) if y is None else y
    twin = copy.copy(orig)
    twin.linear = lin
    twin.heads(own, *args, table)
    return seen[0], seen[1], seen[2]


@pytest.mark.parametrize('variant', VARIANTS)
def test_c0_components(variant):
    """The declared criteria (C0_CRITERIA) are recorded over all events, written to the report, and only
    then asserted with one pytest.fail(C0_REJECT ...): that is the only failure the step B runner accepts
    as a numerical rejection of the variant. Every other failure (a structural assert, a non-finite
    stage output in _ratio, a CUDA or compile error) raises before the report row and ends the attempt."""
    fx = _fixtures()
    calls: list = []
    orig, fast, table = _scorers(calls)
    s = fast[variant]
    fast_calls: list = []
    _recording(s, fast_calls)
    trunk = ('w_layer', 'h_proj', 'fc1', 'fc2', 'we')
    worst = dict.fromkeys(trunk + ('wb', 'wb64'), 0.0)
    gelu_worst = dict.fromkeys(('fc1', 'fc2', 'we'), 0.0)
    exact = dict.fromkeys(worst, 0)
    gelu_exact = dict.fromkeys(gelu_worst, 0)
    total = dict.fromkeys(worst, 0)
    bad = dict(state=[], x=[], assembly=[])
    events = scored = 0
    for e in fx['events']:
        if e['kind'] == 'reset':
            orig.reset()
            s.reset()
            events += 1
            continue
        x_in = e['inputs']
        step = torch.tensor([e['step']], dtype=torch.int32, device=CUDA)
        routes, res = x_in['routes'].to(CUDA), x_in['res'].float().to(CUDA)
        orig.advance(routes, step)
        x = s.advance_and_own(routes, step, res)
        events += 1
        if not (torch.equal(s.ema.cpu(), e['expect']['ema']) and torch.equal(s.age.cpu(), e['expect']['age'].long())
                and torch.equal(s.routed, orig.routed)):
            bad['state'].append(events)
        own = orig.own(res)
        if not torch.equal(x, _x_of(own)):
            bad['x'].append(events)
        if not e['expect']['valid']:
            continue
        scored += 1
        args = _args(x_in, fx['meta']['hidden_layers'], CUDA)
        # The fast trunk's own feature assembly (Codex 8b620dc5 item 4): its fc1 input must be the bytes
        # the original code assembles from the same w_layer and h_proj outputs, and each later stage must
        # read the previous fast stage's output.
        fast_calls.clear()
        s.trunk(x, *args, table)
        assert [n for n, _, _ in fast_calls] == ['w_layer', 'h_proj', 'fc1', 'fc2', 'we', 'wbx'], fast_calls
        f = {n: (xi, y) for n, xi, y in fast_calls}
        wl_in, hp_in, fc1_in = _assembled_fc1_input(orig, own, args, table, f['w_layer'][1], f['h_proj'][1])
        if not (torch.equal(f['w_layer'][0], wl_in) and torch.equal(f['h_proj'][0], hp_in)
                and torch.equal(f['fc1'][0], fc1_in) and torch.equal(f['fc2'][0], f['fc1'][1])
                and torch.equal(f['we'][0], f['fc2'][1]) and torch.equal(f['wbx'][0], f['we'][1])):
            bad['assembly'].append(events)
        # Each stage on the ORIGINAL path's input of that stage: the arithmetic alone.
        calls.clear()
        orig.heads(own, *args, table)
        for i, name in enumerate(trunk):
            xin, codes, scale, bias, group, y_orig = calls[i]
            shared = name == 'we'
            if shared:
                assert bool((xin == xin[:1]).all())  # the original expands one row
            arg = xin[:1].contiguous() if shared else xin
            pre = s.matvec(arg, name, shared_row=shared)
            worst[name] = max(worst[name], float(_ratio(pre, xin, codes, scale, bias, group).max()))
            exact[name] += int((pre == y_orig).sum())
            total[name] += pre.numel()
            if name in gelu_worst:
                post = s.matvec(arg, name, gelu=True, shared_row=shared)
                r = _gelu_ratio(post, pre)
                gelu_worst[name] = max(gelu_worst[name], float(r.max()))
                gelu_exact[name] += int((r == 0).sum())
        he = calls[5][0]
        assert torch.equal(he, calls[6][0])
        out = s.matvec(he, 'wbx')
        n_wb = calls[5][5].shape[1]
        for j, name in ((5, 'wb'), (6, 'wb64')):
            xin, codes, scale, bias, group, y_orig = calls[j]
            part = out[:, :n_wb] if name == 'wb' else out[:, n_wb:]
            worst[name] = max(worst[name], float(_ratio(part.contiguous(), xin, codes, scale, bias, group).max()))
            exact[name] += int((part == y_orig).sum())
            total[name] += part.numel()
    assert scored == sum(e['expect']['valid'] for e in fx['events']) > 0 and events == len(fx['events'])
    criteria = dict(state_exact=not bad['state'], x_exact=not bad['x'], assembly_exact=not bad['assembly'],
                    stage_ratio_le_1=max(worst.values()) <= 1.0, gelu_spacings_le_1=max(gelu_worst.values()) <= 1.0)
    assert tuple(criteria) == C0_CRITERIA
    failed = [k for k, ok in criteria.items() if not ok]
    _report('c0', variant=variant, completed=True, criteria=criteria, failed=failed, events=events, scored=scored,
            bad_events={k: v[:20] for k, v in bad.items()}, worst_ratio=worst, gelu_worst_spacings=gelu_worst,
            exact_share_vs_qlinear={n: exact[n] / max(total[n], 1) for n in total},
            gelu_exact_share={n: gelu_exact[n] / max(total[n], 1) for n in gelu_exact})
    if failed:
        pytest.fail(C0_REJECT + ','.join(failed), pytrace=False)


# --- C1: qgemv2 shapes ------------------------------------------------------------------------------------

C1_CASES = [  # H, N, K, block_n, split, bias, gelu, shared_row
    (2, 100, 300, 32, 1, True, True, False),
    (1, 8, 128, 8, 1, False, False, False),
    (3, 40, 7 * 128, 32, 3, True, True, False),   # 7 groups as 3/3/1
    (2, 64, 5 * 128 - 5, 32, 2, False, False, True),  # 5 groups as 3/2, a short last group, head stride 0
]


@pytest.mark.parametrize('case', C1_CASES)
def test_c1_qgemv2_shapes(case):
    from commitmoe_prefetch import fast_kernels as fk
    from commitmoe_prefetch.shared_evict_fast import NO_FTZ, REDUCE_BLOCK, split_groups
    import triton
    H, N, Kd, block_n, split, has_bias, gelu, shared = case
    group = 128
    KG = triton.cdiv(Kd, group)
    g = torch.Generator().manual_seed(17)
    codes = torch.randint(-127, 128, (H, N, Kd), generator=g, dtype=torch.int8).to(CUDA)
    scale = (torch.rand(H, N, KG, generator=g) * 1e-2 + 1e-3).half().to(CUDA)
    bias = torch.randn(H, N, generator=g).to(CUDA) if has_bias else None
    x = torch.randn(1 if shared else H, Kd, generator=g).to(BF).to(CUDA)
    xin = x.expand(H, -1).contiguous() if shared else x
    stride = 0 if shared else Kd
    y = torch.empty(H, N, dtype=BF, device=CUDA)
    b = bias if bias is not None else y
    values = {}
    for act in ((False, True) if gelu else (False,)):
        if split == 1:
            fk.qgemv2[(H, triton.cdiv(N, block_n), 1)](x, codes, scale, b, y, y, N, Kd, KG, stride, KG,
                                                       HAS_BIAS=has_bias, GELU=act, SPLIT=1, GROUP=group,
                                                       BLOCK_N=block_n, num_warps=4, **NO_FTZ)
        else:
            P = torch.full((split, H, N), float('nan'), device=CUDA)
            fk.qgemv2[(H, triton.cdiv(N, block_n), split)](x, codes, scale, y, y, P, N, Kd, KG, stride,
                                                           split_groups(KG, split), HAS_BIAS=False, GELU=False,
                                                           SPLIT=split, GROUP=group, BLOCK_N=block_n,
                                                           num_warps=4, **NO_FTZ)
            assert bool(torch.isfinite(P).all()), 'a partial was not written'
            fk.splitk_reduce[(triton.cdiv(H * N, REDUCE_BLOCK),)](P, b, y, H * N, HAS_BIAS=has_bias, GELU=act,
                                                                 SPLIT=split, BLOCK=REDUCE_BLOCK, num_warps=4,
                                                                 **NO_FTZ)
        torch.cuda.synchronize()
        if not act:
            pre = y.clone()
            values['pre_ratio_max'] = float(_ratio(pre, xin, codes, scale, bias, group).max())
        else:
            values['gelu_spacings_max'] = float(_gelu_ratio(y, pre).max())
    _report('c1', case=list(case), **values)
    assert values['pre_ratio_max'] <= 1.0 and values.get('gelu_spacings_max', 0.0) <= 1.0, values


# --- C2: readout + publish against the original torch tail ------------------------------------------------

def _original_readout(orig, own: torch.Tensor, wbo: torch.Tensor, table: torch.Tensor):
    """(u, z) of the served torch code for given wb/wb64 outputs: SharedEvict.heads runs with a linear that
    returns zeros for the five trunk stages and wbo's slices for wb and wb64, then SharedEvict.mix. The
    tail that runs is the original code, not a copy of it."""
    Lw, E4 = wbo.shape
    outs = iter([None] * 5 + [wbo[:, :3 * E4 // 4].contiguous(), wbo[:, 3 * E4 // 4:].contiguous()])

    def lin(x, codes, scale, bias, group):
        y = next(outs)
        return torch.zeros(codes.shape[0], codes.shape[1], dtype=BF, device=CUDA) if y is None else y
    twin = copy.copy(orig)
    twin.linear = lin
    one = dict(dtype=torch.int64, device=CUDA)
    u = twin.heads(own, torch.zeros(Lw, D, dtype=torch.float16, device=CUDA), torch.zeros(1, **one),
                   torch.zeros(1, **one), torch.zeros(1, 1, device=CUDA), table)
    return u, twin.mix(u)


def _bf16_next(t: torch.Tensor) -> torch.Tensor:
    """The next BF16 value away from zero (one spacing), for near ties."""
    return (t.contiguous().view(torch.int16) + 1).view(BF)


def _adversarial(orig, gen: torch.Generator) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """(name, own FP32 with BF16 values, wbo BF16 (L, 4E)) cases. own = 0 keeps raw = wbo exactly."""
    zero = torch.zeros(L, E, 4)
    rand_own = torch.stack([(torch.rand(L, E, generator=gen) < 0.1).float(), torch.rand(L, E, generator=gen),
                            orig.age_table.cpu()[torch.randint(0, orig.far + 1, (L, E), generator=gen)],
                            (torch.rand(L, E, generator=gen) < 0.5).float()], -1).to(BF).float()
    sign = torch.where(torch.rand(L, 4 * E, generator=gen) < 0.5, -1.0, 1.0)
    mags = torch.tensor([60.0, 100.0, 1000.0, 3e4, 1e30])
    large = sign * mags[torch.randint(0, len(mags), (L, 4 * E), generator=gen)]
    large[0, :8] = 3e38  # cumsum overflow to inf: both sides must report the same non-finite z
    cancel = torch.zeros(L, E, 4)
    a = torch.tensor([64.0, 512.0, 4096.0])[torch.randint(0, 3, (L, E), generator=gen)]
    cancel[..., 0], cancel[..., 1], cancel[..., 2] = -a, a / 2, a / 2
    cancel[..., 3] = torch.randn(L, E, generator=gen) * 4
    vals = torch.tensor([19.875, 20.0, 20.125, -20.0, 0.0, -0.0])
    threshold = vals[torch.randint(0, len(vals), (L, 4 * E), generator=gen)]
    under = -80 - 30 * torch.rand(L, 4 * E, generator=gen)  # exp(raw) subnormal or zero in FP32
    under.view(L, E, 4)[..., 0] = 0.0                       # u_0 = 0 keeps a subnormal softplus visible
    base = torch.randn(L, 4 * E, generator=gen).to(BF)
    ties = base.clone()
    ties[:, 1::2] = _bf16_next(base[:, 0::2])
    bad = torch.randn(L, 4 * E, generator=gen)  # wbo layout: wb at e * 3 + j, wb64 at 3E + e
    bad[0, 3 * 3 + 0] = float('inf')
    bad[1, 5 * 3 + 1] = float('-inf')
    bad[2, 3 * E + 7] = float('nan')
    bad[3, 9 * 3 + 2] = float('inf')

    def wb_layout(v: torch.Tensor) -> torch.Tensor:  # (L, E, 4) [raw0, raw1, raw2, r] -> wbo layout
        return torch.cat([v[..., :3].reshape(L, 3 * E), v[..., 3]], 1)
    return [('large', rand_own, large.to(BF)), ('cancel', zero, wb_layout(cancel).to(BF)),
            ('threshold', zero, threshold.to(BF)), ('underflow', zero, under.to(BF)),
            ('ties', rand_own, ties), ('ties_own0', zero, ties), ('nonfinite', zero, bad.to(BF))]


def _c2_compare(name, s, orig, own, wbo, table) -> dict:
    u_ref, z_ref = _original_readout(orig, own.to(CUDA), wbo.to(CUDA), table)
    x = _x_of(own.to(CUDA))
    keep = torch.empty(L, E, device=CUDA)
    u = torch.empty(L, E, 4, device=CUDA)
    s.bad.zero_()
    s.readout(wbo.to(CUDA).contiguous(), x, keep, u)
    ell = s.ell
    pos = torch.logsumexp(ell + F.logsigmoid(u_ref), -1)
    neg = torch.logsumexp(ell + F.logsigmoid(-u_ref), -1)
    mag = u_ref[..., :1].abs() + (u_ref - u_ref[..., :1])
    tol_u = TOL_REL * mag
    # An infinite u_j (fast u must be infinite there too: the masks are compared) passes no error into z,
    # because logsigmoid(+-inf) is exact; its infinite magnitude bound is therefore left out of tol_z.
    tol_z = TOL_REL * (pos.abs() + neg.abs()) + 2 * torch.where(torch.isfinite(tol_u), tol_u, 0).amax(-1)
    fu, fz = torch.isfinite(u_ref) & torch.isfinite(tol_u), torch.isfinite(z_ref) & torch.isfinite(tol_z)
    du, dz = (u - u_ref).abs(), (keep - z_ref).abs()
    u_over = int((fu & ~(du <= tol_u)).sum())
    z_over = int((fz & ~(dz <= tol_z)).sum())
    order = (z_ref[:, 1:] > z_ref[:, :-1]) != (keep[:, 1:] > keep[:, :-1])
    return dict(case=name, u_exact_share=float(((u == u_ref) | (u.isnan() & u_ref.isnan())).float().mean()),
                z_exact_share=float(((keep == z_ref) | (keep.isnan() & z_ref.isnan())).float().mean()),
                u_over_tol=u_over, z_over_tol=z_over,
                u_finite_masks_equal=bool(torch.equal(torch.isfinite(u), torch.isfinite(u_ref))),
                z_finite_masks_equal=bool(torch.equal(torch.isfinite(keep), torch.isfinite(z_ref))),
                z_nonfinite_ref=int((~torch.isfinite(z_ref)).sum()), bad_count=int(s.bad),
                z_max_abs_diff_finite=float(dz[fz].max()) if fz.any() else 0.0,
                adjacent_order_flips=int(order.sum()))


def test_c2_readout_and_publish():
    fx = _fixtures()
    calls: list = []
    orig, fast, table = _scorers(calls)
    s = fast['blockn']  # readout and publish do not depend on the F8 variant
    rows = []
    n_fixture = 0
    for e in fx['events']:
        if e['kind'] == 'reset':
            orig.reset()
            continue
        orig.advance(e['inputs']['routes'].to(CUDA), torch.tensor([e['step']], dtype=torch.int32, device=CUDA))
        if not e['expect']['valid'] or n_fixture >= 8:
            continue
        own = orig.own(e['inputs']['res'].float().to(CUDA))
        calls.clear()
        orig.heads(own, *_args(e['inputs'], fx['meta']['hidden_layers'], CUDA), table)
        wbo = torch.cat([calls[5][5], calls[6][5]], 1)
        rows.append(_c2_compare(f'fixture{n_fixture}', s, orig, own.to(BF).float(), wbo, table))
        n_fixture += 1
    for name, own, wbo in _adversarial(orig, torch.Generator().manual_seed(23)):
        rows.append(_c2_compare(name, s, orig, own, wbo, table))
    # publish: valid = scored and finite; a scored non-finite token is counted; the count is cleared.
    pub = []
    for bad, step in ((0, 0), (0, 2), (3, 0), (3, 2)):
        valid = torch.full((1,), 7, dtype=torch.int32, device=CUDA)
        nonfinite = torch.full((1,), 5, dtype=torch.int64, device=CUDA)
        s.bad.fill_(bad)
        s.publish(torch.tensor([step], dtype=torch.int32, device=CUDA), valid, nonfinite)
        pub.append(dict(bad=bad, step=step, valid=int(valid), nonfinite=int(nonfinite), bad_after=int(s.bad)))
    _report('c2', rows=rows, publish=pub)
    assert n_fixture == 8
    for r in rows:
        assert r['u_over_tol'] == 0 and r['z_over_tol'] == 0, r
        assert r['u_finite_masks_equal'] and r['z_finite_masks_equal'], r
        assert r['bad_count'] == r['z_nonfinite_ref'], r
    assert [r for r in rows if r['case'] == 'nonfinite'][0]['z_nonfinite_ref'] > 0
    assert [(p['valid'], p['nonfinite'], p['bad_after']) for p in pub] == [(0, 5, 0), (1, 5, 0), (0, 5, 0), (0, 6, 0)]


# --- C3: the fused state kernel on seeded states -----------------------------------------------------------

TINY = 2.0 ** -126  # the smallest normal FP32; below it a flush-to-zero build would differ from torch
# 1e-30 * 0.9^n passes 2^-126 at n = 174 and is 13 subnormal units at n = 300 (FP32, measured on the CPU).
# It never reaches 0: 0.9 k rounds back to k for k <= 4. A flush-to-zero build would give 0 instead.
DECAY_TOKENS = 300


def _seeded_state(far: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(ema, age, routed) on the CPU with the edge values of each field mixed over (L, E)."""
    pool = torch.tensor([0.0, 2.0 ** -149, 1e-40, 2.0 ** -127, TINY - 2.0 ** -149, TINY, TINY * (1 + 2.0 ** -23),
                         1e-30, 0.5, 1.0])
    ema = torch.where(torch.rand(L, E, generator=gen) < 0.7, pool[torch.randint(0, len(pool), (L, E), generator=gen)],
                      torch.rand(L, E, generator=gen))
    ages = torch.tensor([0, 1, far - 2, far - 1, far])
    age = torch.where(torch.rand(L, E, generator=gen) < 0.7, ages[torch.randint(0, len(ages), (L, E), generator=gen)],
                      torch.randint(0, far + 1, (L, E), generator=gen))
    return ema, age.long(), (torch.rand(L, E, generator=gen) < 0.1).float()


def _routes(gen: torch.Generator, fixed: bool = False) -> torch.Tensor:
    """(L, K) int64 route ids; every third layer repeats ids (the kernel and scatter_ must both give 1)."""
    r = torch.arange(K).repeat(L, 1) if fixed else torch.stack([torch.randperm(E, generator=gen)[:K] for _ in range(L)])
    r[::3, K // 2:] = r[::3, :K // 2]
    return r.long()


def test_c3_state_own():
    """state_own against SharedEvict.advance + own (Codex 8b620dc5 item 2): FP32 ema and routed, int64 age
    and the BF16 own features bit for bit, after every token of a seeded sequence (steps with 0 after a
    nonzero state), a 300-token decay through the subnormal range, then reset and a replay that must
    repeat the first run bit for bit."""
    orig, fast, _ = _scorers()
    s = fast['blockn']  # state_own does not depend on the F8 variant
    gen = torch.Generator().manual_seed(31)
    seed = _seeded_state(s.far, gen)
    seq = [(st, _routes(gen), (torch.rand(L, E, generator=gen) < 0.5).float()) for st in (3, 0, 5, 1, 0, 2)]
    decay = [(1, _routes(gen, fixed=True), torch.zeros(L, E))] * DECAY_TOKENS
    stats = dict(tokens=0, mismatches=[], subnormal_ref=0, normal_to_subnormal=0, duplicate_layers=len(range(0, L, 3)))

    def load(m, state):
        for t, v in zip((m.ema, m.age, m.routed), state):
            t.copy_(v.to(CUDA))

    def run(tokens, tag) -> list[tuple[torch.Tensor, ...]]:
        outs = []
        for i, (st, routes, res) in enumerate(tokens):
            step = torch.tensor([st], dtype=torch.int32, device=CUDA)
            routes, res = routes.to(CUDA), res.to(CUDA)
            prev = orig.ema.clone()
            orig.advance(routes, step)
            x_ref = _x_of(orig.own(res))
            x = s.advance_and_own(routes, step, res)
            stats['tokens'] += 1
            if not (torch.equal(s.ema, orig.ema) and torch.equal(s.age, orig.age) and torch.equal(s.routed, orig.routed)
                    and torch.equal(x, x_ref)):
                stats['mismatches'].append((tag, i))
            sub = (orig.ema > 0) & (orig.ema < TINY)
            stats['subnormal_ref'] += int(sub.sum())
            stats['normal_to_subnormal'] += int(((prev >= TINY) & sub).sum())
            if st == 0:
                assert torch.equal(orig.ema, prev), (tag, i)  # the reference itself keeps the state at step 0
            outs.append((s.ema.clone(), s.age.clone(), s.routed.clone(), x.clone()))
        return outs

    load(orig, seed)
    load(s, seed)
    first = run(seq, 'seq')
    run(decay, 'decay')
    orig.reset()
    s.reset()
    reset_ok = (not s.ema.any() and bool(s.age.eq(s.far).all()) and not s.routed.any() and int(s.bad) == 0
                and torch.equal(s.ema, orig.ema) and torch.equal(s.age, orig.age))
    load(orig, seed)
    load(s, seed)
    again = run(seq, 'replay')
    replay_same = all(torch.equal(a, b) for fa, fb in zip(first, again, strict=True) for a, b in zip(fa, fb))
    _report('c3', **{k: v[:20] if isinstance(v, list) else v for k, v in stats.items()}, reset_ok=reset_ok,
            replay_same=replay_same)
    assert not stats['mismatches'], stats['mismatches'][:20]
    assert reset_ok and replay_same
    assert stats['subnormal_ref'] > 0 and stats['normal_to_subnormal'] > 0, stats  # the edge was reached


# --- P1F: graph against eager with the fast hooks ---------------------------------------------------------

def _hooks(fast: str | None, start_layer: int = START, timing: Path | None = None, r36: bool = False):
    from commitmoe_prefetch import bank_loader
    from commitmoe_prefetch import freetoken as ft
    release, _ = _release()
    rule = dict(evict=(_pinned(R36, 'EXP526_EVICT_BUNDLE_SHA256'), start_layer)) if r36 else \
        dict(shared=(BUNDLE, start_layer), shared_fast=fast)
    cfg, state = bank_loader.load(PUBLISHED)
    hooks = ft.PrefetchHooks(cfg, state, demand_stripes=16, timing=timing, release_sha256=release, **rule)
    before = torch.cuda.memory_allocated()
    hooks.reserve(SimpleNamespace(device=CUDA))
    reserved = torch.cuda.memory_allocated() - before
    cache, hosts = _cache()
    hooks.bind(None, cache)
    assert cache.learned_evict is hooks.learned
    return hooks, cache, hosts, reserved


def _fast_twin(s, ema, age, routed):
    """An eager copy of fast scorer ``s``: the same weights, its own state and scratch."""
    twin = copy.copy(s)
    twin.ema, twin.age, twin.routed = ema.clone(), age.clone(), routed.clone()
    twin.x, twin.bad = s.x.clone(), torch.zeros_like(s.bad)
    twin.partial = {k: v.clone() for k, v in s.partial.items()}
    return twin


def _prior_once(prior_before: torch.Tensor, curr: torch.Tensor, beta: float, step: int) -> torch.Tensor:
    """The prior after one advance of freetoken._score (unchanged at step 0)."""
    p = prior_before.clone()
    if step > 0:
        p.mul_(beta)
        p.scatter_(1, curr, p.gather(1, curr) + (1 - beta))
    return p


@pytest.mark.parametrize('split', [False, True])
def test_p1f_graph_equals_eager(split):
    from freetoken.moe.learned_evict import LearnedEvict, learned_ensure
    variant = _variant()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    hooks, cache, hosts, reserved = _hooks(variant)
    peak_build = torch.cuda.max_memory_allocated()
    s, rule = hooks.shared, hooks.learned
    inputs = _Inputs()
    alloc0, pool0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    graph = _capture(hooks, cache, inputs, split=split)
    torch.cuda.synchronize()
    capture_alloc, capture_reserved = torch.cuda.memory_allocated() - alloc0, torch.cuda.memory_reserved() - pool0
    generator = torch.Generator().manual_seed(7)
    table = hooks.bank.buffers['table']
    beta = hooks.cfg.features.ema_beta
    for uid in (1, 2):
        hooks.begin_request(uid)
        assert int(rule.valid) == 0 and not s.ema.any() and s.age.eq(s.far).all() and not s.routed.any()
        assert int(s.bad) == 0
        shadow = _shadow(cache)
        shadow_rule = LearnedEvict(keep=rule.keep.clone(), valid=rule.valid.clone(),
                                   target_of_layer=rule.target_of_layer, start_layer=START,
                                   counters=torch.zeros_like(rule.counters))
        for step in range(STEPS):
            inputs.draw(generator)
            state = (s.ema.clone(), s.age.clone(), s.routed.clone())
            resident = (cache.slot_for_id >= 0).float()
            prior_before = hooks.raw['prior_before'].clone()
            hooks.before_decode(step)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(hooks.resident, resident), (uid, step)
            twin = _fast_twin(s, *state)
            x = twin.advance_and_own(hooks.score_inputs['routes'], hooks.step, resident)
            keep = torch.empty_like(rule.keep)
            twin.readout(twin.trunk(x, hooks.score_inputs['h'], hooks.raw['tid'], hooks.raw['upcoming_tid'],
                                    hooks.raw['pos'][:1], table), x, keep)
            assert torch.equal(rule.keep, keep), (uid, step)
            assert torch.equal(s.ema, twin.ema) and torch.equal(s.age, twin.age), (uid, step)
            assert torch.equal(s.routed, twin.routed) and torch.equal(s.x, twin.x), (uid, step)
            assert bool(torch.isfinite(keep).all()) and int(rule.valid) == int(step > 0), (uid, step)
            assert int(s.bad) == 0, (uid, step)
            if step == 0:
                assert torch.equal(s.ema, state[0]) and torch.equal(s.age, state[1])
            g = hooks.bank.trunk({**hooks.raw, 'prior_before': prior_before})
            assert torch.equal(hooks.probs, hooks.bank.head(g)), (uid, step)
            expect_prior = _prior_once(prior_before, hooks.raw['curr'], beta, step)
            assert torch.equal(hooks.raw['prior_before'], expect_prior), (uid, step)
            shadow_rule.keep.copy_(rule.keep)
            shadow_rule.valid.copy_(rule.valid)
            for layer in range(L):
                learned_ensure(shadow, layer, inputs.ids[layer].clone(), shadow_rule)
            for name in ('slot_for_id', 'id_of_slot', 'usage', 'step'):
                assert torch.equal(getattr(shadow, name), getattr(cache, name)), (uid, step, name)
            assert _check_resident_rows(cache, hosts) > 0
            hooks.after_decode(step, torch.tensor([100 + step], device=CUDA), torch.tensor([200 + step], device=CUDA))
        assert torch.equal(rule.counters, shadow_rule.counters)
        events, reranked, rescued = rule.counters.tolist()
        assert events > 0 and reranked > 0, rule.counters
    scratch = sum(t.numel() * t.element_size() for t in s.partial.values())
    _report('p1f', variant=variant, split=split, requests=2, steps=STEPS,
            evict_counters_last_request=rule.counters.tolist(), nonfinite_tokens=int(hooks.nonfinite_tokens),
            reserve_allocated_mib=reserved / 2**20, shared_evict_mib=s.nbytes / 2**20,
            splitk_scratch_bytes=scratch, build_peak_allocated_mib=peak_build / 2**20,
            capture_allocated_mib=capture_alloc / 2**20, capture_reserved_mib=capture_reserved / 2**20)
    assert int(hooks.nonfinite_tokens) == 0


# --- P2F: the P2 gates with the fast scorer ---------------------------------------------------------------

def test_p2f_scores_and_victims():
    from freetoken.moe.learned_evict import learned_ensure
    variant = _variant()
    fx = _fixtures()
    _, fast, table = _scorers()
    s = fast[variant]
    rows = []
    keep = torch.empty(L, E, device=CUDA)
    valid = torch.zeros(1, dtype=torch.int32, device=CUDA)
    nonfinite = torch.zeros(1, dtype=torch.int64, device=CUDA)
    for e in fx['events']:
        if e['kind'] == 'reset':
            s.reset()
            continue
        x = e['inputs']
        step = torch.tensor([e['step']], dtype=torch.int32, device=CUDA)
        s.score(x['routes'].to(CUDA), step, x['res'].float().to(CUDA),
                *_args(x, fx['meta']['hidden_layers'], CUDA), table, keep, valid, nonfinite)
        assert int(valid) == int(bool(e['expect']['valid'])), e['step']
        if e['expect']['valid']:
            rows.append((keep.cpu(), e['expect']['z'], x['res'], x['routes']))
    finite = all(bool(torch.isfinite(z).all()) for z, _, _, _ in rows)
    overlap = torch.stack([bottom_overlap(z, ze, res) for z, ze, res, _ in rows])
    diff = torch.stack([z - ze for z, ze, _, _ in rows])
    q = torch.quantile(diff.abs().flatten(), torch.tensor([0.5, 0.99, 0.999]))
    # The victim interface, as test_gpu_shared_evict.test_p2_scores_and_victims.
    cache, _ = _cache()
    base = _shadow(cache)
    warm = _Inputs()
    generator = torch.Generator().manual_seed(11)
    lru = _rule(torch.zeros(L, E, device=CUDA), 0)
    ev, agree, reranked, reranked_agree = (torch.zeros(L, dtype=torch.int64) for _ in range(4))
    for i, (z, ze, _, _) in enumerate(rows):
        warm.draw(generator)
        for layer in range(L):
            learned_ensure(base, layer, warm.ids[layer].clone(), lru)
        nxt = rows[(i + 1) % len(rows)][3]
        a, b, c = _shadow(base), _shadow(base), _shadow(base)
        ra, rb = _rule(z.to(CUDA), 1), _rule(ze.to(CUDA), 1)
        for layer in range(L):
            ids = nxt[layer].to(CUDA).int().view(1, -1)
            before = a.slot_for_id.clone()
            learned_ensure(a, layer, ids.clone(), ra)
            learned_ensure(b, layer, ids.clone(), rb)
            learned_ensure(c, layer, ids.clone(), lru)
            gone_a, gone_b, gone_c = ((before >= 0) & (m.slot_for_id < 0) for m in (a, b, c))
            ev[layer] += int(gone_a.sum())
            agree[layer] += int((gone_a & gone_b).sum())
            reranked[layer] += int((gone_a & ~gone_c).sum())
            reranked_agree[layer] += int((gone_a & ~gone_c & gone_b).sum())
            _follow(b, a)
            _follow(c, a)
    total = int(ev.sum())
    values = dict(variant=variant, rows=len(rows), finite=finite, overlap_mean=float(overlap.mean()),
                  overlap_min=float(overlap.min()), layer_min=float(overlap.mean(0).min()),
                  layer_min_index=int(overlap.mean(0).argmin()), bad_layer_rows=int((overlap < 1).sum()),
                  layer_rows=overlap.numel(), abs_diff_q50_q99_q999=q.tolist(), abs_diff_max=float(diff.abs().max()),
                  signed_mean=float(diff.mean()), evictions=total, victim_same_share=int(agree.sum()) / max(total, 1),
                  reranked=int(reranked.sum()),
                  reranked_same_share=int(reranked_agree.sum()) / max(int(reranked.sum()), 1),
                  evictions_by_layer=ev.tolist(), disagreements_by_layer=(ev - agree).tolist(),
                  reranked_by_layer=reranked.tolist(), nonfinite_tokens=int(nonfinite))
    _report('p2f', **values)
    assert finite and len(rows) == sum(e['expect']['valid'] for e in fx['events']) > 0, values
    assert total > 0 and int(reranked.sum()) > 0, values
    assert values['overlap_mean'] >= OVERLAP_GATE, values
    assert values['victim_same_share'] >= VICTIM_GATE, values


# --- F1: asynchronous lifetime of the post-ready raw copies -----------------------------------------------

class _Draws:
    """Per-token decode inputs, all on the device, so that feeding a token needs no host copy (a pageable
    host-to-device copy could synchronize and hide a race)."""

    def __init__(self, tokens: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.hidden = torch.randn(tokens, L, 1, D, generator=g).to(BF).to(CUDA)
        self.logits = torch.randn(tokens, L, 1, E, generator=g).to(BF).to(CUDA)
        self.ids = torch.stack([torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(L)])
                                for _ in range(tokens)])[:, :, None].int().to(CUDA)
        self.consumed = torch.randint(0, 1000, (tokens, 1), generator=g).to(CUDA)
        self.sampled = torch.randint(0, 1000, (tokens, 1), generator=g).to(CUDA)

    def feed(self, inputs: _Inputs, i: int) -> None:
        inputs.hidden.copy_(self.hidden[i])
        inputs.logits.copy_(self.logits[i])
        inputs.ids.copy_(self.ids[i])


def _sequence(monkeypatch, fast: str | None, split: bool, graph_mode: bool, sync: bool,
              no_join: bool = False) -> list[dict]:
    """Two requests of STEPS tokens; per token, the saved outputs (cloned on main after the decode)."""
    hooks, cache, _, _ = _hooks(fast)
    if fast is not None:
        real = hooks._copy_raw

        def delayed() -> None:  # runs on the probe stream after scores-ready (F1)
            torch.cuda._sleep(DELAY_CYCLES)
            real()
        monkeypatch.setattr(hooks, '_copy_raw', delayed)
    if no_join:
        join = torch.cuda.Stream.wait_stream
        main = torch.cuda.current_stream()

        def wait_stream(self, stream):  # the mutation: main never joins the probe stream
            if self == main and stream == hooks.probe_stream:
                return None
            return join(self, stream)
        monkeypatch.setattr(torch.cuda.Stream, 'wait_stream', wait_stream)
    inputs = _Inputs()
    draws = _Draws(2 * STEPS, seed=29)
    graph = _capture(hooks, cache, inputs, split=split) if graph_mode else None
    out = []
    i = 0
    for uid in (1, 2):
        hooks.begin_request(uid)  # request 2 starts right after request 1's last token, no sync
        for step in range(STEPS):
            draws.feed(inputs, i)
            hooks.before_decode(step)
            if graph is not None:
                graph.replay()
            else:
                _decode(hooks, cache, inputs, split=split)
            if sync:
                torch.cuda.synchronize()
            saved = {f'raw.{k}': v.clone() for k, v in hooks.raw.items()}
            saved.update(keep=hooks.learned.keep.clone(), valid=hooks.learned.valid.clone(),
                         nonfinite=hooks.nonfinite_tokens.clone(), ema=hooks.shared.ema.clone(),
                         age=hooks.shared.age.clone(), probs=hooks.probs.clone(),
                         slot_for_id=cache.slot_for_id.clone())
            out.append(saved)
            hooks.after_decode(step, draws.consumed[i], draws.sampled[i])
            if sync:
                torch.cuda.synchronize()
            i += 1
    torch.cuda.synchronize()
    monkeypatch.undo()
    return [{k: v.cpu() for k, v in s.items()} for s in out]


def _mismatches(a: list[dict], b: list[dict], keys) -> list[tuple[int, str]]:
    return [(i, k) for i, (x, y) in enumerate(zip(a, b, strict=True)) for k in keys if not torch.equal(x[k], y[k])]


@pytest.mark.parametrize('graph_mode', [True, False])
@pytest.mark.parametrize('split', [False, True])
def test_f1_async_lifetime(monkeypatch, split, graph_mode):
    variant = _variant()
    ref = _sequence(monkeypatch, variant, split, graph_mode, sync=True)
    run = _sequence(monkeypatch, variant, split, graph_mode, sync=False)
    original = _sequence(monkeypatch, None, split, graph_mode, sync=True)
    keys = list(ref[0])
    bank_keys = [k for k in keys if k.startswith('raw.')] + ['probs']
    async_diff = _mismatches(ref, run, keys)
    original_diff = _mismatches(ref, original, bank_keys)
    values = dict(variant=variant, split=split, graph=graph_mode, tokens=len(ref), async_mismatches=async_diff[:20],
                  original_bank_mismatches=original_diff[:20],
                  valid=[int(r['valid']) for r in run])
    if not graph_mode:
        mutant = _sequence(monkeypatch, variant, split, False, sync=False, no_join=True)
        values['mutation_mismatches'] = len(_mismatches(ref, mutant, keys))
    _report('f1', **values)
    assert not async_diff and not original_diff, values
    assert values['valid'] == [int(s > 0) for s in range(STEPS)] * 2, values
    if not graph_mode:
        assert values['mutation_mismatches'] > 0, 'removing the probe join did not change the result'


# --- B2: component timings (step B) -----------------------------------------------------------------------

def _tensors(r) -> list[torch.Tensor]:
    if isinstance(r, torch.Tensor):
        return [r]
    return [t for x in r for t in _tensors(x)] if isinstance(r, (tuple, list)) else []


def _stage_graph(body, stamps: torch.Tensor) -> tuple[torch.cuda.CUDAGraph, list[torch.Tensor]]:
    """(graph of stamp, body, stamp; the tensors that body returned in the capture)."""
    from commitmoe_prefetch import kernels

    def stamped():
        kernels.stamp[(1,)](stamps, IDX=0)
        r = body()
        kernels.stamp[(1,)](stamps, IDX=1)
        return r
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        stamped()  # eager warm-up compiles every kernel before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        r = stamped()
    return graph, _tensors(r)


def _stage_block(graph, stamps, flush, outputs) -> list[int]:
    out = []
    for i in range(WARMUP + TIMED):
        flush.fill_(i & 0xFF)
        stamps.zero_()
        graph.replay()
        torch.cuda.synchronize()
        t0, t1 = stamps.tolist()
        assert 0 < t0 <= t1, (t0, t1)
        if i >= WARMUP:
            out.append(t1 - t0)
    for t in outputs:
        assert bool(torch.isfinite(t.float()).all()), 'a stage output is not finite'
    return out


def _stats(v: list[int]) -> dict:
    t = torch.tensor(v, dtype=torch.float64)
    return dict(median_us=float(t.median()) / 1e3, p90_us=float(torch.quantile(t, 0.9)) / 1e3,
                max_us=float(t.max()) / 1e3, replays=len(v))


def test_b2_component_bench():
    """Original against fast stages, cold L2. Reported only; the runner applies the F8 rule to the pooled
    'scorer' medians. Setup (a prospective change from plan revision 0, which named a2's inputs): every
    stage runs on the first valid fixture event with step 1, the same routes on every replay; the 'state'
    and 'scorer' arms advance their own state on each replay. These times compare implementations of
    the same work; they are not a2 and establish nothing about the 150 us gate."""
    from commitmoe_prefetch.qbank import qlinear
    fx = _fixtures()
    calls: list = []
    orig, fast, table = _scorers(calls)
    ev = next(e for e in fx['events'] if e['kind'] != 'reset' and e['expect']['valid'])
    x_in = ev['inputs']
    routes, res = x_in['routes'].to(CUDA), x_in['res'].float().to(CUDA)
    step = torch.ones(1, dtype=torch.int32, device=CUDA)
    args = _args(x_in, fx['meta']['hidden_layers'], CUDA)
    orig.advance(routes, step)
    own = orig.own(res)
    calls.clear()
    orig.heads(own, *args, table)
    stage_in = {n: calls[i][0] for i, n in enumerate(('w_layer', 'h_proj', 'fc1', 'fc2', 'we', 'wb'))}
    t, g = orig.t, orig.group
    keep = torch.zeros(L, E, device=CUDA)
    valid = torch.zeros(1, dtype=torch.int32, device=CUDA)
    nonfinite = torch.zeros(1, dtype=torch.int64, device=CUDA)
    wbo = torch.cat([calls[5][5], calls[6][5]], 1)
    x_own = _x_of(own)

    def lin(name, gelu=False):
        y = qlinear(stage_in[name], t[name + '_q'], t[name + '_s'], t.get(name + '_b'), g)
        return F.gelu(y) if gelu else y

    def orig_state():
        orig.advance(routes, step)
        return _x_of(orig.own(res))

    def orig_score():
        orig.advance(routes, step)
        keep.copy_(orig.scores(orig.own(res), *args, table))
        finite = torch.isfinite(keep).all()
        valid.copy_(((step[0] > 0) & finite).to(torch.int32).view(1))
        nonfinite.add_(((step[0] > 0) & ~finite).to(torch.int64))

    def fast_stage(v, name, gelu=False):
        s = fast[v]
        shared = name == 'we'
        x = stage_in[name][:1].contiguous() if shared else stage_in[name]
        return lambda: s.matvec(x, name, gelu=gelu, shared_row=shared)

    stages = {
        'state': dict(original=orig_state, fast=lambda: fast['blockn'].advance_and_own(routes, step, res)),
        'w_layer': dict(original=lambda: lin('w_layer'), fast=fast_stage('blockn', 'w_layer')),
        'h_proj': dict(original=lambda: lin('h_proj'), i=fast_stage('blockn', 'h_proj'),
                       ii=fast_stage('splitk', 'h_proj')),
        'fc1_gelu': dict(original=lambda: lin('fc1', True), i=fast_stage('blockn', 'fc1', True),
                         ii=fast_stage('splitk', 'fc1', True)),
        'fc2_gelu': dict(original=lambda: lin('fc2', True), fast=fast_stage('blockn', 'fc2', True)),
        'we_gelu': dict(original=lambda: F.gelu(qlinear(stage_in['we'], t['we_q'], t['we_s'], t['we_b'], g)),
                        fast=fast_stage('blockn', 'we', True)),
        'wb_wb64': dict(original=lambda: (lin('wb'), qlinear(stage_in['wb'], t['wb64_q'], t['wb64_s'],
                                                             t['wb64_b'], g)),
                        fast=lambda: fast['blockn'].matvec(stage_in['wb'], 'wbx')),
        # No original arm: heads() cannot run its tail alone without extra trunk work, so an original
        # readout time would be biased; EXP585P's stamped intervals stand for it, and 'scorer' compares
        # the full original path with the full fast path.
        'readout': dict(fast=lambda: (fast['blockn'].readout(wbo, x_own, keep),
                                      fast['blockn'].publish(step, valid, nonfinite))),
        'scorer': dict(original=orig_score,
                       i=lambda: fast['blockn'].score(routes, step, res, *args, table, keep, valid, nonfinite),
                       ii=lambda: fast['splitk'].score(routes, step, res, *args, table, keep, valid, nonfinite)),
    }
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=CUDA)
    results, blocks = {}, {}
    for name, arms in stages.items():
        stamps = torch.zeros(2, dtype=torch.int64, device=CUDA)
        graphs = {arm: _stage_graph(body, stamps) for arm, body in arms.items()}
        names = list(arms)
        times = {arm: [] for arm in names}
        blocks[name] = []
        for order, arm in enumerate(names + names[::-1]):  # original/fast/fast/original or original/i/ii/ii/i/original
            graph, outputs = graphs[arm]
            v = _stage_block(graph, stamps, flush, outputs + [keep])
            times[arm] += v
            blocks[name].append(dict(order=order, arm=arm, **_stats(v)))
        results[name] = {arm: _stats(v) for arm, v in times.items()}  # pooled over the arm's blocks
        del graphs
    _report('b2', stages=results, blocks=blocks, nonfinite_tokens=int(nonfinite),
            setup='first valid fixture event, step 1, the same routes each replay (scorer arms advance their '
                  'state every replay); not the a2 START token, so no a2 or 150 us conclusion')
    assert int(nonfinite) == 0
    assert all(r['replays'] == 2 * TIMED for arms in results.values() for r in arms.values())


# --- a2: the unchanged gate, with a separate diagnostic graph first ---------------------------------------

class _StampedProbs:
    """Stands in for hooks.probs inside the diagnostic graph only: the copy, then a stamp."""

    def __init__(self, probs: torch.Tensor, stamps: torch.Tensor, idx: int):
        self.probs, self.stamps, self.idx = probs, stamps, idx

    def copy_(self, src: torch.Tensor) -> torch.Tensor:
        from commitmoe_prefetch import kernels
        self.probs.copy_(src)
        kernels.stamp[(1,)](self.stamps, IDX=self.idx)
        return self.probs


def _diag_graph(hooks, cache, snapshot: bool) -> torch.cuda.CUDAGraph:
    """_prefix_graph's body plus two stamps: probs written (column L + 1) and the end of _score after the
    prior update (column L + 2). Not the gate graph."""
    from commitmoe_prefetch import kernels
    hooks.step.fill_(1)
    probs = hooks.probs
    hooks.probs = _StampedProbs(probs, hooks.stamps, L + 1)

    def body():
        kernels.stamp[(1,)](hooks.stamps, IDX=0)
        if snapshot:
            hooks.resident.copy_(cache.slot_for_id >= 0)
        hooks._score()
        kernels.stamp[(1,)](hooks.stamps, IDX=L + 2)
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
    finally:
        hooks.probs = probs
    return graph


def _diag_block(graph, hooks, flush) -> dict[str, list[int]]:
    out = dict(ready=[], probs=[], end=[])
    for i in range(WARMUP + TIMED):
        flush.fill_(i & 0xFF)
        hooks.stamps.zero_()
        graph.replay()
        torch.cuda.synchronize()
        s = hooks.stamps.cpu()
        obs, ready, probs, end = int(s[0]), int(s[L]), int(s[L + 1]), int(s[L + 2])
        assert 0 < obs <= ready <= probs <= end, (obs, ready, probs, end)
        if i >= WARMUP:
            out['ready'].append(ready - obs)
            out['probs'].append(probs - obs)
            out['end'].append(end - obs)
    return out


def _median_us(v: list[int]) -> float:
    return float(torch.tensor(v, dtype=torch.float64).median()) / 1e3


def test_a2_cold_l2_prefix_gate(tmp_path):
    variant = _variant()
    base, base_cache, _, base_mem = _hooks(None, timing=tmp_path / 'b.pt', r36=True)
    cand, cand_cache, _, cand_mem = _hooks(variant, timing=tmp_path / 'c.pt')
    hooks = dict(base=base, cand=cand)
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=CUDA)
    diag = dict(base=_diag_graph(base, base_cache, False), cand=_diag_graph(cand, cand_cache, True))
    d = {arm: dict(ready=[], probs=[], end=[]) for arm in hooks}
    for arm in ('base', 'cand', 'cand', 'base'):
        for k, v in _diag_block(diag[arm], hooks[arm], flush).items():
            d[arm][k] += v
    del diag
    graphs = dict(base=_prefix_graph(base, base_cache, False), cand=_prefix_graph(cand, cand_cache, True))
    times = dict(base=[], cand=[])
    for arm in ('base', 'cand', 'cand', 'base'):
        times[arm] += _block(graphs[arm], hooks[arm], flush)
    for arm, h in hooks.items():  # a2's synthetic token: scored, finite, valid
        assert bool(torch.isfinite(h.learned.keep).all()) and int(h.learned.valid) == 1, arm
    t = {arm: torch.tensor(v, dtype=torch.float64) for arm, v in times.items()}
    values = {f'{arm}_{k}_us': float(fn(t[arm])) / 1e3 for arm in t
              for k, fn in (('median', torch.median), ('p90', lambda v: torch.quantile(v, 0.9)), ('max', torch.max))}
    values.update(variant=variant, replays_per_arm=len(times['cand']), shared_evict_mib=cand.shared.nbytes / 2**20,
                  reserve_allocated_mib=dict(base=base_mem / 2**20, cand=cand_mem / 2**20),
                  diagnostic={arm: {f'{k}_median_us': _median_us(v) for k, v in d[arm].items()} for arm in d},
                  diagnostic_suffix_median_us={arm: _median_us([e - r for e, r in zip(d[arm]['end'], d[arm]['ready'])])
                                               for arm in d})
    _report('a2_cold_l2', **values)
    assert len(times['cand']) == len(times['base']) == 2 * TIMED
    assert float(t['cand'].median()) <= A2_GATE_NS, values
