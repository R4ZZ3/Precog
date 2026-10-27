"""Shared learned eviction (EXP585): keep scores of all layers from R584's own trunk and heads.

A bundle (experiments/exp585_export.py, format FORMAT) holds R584, the EXP584 eviction model: the frozen
J_s573 trunk and a 40-layer head with cumulative next-use logits for 8, 16, 32 and 64 tokens, mixed per
layer by mix_alpha into one signed keep score z (higher = keep, the victim rule evicts the lowest). Unlike
EvictRows it does not read the prefetch bank's trunk; it reads only the bank's BF16 token table.

Per decoded token, from the inputs that exist at the end of token t (all on the device, fixed shapes):
  state     ema = ema * beta + routed * (1 - beta) and age = 0 if routed else min(age + 1, far), for every
            layer and expert, from ema 0 and age far at the request start; advanced once per scored token
            and only when step > 0 (step 0 has no decoded token). The prefetch prior is separate.
  own       [routed, ema, agef, resident] per layer and expert, agef = AGE_TABLE[age] = log2(1 + age) / 8;
            resident is the slot map snapshot (slot >= 0) taken on main at the end of token t.
  trunk     z_L = w_layer_L x_L for x_L = routed | ema | agef | resident; parts = [z, table[tid],
            table[upcoming], pos, h_proj_j(layer_norm(h at hidden_layers[j]))]; c = gelu(fc2(gelu(fc1(parts)))).
  head      he = gelu(we c + we_b); cumulative P8/P16/P32 = cumsum(raw_0, softplus(raw_1), softplus(raw_2))
            for raw = wb he + wb_b + own @ wide; P64 = P32 + softplus(wb64 he + wb64_b + own @ wide64).
  readout   z = logsumexp(ell + logsigmoid(u)) - logsumexp(ell + logsigmoid(-u)), ell = log_softmax(alpha),
            in FP32 (experiments/exp584_rank.mix_logit, the stable log-domain form; no sigmoid or clip).
Arithmetic: every matrix product is ``linear`` (qbank.qlinear on the GPU: BF16 input, int8 codes with FP16
group scales, FP32 accumulation and bias, BF16 output). The skip products own @ wide use BF16-rounded
operands, FP32 products summed in feature order, and a BF16-rounded result, the values training's autocast
einsum gave; gelu reads and writes BF16. ``reference_linear`` is the same contract in plain torch for CPU
tests and fixtures; it is not bit-equal to qgemv (the FP32 sums run in another order).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

FORMAT = 'exp585_shared_evict_v1'
# bundle matrix -> its bias (None: no bias); the exporter's MATRICES/BIASES tables
MATRICES = dict(w_layer=None, h_proj0='h_proj0_b', h_proj1='h_proj1_b', fc1='fc1_b', fc2='fc2_b',
                we='we_b', wb='wb_b', wb64='wb64_b')
KEPT = ('wide', 'wide64', 'mix_alpha')
GEOMETRY = ('n_layer', 'n_exp', 'n_feat', 'layer_dim', 'hidden_dim', 'hidden_proj', 'trunk_hidden', 'trunk_dim',
            'evict_hidden', 'tid_emb_dim', 'token_table_rows', 'hidden_layers', 'ema_beta', 'far', 'n_tok',
            'horizons')
Linear = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, 'torch.Tensor | None', int], torch.Tensor]


def schema(f: dict, group: int) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Name -> (dtype, shape) of every bundle tensor for geometry ``f`` (exp585_export.bundle_schema)."""
    L, E, Fe, H = f['n_layer'], f['n_exp'], f['n_feat'], len(f['horizons'])
    n_h = len(f['hidden_layers'])
    fc1_in = L * f['layer_dim'] + 2 * f['tid_emb_dim'] + 1 + n_h * f['hidden_proj']
    hnk = dict(w_layer=(L, f['layer_dim'], Fe * E), fc1=(1, f['trunk_hidden'], fc1_in),
               fc2=(1, f['trunk_dim'], f['trunk_hidden']), we=(L, f['evict_hidden'], f['trunk_dim']),
               wb=(L, E * (H - 1), f['evict_hidden']), wb64=(L, E, f['evict_hidden']))
    for j in range(n_h):
        hnk[f'h_proj{j}'] = (1, f['hidden_proj'], f['hidden_dim'])
    if set(hnk) != set(MATRICES):
        raise ValueError(f'shared eviction: {n_h} hidden projections; the bundle format has h_proj0 and h_proj1')
    out = {}
    for name, (Hh, N, K) in hnk.items():
        out[name + '_q'] = (torch.int8, (Hh, N, K))
        out[name + '_s'] = (torch.float16, (Hh, N, -(-K // group)))
        if MATRICES[name] is not None:
            out[MATRICES[name]] = (torch.float32, (Hh, N))
    out.update(wide=(torch.float32, (L, Fe, H - 1)), wide64=(torch.float32, (L, Fe)),
               mix_alpha=(torch.float32, (L, H)))
    return out


def age_table(far: int) -> torch.Tensor:
    """agef of every age 0..far, (far + 1,) FP32 on the CPU: the expression training used. The scorer copies
    these bytes to its device and the P0 fixtures read the same table; log2 on another device can differ."""
    return torch.log2(1 + torch.arange(far + 1).float()) / 8


def load_bundle(path: Path, release_sha256: str | None, num_layers: int, num_experts: int, top_k: int,
                hidden_size: int, table: torch.Tensor, position_scale: int,
                bank_group: int) -> tuple[dict, dict[str, torch.Tensor]]:
    """(meta, CPU tensors) of a checked bundle; raises ValueError listing every problem.

    ``release_sha256`` is the served bank's model.safetensors hash: the scorer reads that bank's token table,
    so a bundle exported against another release is refused. ``table`` is the served table (shape only).
    """
    b = torch.load(path, map_location='cpu', weights_only=True)
    meta, t = b.get('meta', {}), b.get('tensors', {})
    if not isinstance(meta, dict) or not isinstance(t, dict) or meta.get('format') != FORMAT:
        raise ValueError(f'shared eviction bundle {path}: format {meta.get("format") if isinstance(meta, dict) else None}, '
                         f'expected {FORMAT}')
    f, group, problems = meta.get('features', {}), meta.get('group'), []
    if missing := [k for k in GEOMETRY if k not in f]:
        raise ValueError(f'shared eviction bundle {path}: features without {missing}')
    if group != bank_group:
        problems.append(f'INT8 group {group}, the bank uses {bank_group}')
    if release_sha256 is not None and meta.get('release_sha256') != release_sha256:
        problems.append(f'exported against release {str(meta.get("release_sha256"))[:12]}, '
                        f'served release is {release_sha256[:12]}')
    if (f['n_layer'], f['n_exp']) != (num_layers, num_experts):
        problems.append(f'{f["n_layer"]} x {f["n_exp"]} experts, the model has {num_layers} x {num_experts}')
    if f['hidden_dim'] != hidden_size:
        problems.append(f'hidden width {f["hidden_dim"]}, the model has {hidden_size}')
    if f['n_feat'] != 4 or list(f['horizons']) != [8, 16, 32, 64]:
        problems.append(f'features {f["n_feat"]} and horizons {f["horizons"]}: the scorer implements 4 and 8/16/32/64')
    if tuple(table.shape) != (f['token_table_rows'], f['tid_emb_dim']) or table.dtype != torch.bfloat16:
        problems.append(f'served table {table.dtype} {tuple(table.shape)}, bundle expects bfloat16 '
                        f'{(f["token_table_rows"], f["tid_emb_dim"])}')
    if f['n_tok'] != position_scale:
        problems.append(f'position scale {position_scale}, the bundle was trained with (t + 1) / {f["n_tok"]}')
    if not (0 < f['ema_beta'] < 1) or not (isinstance(f['far'], int) and 0 < f['far'] < 2**15):
        problems.append(f'ema_beta {f["ema_beta"]} or far {f["far"]} out of range')
    if bad := [L for L in f['hidden_layers'] if not 0 <= L < num_layers]:
        problems.append(f'hidden layers {bad} are not layers 0..{num_layers - 1}')
    if top_k < 1:
        problems.append(f'top_k {top_k}')
    if problems:
        raise ValueError(f'shared eviction bundle {path}: ' + '; '.join(problems))
    want = schema(f, group)
    got = {k: (v.dtype, tuple(v.shape)) for k, v in t.items() if isinstance(v, torch.Tensor)}
    if got != want or len(got) != len(t):
        diff = sorted(k for k in set(want) | set(t) if want.get(k) != got.get(k))
        raise ValueError(f'shared eviction bundle {path}: tensors differ from the schema at {diff}')
    for k, v in t.items():
        # qgemv indexes codes and scales as C-contiguous arrays; a same-shape transposed view would be misread.
        if not v.is_contiguous():
            problems.append(f'{k} is not C-contiguous (stride {v.stride()})')
        elif v.dtype == torch.int8:
            if int(v.min()) < -127:
                problems.append(f'{k} has code -128 (the quantizer clamps to -127..127)')
        elif not bool(torch.isfinite(v.float()).all()):
            problems.append(f'{k} non-finite')
        elif k.endswith('_s') and not bool((v > 0).all()):
            problems.append(f'{k} has a scale <= 0')
    if problems:
        raise ValueError(f'shared eviction bundle {path}: ' + '; '.join(problems))
    return meta, t


def reference_linear(x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None,
                     group: int) -> torch.Tensor:
    """qgemv's contract in plain torch: (H, K) x int8 (H, N, K) -> (H, N) BF16 (FP32 sums and bias)."""
    H, N, K = codes.shape
    assert x.shape == (H, K), (tuple(x.shape), H, K)
    w = codes.float() * scale.float().repeat_interleave(group, -1)[..., :K]
    y = torch.einsum('hk,hnk->hn', x.to(torch.bfloat16).float(), w)
    return (y if bias is None else y + bias.float()).to(torch.bfloat16)


def skip(own_bf: torch.Tensor, w_bf: torch.Tensor) -> torch.Tensor:
    """BF16(sum_f own[l, e, f] * w[l, f, ...]) with FP32 products added in feature order.

    own_bf (L, E, F) and w_bf (L, F, ...) hold BF16-rounded values in FP32, so each product is exact and
    the explicit order makes the CPU reference and the GPU port add the same numbers in the same order."""
    extra = w_bf.dim() - 2
    acc = own_bf[:, :, 0].view(*own_bf.shape[:2], *([1] * extra)) * w_bf[:, 0].unsqueeze(1)
    for k in range(1, own_bf.shape[2]):
        acc = acc + own_bf[:, :, k].view(*own_bf.shape[:2], *([1] * extra)) * w_bf[:, k].unsqueeze(1)
    return acc.to(torch.bfloat16).float()


class SharedEvict:
    """R584 on ``device``: the eviction state and the scorer. ``linear`` is qbank.qlinear for the served
    path or ``reference_linear``; the bundle's tensors are moved to ``device`` once."""

    def __init__(self, meta: dict, tensors: dict[str, torch.Tensor], device: torch.device, linear: Linear):
        f = meta['features']
        self.f, self.group, self.linear = f, meta['group'], linear
        L, E = f['n_layer'], f['n_exp']
        self.beta, self.far = float(f['ema_beta']), int(f['far'])
        self.hidden_layers = torch.tensor(list(f['hidden_layers']), dtype=torch.int64, device=device)
        t = {k: v.to(device).contiguous() for k, v in tensors.items()}
        # The hidden projections share one launch: same shape, stacked as heads.
        n_h = len(f['hidden_layers'])
        for part in ('_q', '_s', '_b'):
            t['h_proj' + part] = torch.cat([t.pop(f'h_proj{j}{part}') for j in range(n_h)])
        self.t = t
        # Fixed per bundle: the BF16-rounded skip weights (training's autocast operands) and ell.
        self.wide_bf = t['wide'].to(torch.bfloat16).float()
        self.wide64_bf = t['wide64'].to(torch.bfloat16).float()
        self.ell = F.log_softmax(t['mix_alpha'], -1).unsqueeze(1)  # (L, 1, 4)
        self.age_table = age_table(self.far).to(device)
        self.ema = torch.zeros(L, E, dtype=torch.float32, device=device)
        self.age = torch.full((L, E), self.far, dtype=torch.int64, device=device)  # an index of age_table
        self.routed = torch.zeros(L, E, dtype=torch.float32, device=device)

    @property
    def nbytes(self) -> int:
        bufs = list(self.t.values()) + [self.wide_bf, self.wide64_bf, self.ell, self.age_table, self.ema,
                                        self.age, self.routed, self.hidden_layers]
        return sum(b.numel() * b.element_size() for b in bufs)

    def reset(self) -> None:
        """Request start: no token decoded yet."""
        self.ema.zero_()
        self.age.fill_(self.far)
        self.routed.zero_()

    def advance(self, routes: torch.Tensor, step: torch.Tensor) -> None:
        """Fold token t's routes (L, K) into the state when step > 0; at step 0 the routes are the reset
        zeros (no token decoded), so neither the state nor ``routed`` changes. Graph-safe: no host sync."""
        go = step.view(1, 1) > 0
        routed = torch.zeros_like(self.routed).scatter_(1, routes, 1.0)
        ema = self.ema * self.beta + routed * (1 - self.beta)
        age = torch.where(routed > 0, torch.zeros_like(self.age), (self.age + 1).clamp(max=self.far))
        self.ema.copy_(torch.where(go, ema, self.ema))
        self.age.copy_(torch.where(go, age, self.age))
        self.routed.copy_(torch.where(go, routed, self.routed))

    def own(self, resident: torch.Tensor) -> torch.Tensor:
        """(L, E, 4) FP32: routed, ema, agef, resident of the end of token t."""
        return torch.stack([self.routed, self.ema, self.age_table[self.age], resident], -1)

    def scores(self, own: torch.Tensor, h: torch.Tensor, tid: torch.Tensor, upcoming: torch.Tensor,
               pos: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """Keep scores z (L, E) FP32. h (layers, D) the captured MoE inputs, tid/upcoming (1,) int64,
        pos (1, 1) FP32, table the bank's BF16 token table."""
        return self.mix(self.heads(own, h, tid, upcoming, pos, table))

    def heads(self, own: torch.Tensor, h: torch.Tensor, tid: torch.Tensor, upcoming: torch.Tensor,
              pos: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """Cumulative next-use logits u (L, E, 4) FP32 for 8, 16, 32 and 64 tokens, non-decreasing."""
        t, lin, g = self.t, self.linear, self.group
        L, E, Fe = own.shape
        bf = torch.bfloat16
        x = own.permute(0, 2, 1).reshape(L, Fe * E).to(bf).contiguous()
        z = lin(x, t['w_layer_q'], t['w_layer_s'], None, g).reshape(1, -1)
        hn = F.layer_norm(h.index_select(0, self.hidden_layers).float(), (h.shape[-1],)).to(bf).contiguous()
        hp = lin(hn, t['h_proj_q'], t['h_proj_s'], t['h_proj_b'], g).reshape(1, -1)
        parts = torch.cat([z, F.embedding(tid.view(1), table), F.embedding(upcoming.view(1), table),
                           pos.view(1, 1).to(bf), hp], 1).contiguous()
        a = F.gelu(lin(parts, t['fc1_q'], t['fc1_s'], t['fc1_b'], g))
        c = F.gelu(lin(a, t['fc2_q'], t['fc2_s'], t['fc2_b'], g))
        he = F.gelu(lin(c.expand(L, -1).contiguous(), t['we_q'], t['we_s'], t['we_b'], g))
        own_bf = own.to(bf).float()
        raw = lin(he, t['wb_q'], t['wb_s'], t['wb_b'], g).float().view(L, E, -1) + skip(own_bf, self.wide_bf)
        ev = torch.cat([raw[..., :1], F.softplus(raw[..., 1:])], -1).cumsum(-1)
        r = lin(he, t['wb64_q'], t['wb64_s'], t['wb64_b'], g).float() + skip(own_bf, self.wide64_bf)
        return torch.cat([ev, (ev[..., -1] + F.softplus(r)).unsqueeze(-1)], -1)

    def mix(self, u: torch.Tensor) -> torch.Tensor:
        """The logit of sum_h softmax(alpha)_h sigmoid(u_h), (L, E) FP32, in the log domain."""
        return torch.logsumexp(self.ell + F.logsigmoid(u), -1) - torch.logsumexp(self.ell + F.logsigmoid(-u), -1)
