"""Fast R584 scorer (EXP586): the function of shared_evict.SharedEvict with fewer kernel launches.

Same bundle, state, features and arithmetic contract as SharedEvict; only the launch structure differs:
  F4   advance_and_own: one fast_kernels.state_own launch advances the state and writes the BF16 own
       features in the w_layer input layout (SharedEvict: advance, own, permute and cast).
  F6   the GELU after fc1, fc2 and we runs in the qgemv epilogue, and we reads the single trunk row with
       head stride 0 instead of an expanded copy.
  F8   fc1 and h_proj get more programs: 'blockn' (BLOCK_N 8, the qgemv k-loop order kept) or 'splitk'
       (K groups split, FP32 partials added in split order, bias and BF16 rounding once).
  F3b  wb and wb64 run as one qgemv launch on codes, scales and biases concatenated along N at load.
  F3   readout + publish: the head tail, the mix, the keep scores, valid and the non-finite count.
The GPU tests (tests/test_gpu_exp586_fast.py) compare each part with the SharedEvict kernels.

kernels=False runs every part with torch ops (the CPU path of the tests); its linear is
shared_evict.reference_linear and ignores the F8 launch settings, which change only the GPU sum order.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from . import fast_kernels as fk
from .qbank import qlinear
from .shared_evict import SharedEvict, reference_linear, skip

F8 = ('blockn', 'splitk')
# Launch settings fixed before any GPU measurement (EXP586 plan, Codex 25fe905f item 5).
BLOCK_N = 32                         # qbank.BLOCK_N, every stage without an F8 setting
F8_BLOCK_N = dict(fc1=8, h_proj=8)   # 'blockn': fc1 1024 / 8 = 128 programs, h_proj 2 x 256 / 8 = 64
F8_SPLITS = dict(fc1=4, h_proj=8)    # 'splitk': fc1 45 groups as 12/12/12/9, h_proj 16 groups as 2 each
REDUCE_BLOCK = 256
# torch keeps denormals; Triton's default links libdevice with flush-to-zero (see fast_kernels).
NO_FTZ = dict(enable_reflect_ftz=False)


def split_groups(groups: int, splits: int) -> int:
    """Groups per split for qgemv2: ceil(groups / splits). Refuses a split count that leaves a split
    without a group, because that split would write zero partials that the plan did not declare."""
    per = triton.cdiv(groups, splits)
    if splits < 2 or (splits - 1) * per >= groups:
        raise ValueError(f'{groups} K groups cannot be split {splits} ways with a group in every split')
    return per


class FastSharedEvict(SharedEvict):
    """R584 on ``device`` with the EXP586 kernels. ``score`` does the whole token; the parts are public
    for the tests. The SharedEvict methods advance/own/heads/mix/scores are refused: they read the
    separate wb/wb64 tensors, which this class concatenates and drops."""

    def __init__(self, meta: dict, tensors: dict[str, torch.Tensor], device: torch.device, f8: str,
                 kernels: bool = True):
        if f8 not in F8:
            raise ValueError(f'F8 variant {f8!r}, expected one of {F8}')
        super().__init__(meta, tensors, device, qlinear if kernels else reference_linear)
        L, E = self.f['n_layer'], self.f['n_exp']
        if kernels and E & (E - 1):
            raise ValueError(f'{E} experts per layer: the fused kernels need a power of two')
        self.kernels, self.f8 = kernels, f8
        t = self.t
        for part in ('_q', '_s', '_b'):
            t['wbx' + part] = torch.cat([t.pop('wb' + part), t.pop('wb64' + part)], 1).contiguous()
        self.ell2 = self.ell.reshape(L, -1).contiguous()          # (L, 4) for the readout kernel
        self.x = torch.zeros(L, 4 * E, dtype=torch.bfloat16, device=device)  # own features, w_layer input
        self.bad = torch.zeros(1, dtype=torch.int32, device=device)            # non-finite z of this token
        self.partial: dict[str, torch.Tensor] = {}
        if f8 == 'splitk':
            for name, splits in F8_SPLITS.items():
                H, N, _ = t[name + '_q'].shape
                split_groups(t[name + '_s'].shape[2], splits)
                self.partial[name] = torch.zeros(splits, H, N, dtype=torch.float32, device=device)

    @property
    def nbytes(self) -> int:
        extra = [self.ell2, self.x, self.bad] + list(self.partial.values())
        return super().nbytes + sum(b.numel() * b.element_size() for b in extra)

    def reset(self) -> None:
        super().reset()
        self.bad.zero_()

    def advance(self, *args, **kwargs):
        raise NotImplementedError('FastSharedEvict: use advance_and_own')

    own = heads = mix = scores = advance

    # -- parts ------------------------------------------------------------------------------------

    def advance_and_own(self, routes: torch.Tensor, step: torch.Tensor, resident: torch.Tensor) -> torch.Tensor:
        """Advance the state with token t's routes (step > 0 only) and return the own features x (L, 4E)
        BF16, x[l, f * E + e] for f = routed, ema, agef, resident after the update."""
        L, E = self.ema.shape
        if not self.kernels:
            SharedEvict.advance(self, routes, step)
            own = SharedEvict.own(self, resident)
            self.x.copy_(own.permute(0, 2, 1).reshape(L, 4 * E).to(torch.bfloat16))
            return self.x
        K = routes.shape[1]
        fk.state_own[(L,)](routes, step, self.ema, self.age, self.routed, self.age_table, resident, self.x,
                           self.beta, 1 - self.beta, self.far, E=E, K=K, K_POW2=triton.next_power_of_2(K),
                           num_warps=4, **NO_FTZ)
        return self.x

    def matvec(self, x: torch.Tensor, name: str, gelu: bool = False, shared_row: bool = False) -> torch.Tensor:
        """(H, N) BF16 = [gelu](qgemv of ``name``). shared_row: x is one (1, K) row read by every head."""
        t = self.t
        codes, scale, bias = t[name + '_q'], t[name + '_s'], t.get(name + '_b')
        H, N, K = codes.shape
        if not self.kernels:
            y = reference_linear(x.expand(H, -1).contiguous() if shared_row else x, codes, scale, bias, self.group)
            return F.gelu(y) if gelu else y
        assert x.shape == ((1, K) if shared_row else (H, K)) and x.is_contiguous(), (name, tuple(x.shape))
        y = torch.empty(H, N, dtype=torch.bfloat16, device=x.device)
        KG, stride, b = scale.shape[2], 0 if shared_row else K, bias if bias is not None else y
        if name in self.partial:
            P, splits = self.partial[name], F8_SPLITS[name]
            fk.qgemv2[(H, triton.cdiv(N, BLOCK_N), splits)](
                x, codes, scale, y, y, P, N, K, KG, stride, split_groups(KG, splits), HAS_BIAS=False, GELU=False,
                SPLIT=splits, GROUP=self.group, BLOCK_N=BLOCK_N, num_warps=4, **NO_FTZ)
            fk.splitk_reduce[(triton.cdiv(H * N, REDUCE_BLOCK),)](
                P, b, y, H * N, HAS_BIAS=bias is not None, GELU=gelu, SPLIT=splits, BLOCK=REDUCE_BLOCK,
                num_warps=4, **NO_FTZ)
            return y
        block_n = F8_BLOCK_N.get(name, BLOCK_N) if self.f8 == 'blockn' else BLOCK_N
        fk.qgemv2[(H, triton.cdiv(N, block_n), 1)](
            x, codes, scale, b, y, y, N, K, KG, stride, KG, HAS_BIAS=bias is not None, GELU=gelu, SPLIT=1,
            GROUP=self.group, BLOCK_N=block_n, num_warps=4, **NO_FTZ)
        return y

    def trunk(self, x: torch.Tensor, h: torch.Tensor, tid: torch.Tensor, upcoming: torch.Tensor,
              pos: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """The joint wb/wb64 output (L, 4E) BF16 from the own features and the trunk inputs."""
        bf = torch.bfloat16
        z = self.matvec(x, 'w_layer').reshape(1, -1)
        hn = F.layer_norm(h.index_select(0, self.hidden_layers).float(), (h.shape[-1],)).to(bf).contiguous()
        hp = self.matvec(hn, 'h_proj').reshape(1, -1)
        parts = torch.cat([z, F.embedding(tid.view(1), table), F.embedding(upcoming.view(1), table),
                           pos.view(1, 1).to(bf), hp], 1).contiguous()
        a = self.matvec(parts, 'fc1', gelu=True)
        c = self.matvec(a, 'fc2', gelu=True)
        he = self.matvec(c, 'we', gelu=True, shared_row=True)
        return self.matvec(he, 'wbx')

    def readout(self, wbo: torch.Tensor, x: torch.Tensor, keep: torch.Tensor, u: torch.Tensor | None = None) -> None:
        """keep (L, E) FP32 = z; u (L, E, 4) FP32 = the cumulative logits when given; self.bad += non-finite z."""
        L, E = keep.shape
        if not self.kernels:
            own_bf = x.view(L, 4, E).permute(0, 2, 1).float()
            raw = wbo[:, :3 * E].float().view(L, E, 3) + skip(own_bf, self.wide_bf)
            ev = torch.cat([raw[..., :1], F.softplus(raw[..., 1:])], -1).cumsum(-1)
            r = wbo[:, 3 * E:].float() + skip(own_bf, self.wide64_bf)
            uu = torch.cat([ev, (ev[..., -1] + F.softplus(r)).unsqueeze(-1)], -1)
            z = torch.logsumexp(self.ell + F.logsigmoid(uu), -1) - torch.logsumexp(self.ell + F.logsigmoid(-uu), -1)
            keep.copy_(z)
            if u is not None:
                u.copy_(uu)
            self.bad.add_((~torch.isfinite(z)).sum().to(torch.int32))
            return
        assert wbo.shape == (L, 4 * E) and keep.is_contiguous() and (u is None or u.is_contiguous())
        fk.readout[(L,)](wbo, x, self.wide_bf, self.wide64_bf, self.ell2, keep, keep if u is None else u, self.bad,
                         E=E, WRITE_U=u is not None, num_warps=4, **NO_FTZ)

    def publish(self, step: torch.Tensor, valid: torch.Tensor, nonfinite: torch.Tensor) -> None:
        """valid = (step > 0) & all z finite; nonfinite += (step > 0) & some z not finite; clear the count."""
        if not self.kernels:
            finite, scored = self.bad[0] == 0, step[0] > 0
            valid.copy_((scored & finite).to(torch.int32).view(1))
            nonfinite.add_((scored & ~finite).to(torch.int64))
            self.bad.zero_()
            return
        fk.publish[(1,)](self.bad, step, valid, nonfinite, **NO_FTZ)

    def score(self, routes: torch.Tensor, step: torch.Tensor, resident: torch.Tensor, h: torch.Tensor,
              tid: torch.Tensor, upcoming: torch.Tensor, pos: torch.Tensor, table: torch.Tensor,
              keep: torch.Tensor, valid: torch.Tensor, nonfinite: torch.Tensor) -> None:
        """One scored token: state, trunk, readout into ``keep``, then valid and the non-finite count."""
        x = self.advance_and_own(routes, step, resident)
        self.readout(self.trunk(x, h, tid, upcoming, pos, table), x, keep)
        self.publish(step, valid, nonfinite)
