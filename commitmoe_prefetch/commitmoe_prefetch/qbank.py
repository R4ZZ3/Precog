"""The int8 head bank on the GPU: feature assembly, scoring and calibration in one call.

Same arithmetic as the served bank (experiments/exp493_qbank.QBank): every head matrix is
int8 with one FP16 scale per output row and per group of inputs, and `qgemv` computes
y[h] = Wq[h] @ x[h] for all heads in one launch, accumulating in FP32 and rounding to BF16.
It adds the bias in FP32 before that rounding; the BF16 reference (reference_scorer) adds
it after. The features come from reference_scorer.assemble, which the release gate checked
bit for bit against the served assembly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from . import bank_loader
from .reference_scorer import assemble
from .spec import PredictorConfig

BLOCK_N = 32


@triton.jit
def qgemv(X, W, S, B, Y, N, K, KG, HAS_BIAS: tl.constexpr, GROUP: tl.constexpr, BLOCK_N: tl.constexpr):
    """Y[h, n] = sum_k W[h, n, k] * S[h, n, k // GROUP] * X[h, k] (+ B[h, n]), BF16 out.

    X (H, K) any float dtype; W (H, N, K) int8; S (H, N, KG) fp16 with KG = cdiv(K, GROUP);
    B (H, N) or unused. One program per (head, block of BLOCK_N outputs). The tail group
    of K is masked to zero.
    """
    h = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for g in range(0, KG):
        k = g * GROUP + tl.arange(0, GROUP)
        k_ok = k < K
        x = tl.load(X + h * K + k, mask=k_ok, other=0.0).to(tl.float32)
        w = tl.load(W + (h * N + n[:, None]) * K + k[None, :], mask=n_ok[:, None] & k_ok[None, :],
                    other=0).to(tl.float32)
        s = tl.load(S + (h * N + n) * KG + g, mask=n_ok, other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1) * s
    if HAS_BIAS:
        acc += tl.load(B + h * N + n, mask=n_ok, other=0.0).to(tl.float32)
    tl.store(Y + h * N + n, acc.to(tl.bfloat16), mask=n_ok)


def qlinear(x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None,
            group: int) -> torch.Tensor:
    """(H, K) x int8 (H, N, K) -> (H, N) BF16 with qgemv, one launch for all H heads."""
    H, N, K = codes.shape
    assert x.shape == (H, K) and x.is_contiguous(), (tuple(x.shape), H, K)
    y = torch.empty(H, N, dtype=torch.bfloat16, device=x.device)
    qgemv[(H, triton.cdiv(N, BLOCK_N))](x, codes, scale, bias if bias is not None else y, y, N, K,
                                        scale.shape[2], HAS_BIAS=bias is not None, GROUP=group,
                                        BLOCK_N=BLOCK_N, num_warps=4)
    return y


class Int8Bank:
    """All heads of one predictor as int8 buffers on ``device``; ``__call__`` returns (heads, E) fp32."""

    def __init__(self, cfg: PredictorConfig, state: dict[str, torch.Tensor], device: torch.device):
        self.cfg = cfg
        self.group = cfg.serving_reference.int8_group
        self.buffers = {name: t.to(device) for name, t in bank_loader.to_int8(cfg, state).items()}

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buffers.values())

    def _qlinear(self, x: torch.Tensor, name: str, bias: torch.Tensor | None) -> torch.Tensor:
        return qlinear(x, self.buffers[name + '_q'], self.buffers[name + '_s'], bias, self.group)

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return self._qlinear(x.contiguous(), name, self.buffers.get(name + '_b'))

    def __call__(self, raw: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.trunk(raw))

    def trunk(self, raw: dict[str, torch.Tensor]) -> torch.Tensor:
        """gelu(up(proj(x))) of every head, (heads, up_dim) BF16: the input of the out matrix.

        A learned eviction output (evict.EvictRows) reads the same activation, so the trunk
        runs once per token for both outputs and the EMA prior is advanced once."""
        b = self.buffers
        feats, h, hd = assemble(raw, self.cfg.base_model.num_experts, self.cfg.features.ema_beta)
        feats, h, hd = feats.to(torch.bfloat16), h.to(torch.bfloat16), hd.to(torch.bfloat16)
        h_hat = h + self._linear(F.gelu(self._linear(torch.cat([h, hd], 1), 'fh_down')), 'fh_up')
        trv = self._qlinear(h_hat.contiguous(), 'W', b['router_b'])
        emb = F.embedding(raw['tid'], b['table']).expand(len(h), -1)
        nxt = F.embedding(raw['upcoming_tid'], b['table']).expand(len(h), -1)
        x = torch.cat([feats, trv, emb, nxt], 1)
        return F.gelu(self._linear(self._linear(x, 'proj'), 'up'))

    def head(self, g: torch.Tensor) -> torch.Tensor:
        """The prefetch probabilities (heads, E) FP32 from the trunk activation ``g``."""
        b = self.buffers
        scores = self._linear(g, 'out').float()
        return torch.sigmoid(scores * b['a'] + b['b'])
