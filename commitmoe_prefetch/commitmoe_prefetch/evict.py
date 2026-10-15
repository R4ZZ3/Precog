"""Learned eviction rows (EXP526): keep scores for freetoken.moe.learned_evict from the bank's trunk.

A bundle (experiments/exp526_export_evict.py) holds, for each target layer m, the 256 rows of a
trained output that read the same trunk activation g = gelu(up(proj(x))) as the prefetch output
of head m. Row e gives the logit of "expert e of layer m is used again within the next 8 tokens".
The rows are BF16 in the bundle and quantized here with the bank's own rule, so the arithmetic is
the one the EXP523-EXP525 replay scored ("full INT8": int8 codes, FP32 sums, BF16 output, then a
FP32 sigmoid). The bank's calibration (a, b) belongs to the prefetch output and is not applied.
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import bank_loader
from .qbank import qlinear

BUNDLE_VERSION = 1
HORIZON_TOKENS = 8  # the replay's keep score: next use within 8 tokens (column 0 of the B output)


class EvictRows:
    """Keep-score rows of M target layers on ``device``; ``__call__`` writes (M, E) FP32 in place."""

    def __init__(self, path: Path, target_layers: list[int], up_dim: int, num_experts: int, group: int,
                 device: torch.device, release_sha256: str | None):
        """``release_sha256``: the served bank's model.safetensors hash (from its manifest). The rows
        read that bank's trunk, so a bundle exported against another release is refused."""
        bundle = torch.load(path, map_location='cpu', weights_only=True)
        if bundle.get('version') != BUNDLE_VERSION:
            raise ValueError(f'learned eviction bundle {path}: version {bundle.get("version")}, '
                             f'expected {BUNDLE_VERSION}')
        layers = [int(L) for L in bundle['layers']]
        w, b = bundle['weight'], bundle['bias']
        problems = []
        if bundle.get('horizon_tokens') != HORIZON_TOKENS:
            problems.append(f'horizon {bundle.get("horizon_tokens")} tokens, expected {HORIZON_TOKENS}')
        if release_sha256 is not None and bundle.get('release_sha256') != release_sha256:
            problems.append(f'exported against release {str(bundle.get("release_sha256"))[:12]}, '
                            f'served release is {release_sha256[:12]}')
        if layers != sorted(set(layers)) or not layers:
            problems.append(f'layers {layers} must be non-empty, ascending and unique')
        if missing := [L for L in layers if L not in target_layers]:
            problems.append(f'layers {missing} have no prefetch head (targets {target_layers[0]}..{target_layers[-1]})')
        if w.dtype != torch.bfloat16 or tuple(w.shape) != (len(layers), num_experts, up_dim):
            problems.append(f'weight {w.dtype} {tuple(w.shape)}, expected bfloat16 '
                            f'{(len(layers), num_experts, up_dim)}')
        if b.dtype != torch.bfloat16 or tuple(b.shape) != (len(layers), num_experts):
            problems.append(f'bias {b.dtype} {tuple(b.shape)}, expected bfloat16 {(len(layers), num_experts)}')
        elif not bool(torch.isfinite(b.float()).all()):
            problems.append('non-finite bias')
        if problems:
            raise ValueError(f'learned eviction bundle {path}: ' + '; '.join(problems))
        self.layers = layers
        # Row of the bank's trunk output g for each eviction layer.
        self.heads = torch.tensor([target_layers.index(L) for L in layers], dtype=torch.int64, device=device)
        codes, scale = bank_loader.quantize(w, group)
        self.codes, self.scale = codes.to(device), scale.to(device)
        self.bias = b.to(device)
        self.group = group

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.codes, self.scale, self.bias, self.heads))

    def __call__(self, g: torch.Tensor, keep: torch.Tensor) -> None:
        """keep[m] = sigmoid(rows_m @ g[head of m] + bias_m) for every eviction layer m."""
        x = g.index_select(0, self.heads)  # (M, up_dim) BF16, contiguous
        keep.copy_(torch.sigmoid(qlinear(x, self.codes, self.scale, self.bias, self.group).float()))
