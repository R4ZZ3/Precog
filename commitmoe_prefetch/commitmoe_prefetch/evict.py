"""Learned eviction rows (EXP526): keep scores for freetoken.moe.learned_evict from the bank's trunk.

A bundle (experiments/exp526_export_evict.py) holds, for each owner layer m, the 256 rows of a
trained output that read the trunk activation g = gelu(up(proj(x))) of one prefetch head, the
row's source. Row e gives the logit of "expert e of layer m is used again within the next 8 tokens".
The rows are BF16 in the bundle and quantized here with the bank's own rule, so the arithmetic is
the one the EXP523-EXP525 replay scored ("full INT8": int8 codes, FP32 sums, BF16 output, then a
FP32 sigmoid). The bank's calibration (a, b) belongs to the prefetch output and is not applied.

Bundle versions:
  1  the source of each row is its owner layer's own head (EXP526, EXP531). Owner layers without
     a prefetch head (L0-L5) cannot have rows.
  2  ``source_layers`` names the head of each row separately (EXP540). The rows of L0-L5 read the
     L6 head's trunk activation (EXP537, EXP539), so several rows can share one source.
Both versions may carry ``sources``: the exporter's provenance (checkpoint path and hash per
layer), which the loader does not read.
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import bank_loader
from .qbank import qlinear

BUNDLE_VERSIONS = (1, 2)
HORIZON_TOKENS = 8  # the replay's keep score: next use within 8 tokens (column 0 of the B output)


def _ints(values) -> list[int] | None:
    """``values`` as a list of Python ints, or None if any entry is not an integer."""
    out = []
    for v in values:
        if isinstance(v, torch.Tensor):
            if v.numel() != 1 or v.dtype.is_floating_point:
                return None
            v = v.item()
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        out.append(v)
    return out


class EvictRows:
    """Keep-score rows of M owner layers on ``device``; ``__call__`` writes (M, E) FP32 in place."""

    def __init__(self, path: Path, target_layers: list[int], up_dim: int, num_experts: int, group: int,
                 device: torch.device, release_sha256: str | None, num_layers: int):
        """``release_sha256``: the served bank's model.safetensors hash (from its manifest). The rows
        read that bank's trunk, so a bundle exported against another release is refused.
        ``num_layers``: the model's MoE layer count; every owner layer must be one of them."""
        bundle = torch.load(path, map_location='cpu', weights_only=True)
        version = bundle.get('version')
        if version not in BUNDLE_VERSIONS:
            raise ValueError(f'learned eviction bundle {path}: version {version}, expected one of {BUNDLE_VERSIONS}')
        problems = []
        layers = _ints(bundle['layers'])
        # Version 1 has no source_layers entry: each row reads its own layer's head.
        source_layers = _ints(bundle['source_layers']) if version == 2 and 'source_layers' in bundle else None
        if version == 1:
            if 'source_layers' in bundle:
                problems.append('version 1 bundle has source_layers; they need version 2')
            source_layers = layers
        elif source_layers is None:
            problems.append('version 2 bundle needs integer source_layers, one per row')
        w, b = bundle['weight'], bundle['bias']
        if bundle.get('horizon_tokens') != HORIZON_TOKENS:
            problems.append(f'horizon {bundle.get("horizon_tokens")} tokens, expected {HORIZON_TOKENS}')
        if release_sha256 is not None and bundle.get('release_sha256') != release_sha256:
            problems.append(f'exported against release {str(bundle.get("release_sha256"))[:12]}, '
                            f'served release is {release_sha256[:12]}')
        if layers is None:
            problems.append('layers must be integers')
        else:
            if layers != sorted(set(layers)) or not layers:
                problems.append(f'owner layers {layers} must be non-empty, ascending and unique')
            if bad := [L for L in layers if not 0 <= L < num_layers]:
                problems.append(f'owner layers {bad} are not layers 0..{num_layers - 1}')
            if source_layers is not None:
                if len(source_layers) != len(layers):
                    problems.append(f'{len(source_layers)} source layers for {len(layers)} owner layers')
                # A missing head is a source problem, not an owner problem: in version 2, owner L0 is
                # valid when its source is a predicted layer.
                if missing := sorted({S for S in source_layers if S not in target_layers}):
                    problems.append(f'source layers {missing} have no prefetch head '
                                    f'(targets {target_layers[0]}..{target_layers[-1]})')
        M = len(layers) if layers is not None else -1
        if w.dtype != torch.bfloat16 or tuple(w.shape) != (M, num_experts, up_dim):
            problems.append(f'weight {w.dtype} {tuple(w.shape)}, expected bfloat16 {(M, num_experts, up_dim)}')
        elif not bool(torch.isfinite(w.float()).all()):
            problems.append('non-finite weight')
        if b.dtype != torch.bfloat16 or tuple(b.shape) != (M, num_experts):
            problems.append(f'bias {b.dtype} {tuple(b.shape)}, expected bfloat16 {(M, num_experts)}')
        elif not bool(torch.isfinite(b.float()).all()):
            problems.append('non-finite bias')
        if problems:
            raise ValueError(f'learned eviction bundle {path}: ' + '; '.join(problems))
        self.version = version
        self.layers = layers
        self.source_layers = source_layers
        # Row of the bank's trunk output g for each owner layer. Duplicates are allowed: g is read,
        # never written, so two rows on one head see the same activation.
        self.heads = torch.tensor([target_layers.index(S) for S in source_layers], dtype=torch.int64, device=device)
        codes, scale = bank_loader.quantize(w, group)
        self.codes, self.scale = codes.to(device), scale.to(device)
        self.bias = b.to(device)
        self.group = group

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.codes, self.scale, self.bias, self.heads))

    def __call__(self, g: torch.Tensor, keep: torch.Tensor) -> None:
        """keep[m] = sigmoid(rows_m @ g[source head of m] + bias_m) for every owner layer m."""
        x = g.index_select(0, self.heads)  # (M, up_dim) BF16, contiguous
        keep.copy_(torch.sigmoid(qlinear(x, self.codes, self.scale, self.bias, self.group).float()))
