"""Load the published predictor bank without the experiment code.

`load` reads config.json and model.safetensors, validates both against each other, and
returns the BF16 buffers. `to_int8` builds the int8 buffers that the served bank used
(EXP493 QBank): symmetric int8 codes with one FP16 scale per output row and per group of
128 consecutive inputs. The arithmetic of `quantize` must stay identical to
experiments/exp493_qbank.quantize, because check_equivalence.py compares the results bit
for bit; change both or neither.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .spec import PredictorConfig

# Stacked nn.Linear weights per head; with the router W these are the int8-stored matrices.
LINEARS = ('fh_down', 'fh_up', 'proj', 'up', 'out')
BIASED = ('fh_down', 'fh_up', 'up', 'out')  # proj has no bias
EXPECTED = {f'{n}_w' for n in LINEARS} | {f'{n}_b' for n in BIASED} | {'W', 'router_b', 'table', 'a', 'b'}


def expected_shapes(cfg: PredictorConfig) -> dict[str, tuple[int, ...]]:
    n = len(cfg.target_layers)
    e, d = cfg.base_model.num_experts, cfg.base_model.hidden_size
    arch, feats = cfg.architecture, cfg.features
    proj_in = arch.feature_dim + e + 2 * feats.token_embedding_dim
    return {
        'fh_down_w': (n, arch.forecast_hidden, 2 * d), 'fh_down_b': (n, arch.forecast_hidden),
        'fh_up_w': (n, d, arch.forecast_hidden), 'fh_up_b': (n, d),
        'proj_w': (n, arch.proj_dim, proj_in),
        'up_w': (n, arch.up_dim, arch.proj_dim), 'up_b': (n, arch.up_dim),
        'out_w': (n, e, arch.up_dim), 'out_b': (n, e),
        'W': (n, e, d), 'router_b': (n, e),
        'table': (feats.token_table_rows, feats.token_embedding_dim),
        'a': (n, 1), 'b': (n, 1),
    }


def load(folder: Path) -> tuple[PredictorConfig, dict[str, torch.Tensor]]:
    folder = Path(folder)
    cfg = PredictorConfig.model_validate(json.loads((folder / 'config.json').read_text()))
    state = load_file(str(folder / 'model.safetensors'))
    if set(state) != EXPECTED:
        raise ValueError(f'tensor names differ: missing {EXPECTED - set(state)}, extra {set(state) - EXPECTED}')
    for name, shape in expected_shapes(cfg).items():
        if tuple(state[name].shape) != shape:
            raise ValueError(f'{name}: shape {tuple(state[name].shape)}, config implies {shape}')
        # Calibration stays FP32 (the served bank restores it from the FP32 reference).
        want = torch.float32 if name in ('a', 'b') else torch.bfloat16
        if state[name].dtype != want:
            raise ValueError(f'{name}: dtype {state[name].dtype}, expected {want}')
        if not bool(torch.isfinite(state[name].float()).all()):
            raise ValueError(f'{name}: non-finite values')
    return cfg, state


def quantize(w: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(H, N, K) float -> int8 codes (H, N, K) and FP16 scales (H, N, ceil(K / group))."""
    w = w.float()
    assert bool(torch.isfinite(w).all()), 'non-finite weight'
    H, N, K = w.shape
    pad = (-K) % group
    groups = F.pad(w, (0, pad)).view(H, N, -1, group)
    # An all-zero group gets scale 1 so the division stays finite; its codes are all 0.
    scale = groups.abs().amax(3, keepdim=True) / 127
    scale = torch.where(scale > 0, scale, torch.ones_like(scale)).half()
    # A nonzero group with max|w| below about 3.8e-6 rounds to an FP16 scale of 0 and the
    # division would erase it. Fail instead of quantizing it silently.
    assert bool((scale > 0).all()) and bool(torch.isfinite(scale).all()), 'FP16 scale underflow'
    codes = (groups / scale.float()).round().clamp(-127, 127).to(torch.int8)
    return codes.view(H, N, -1)[..., :K].contiguous(), scale.squeeze(3).contiguous()


def to_int8(cfg: PredictorConfig, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The served int8 bank's buffers: <name>_q / <name>_s replace each matrix; the rest is unchanged."""
    group = cfg.serving_reference.int8_group
    out = {k: v for k, v in state.items() if not (k.endswith('_w') and k[:-2] in LINEARS) and k != 'W'}
    for name in LINEARS + ('W',):
        out[name + '_q'], out[name + '_s'] = quantize(state[name + '_w' if name != 'W' else 'W'], group)
    return out
