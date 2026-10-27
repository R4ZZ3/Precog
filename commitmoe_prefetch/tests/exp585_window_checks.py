"""Checks of the EXP585 GPU window (test_gpu_shared_evict.py) that hold on any device; their CPU tests are
in test_exp585_window_checks.py, so a check that would let a NaN or a missing stamp pass fails before the
window, not in it."""
from __future__ import annotations

import contextlib

import torch

# Smallest positive BF16 subnormal: the spacing of BF16 values at and near zero.
BF16_MIN_SPACING = 2.0 ** -133


@contextlib.contextmanager
def fp32_exact():
    """FP32 matmuls without TF32 or reduced-precision paths, for the reference and the tolerance; the
    previous settings come back afterwards."""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[0], old[1]
        torch.set_float32_matmul_precision(old[2])


def bf16_spacing(ref: torch.Tensor) -> torch.Tensor:
    """The BF16 value spacing at each element of ``ref`` (one rounding step): 2^(e - 7) for |ref| in
    [2^e, 2^(e + 1)), never below the subnormal spacing 2^-133 (zero and subnormals included)."""
    a = ref.float().abs()
    return torch.clamp(torch.exp2(torch.floor(torch.log2(a)) - 7), min=BF16_MIN_SPACING)


def qgemv_ratio(y: torch.Tensor, ref: torch.Tensor, x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor,
                group: int) -> torch.Tensor:
    """|y - ref| over the declared tolerance, per output. y is qgemv's BF16 output and ref reference_linear's
    on the same input. Declared tolerance (not a proven bound for every reduction order): 2 K 2^-24 sum|x w|
    (first-order FP32 summation error of either order of the K products) plus one BF16 spacing of ref.
    Raises ValueError when any input, output, tolerance or ratio is not finite, or a tolerance is not > 0:
    a NaN must fail the check, not drop out of a max()."""
    with fp32_exact():
        w = codes.float().abs() * scale.float().repeat_interleave(group, -1)[..., :codes.shape[-1]]
        mag = torch.einsum('hk,hnk->hn', x.float().abs(), w)
    tol = 2 * codes.shape[-1] * 2.0 ** -24 * mag + bf16_spacing(ref)
    diff = (y.float() - ref.float()).abs()
    ratio = diff / tol
    for name, t in (('x', x), ('scale', scale), ('y', y), ('ref', ref), ('tolerance', tol), ('diff', diff),
                    ('ratio', ratio)):
        if not bool(torch.isfinite(t.float()).all()):
            raise ValueError(f'qgemv check: {name} is not finite')
    if not bool((tol > 0).all()):
        raise ValueError('qgemv check: a tolerance is not > 0')
    return ratio


def stamp_intervals(stamps: torch.Tensor, steps: list[int], expected_steps: list[int], num_layers: int) -> dict:
    """(prefix, wait) in ns per row from the hooks' timing rows: prefix = scores-ready (column L) minus main's
    stamp at observe(0) (column 0); wait = main after minus before the wait at J (columns L + 2 and L + 1).
    Raises ValueError unless the rows cover exactly ``expected_steps`` and every row is ordered: all stamps
    > 0, observe(0) <= scores-ready <= after the wait, and before the wait <= after the wait."""
    L = num_layers
    if list(steps) != list(expected_steps):
        raise ValueError(f'timing rows cover steps {list(steps)[:8]}..., expected {list(expected_steps)[:8]}...')
    if stamps.dtype != torch.int64 or tuple(stamps.shape) != (len(expected_steps), L + 3):
        raise ValueError(f'timing rows {stamps.dtype} {tuple(stamps.shape)}, expected int64 '
                         f'{(len(expected_steps), L + 3)}')
    cols = stamps[:, [0, L, L + 1, L + 2]]
    if not bool((cols > 0).all()):
        raise ValueError('timing rows: a stamp is missing (<= 0)')
    obs, ready, before, after = cols.unbind(1)
    for name, ok in (('observe(0) <= scores-ready', obs <= ready), ('scores-ready <= after the wait', ready <= after),
                     ('before <= after the wait', before <= after)):
        if not bool(ok.all()):
            raise ValueError(f'timing rows: {name} fails in {int((~ok).sum())} rows')
    return dict(prefix=ready - obs, wait=after - before)
