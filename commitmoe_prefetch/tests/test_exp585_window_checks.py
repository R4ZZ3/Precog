"""CPU tests of exp585_window_checks: NaN and inf never pass, the tolerance is positive at zero and in the
BF16 subnormal range, TF32 settings come back, and timing rows that are missing, short or out of order are
refused."""
from __future__ import annotations

import pytest
import torch

import exp585_window_checks as wc


def _linear(seed=0, H=2, N=3, K=8, group=4):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(-127, 128, (H, N, K), generator=g, dtype=torch.int8)
    scale = (torch.rand(H, N, K // group, generator=g) + 0.5).half()
    x = torch.randn(H, K, generator=g).bfloat16()
    w = codes.float() * scale.float().repeat_interleave(group, -1)
    ref = torch.einsum('hk,hnk->hn', x.float(), w).bfloat16()
    return x, codes, scale, ref, group


def test_equal_outputs_give_ratio_zero():
    x, codes, scale, ref, group = _linear()
    assert torch.equal(wc.qgemv_ratio(ref.clone(), ref, x, codes, scale, group), torch.zeros(ref.shape))


def test_one_bf16_step_is_within_the_tolerance():
    x, codes, scale, ref, group = _linear()
    y = (ref.float() + wc.bf16_spacing(ref)).bfloat16()
    assert float(wc.qgemv_ratio(y, ref, x, codes, scale, group).max()) <= 1.0


@pytest.mark.parametrize('where', ['y', 'ref', 'x'])
@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_values_raise(where, value):
    x, codes, scale, ref, group = _linear()
    y = ref.clone()
    {'y': y, 'ref': ref, 'x': x}[where][0, 0] = value
    with pytest.raises(ValueError, match='not finite'):
        wc.qgemv_ratio(y, ref, x, codes, scale, group)


def test_max_of_python_floats_would_hide_nan():
    """The failure mode the ValueError guards against."""
    assert max(0.0, float('nan')) == 0.0


def test_zero_and_subnormal_spacing():
    vals = torch.tensor([0.0, 2.0 ** -133, 2.0 ** -130, -2.0 ** -127, 1.0, 1.5, -3.0])
    s = wc.bf16_spacing(vals.bfloat16())
    assert s[0] == s[1] == s[2] == s[3] == wc.BF16_MIN_SPACING
    assert s[4] == 2.0 ** -7 and s[5] == 2.0 ** -7 and s[6] == 2.0 ** -6
    assert bool((s > 0).all())
    # A zero reference with a zero input row still has a positive tolerance.
    x, codes, scale, _, group = _linear()
    x = torch.zeros_like(x)
    ref = torch.zeros(2, 3).bfloat16()
    assert torch.equal(wc.qgemv_ratio(ref.clone(), ref, x, codes, scale, group), torch.zeros(2, 3))


def test_fp32_exact_restores_settings():
    before = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
    torch.set_float32_matmul_precision('high')
    try:
        with wc.fp32_exact():
            assert torch.get_float32_matmul_precision() == 'highest'
            assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.get_float32_matmul_precision() == 'high'
    finally:
        torch.set_float32_matmul_precision(before[1])
        torch.backends.cuda.matmul.allow_tf32 = before[0]


L = 4


def _rows(n=3):
    s = torch.zeros(n, L + 3, dtype=torch.int64)
    base = torch.arange(n, dtype=torch.int64)[:, None] * 1_000_000 + 1000
    s[:, :L] = base + torch.arange(L) * 100
    s[:, L] = base[:, 0] + 50_000      # scores ready
    s[:, L + 1] = base[:, 0] + 120     # main before the wait at J
    s[:, L + 2] = base[:, 0] + 50_100  # main after the wait
    return s


def test_stamp_intervals_of_ordered_rows():
    out = wc.stamp_intervals(_rows(), [0, 1, 2], [0, 1, 2], L)
    assert out['prefix'].tolist() == [50_000] * 3 and out['wait'].tolist() == [49_980] * 3


@pytest.mark.parametrize('change,message', [
    (lambda s: s.__setitem__((1, L), 0), 'missing'),
    (lambda s: s.__setitem__((1, L), s[1, 0] - 1), r'observe\(0\) <= scores-ready'),
    (lambda s: s.__setitem__((2, L + 1), s[2, L + 2] + 1), 'before <= after'),
    (lambda s: s.__setitem__((0, L), s[0, L + 2] + 1), 'scores-ready <= after'),
])
def test_stamp_intervals_refuse_bad_rows(change, message):
    s = _rows()
    change(s)
    with pytest.raises(ValueError, match=message):
        wc.stamp_intervals(s, [0, 1, 2], [0, 1, 2], L)


def test_stamp_intervals_refuse_other_coverage_and_dtype():
    with pytest.raises(ValueError, match='cover steps'):
        wc.stamp_intervals(_rows(), [0, 1, 1], [0, 1, 2], L)
    with pytest.raises(ValueError, match='expected int64'):
        wc.stamp_intervals(_rows().double(), [0, 1, 2], [0, 1, 2], L)
    with pytest.raises(ValueError, match='expected int64'):
        wc.stamp_intervals(_rows(2), [0, 1], [0, 1], L + 1)
