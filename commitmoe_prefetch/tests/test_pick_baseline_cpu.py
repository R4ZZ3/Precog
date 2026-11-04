"""CPU tests of the EXP600 baseline picks: the hash, the scores and the switches. No model, no device."""
from __future__ import annotations

import pytest
import torch

from commitmoe_prefetch import freetoken as ft

E = 256
H = 34
FIRST = 6
IDS = torch.arange(FIRST * E, (FIRST + H) * E).view(H, E)


def scalar(v: int) -> torch.Tensor:
    return torch.tensor([v], dtype=torch.int64)


def step(v: int) -> torch.Tensor:
    return torch.tensor([v], dtype=torch.int32)


def test_the_hash_is_a_fixed_function_of_its_four_inputs():
    a = ft.hash_unit(IDS, step(3), scalar(2), 600)
    assert torch.equal(a, ft.hash_unit(IDS, step(3), scalar(2), 600))
    for other in (ft.hash_unit(IDS, step(4), scalar(2), 600), ft.hash_unit(IDS, step(3), scalar(3), 600),
                  ft.hash_unit(IDS, step(3), scalar(2), 601)):
        assert (a != other).float().mean() > 0.99


def test_the_hash_stays_in_the_unit_interval_in_fp32():
    u = torch.cat([ft.hash_unit(IDS, step(s), scalar(1), 600).flatten() for s in range(64)])
    assert u.dtype == torch.float32 and u.min() >= 0 and u.max() < 1
    # Ten equal bins over 557056 values: each bin holds about 10 %. 3 standard deviations are under 0.13 % of a bin share.
    share = torch.histc(u, bins=10, min=0.0, max=1.0) / u.numel()
    assert ((share - 0.1).abs() < 0.002).all()


def test_the_pick_of_a_section_does_not_favor_an_expert_or_a_step():
    """select_scored takes the top score of 5 layers x 256 experts. Over many steps each expert must win about equally
    often, and consecutive steps must repeat a winner no more often than chance. Adjacent ids and steps are the inputs most at risk."""
    wins = torch.zeros(5 * E)
    last, repeats = None, 0
    for s in range(20000):
        top = int(ft.hash_unit(IDS[:5], step(s), scalar(1), 600).flatten().argmax())
        wins[top] += 1
        repeats += top == last
        last = top
    # 20000 wins over 1280 experts: a mean of 15.6 and a Poisson standard deviation of about 4. Independent picks repeat
    # the last winner with chance 1 / 1280, so repeats have the same mean and standard deviation.
    assert wins.max() <= 15.6 + 6 * 4 and wins.min() >= 1
    assert repeats <= 15.6 + 6 * 4


def test_random_scores_ignore_the_prior():
    prior = torch.rand(H, E)
    a = ft.baseline_scores('random', prior, IDS, step(5), scalar(1), 600)
    assert torch.equal(a, ft.baseline_scores('random', torch.zeros(H, E), IDS, step(5), scalar(1), 600))


def test_ema_scores_keep_the_order_of_used_experts_and_break_ties_among_unused():
    prior = torch.zeros(1, E)
    used = 0.1 * 0.9 ** torch.arange(0, 200, dtype=torch.float32)   # last use from 0 to 199 tokens ago
    prior[0, :200] = used
    u = ft.baseline_scores('ema', prior, IDS[:1], step(5), scalar(1), 600)
    assert u[0, :200].argsort(descending=True).tolist() == list(range(200))
    unused = u[0, 200:]
    assert (unused > 0).all() and unused.unique().numel() > 50 and unused.max() < used.min()
    assert unused.argmax() != 0   # a plain prior would give the kernel's lowest id


def test_the_tie_break_is_under_a_use_240_tokens_ago():
    assert ft.EMA_TIE_BREAK <= 0.1 * 0.9 ** 240 * 1.1


@pytest.mark.parametrize('env, want', [({}, None), ({'FREETOKEN_PICK_BASELINE': ''}, None),
                                       ({'FREETOKEN_PICK_BASELINE': 'random', 'FREETOKEN_PICK_SEED': '600'}, ('random', 600)),
                                       ({'FREETOKEN_PICK_BASELINE': 'ema', 'FREETOKEN_PICK_SEED': '600'}, ('ema', 600))])
def test_the_switch_reads_kind_and_seed(env, want):
    assert ft.pick_baseline_from_env(env) == want


@pytest.mark.parametrize('env', [{'FREETOKEN_PICK_BASELINE': 'lru', 'FREETOKEN_PICK_SEED': '1'},
                                 {'FREETOKEN_PICK_BASELINE': 'random'},
                                 {'FREETOKEN_PICK_BASELINE': 'ema', 'FREETOKEN_PICK_SEED': '-1'}])
def test_the_switch_refuses_an_unknown_kind_or_a_missing_seed(env):
    with pytest.raises(ValueError):
        ft.pick_baseline_from_env(env)
