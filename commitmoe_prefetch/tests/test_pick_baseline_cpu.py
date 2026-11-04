"""CPU tests of the EXP600 baseline picks: the hash, the scores, the switches and flag-off behavior. No device."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from commitmoe_prefetch import freetoken as ft
from commitmoe_prefetch.spec import PredictorConfig

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


def test_ema_scores_keep_every_unequal_prior_in_exact_order():
    """The whole range of one use in a 512-token answer, plus a pair one fp32 step apart. A jitter would reorder that pair."""
    used = 0.1 * 0.9 ** torch.linspace(0, 511, 200, dtype=torch.float64)
    prior = torch.zeros(1, E)
    prior[0, :200] = used.to(torch.float32)
    prior[0, 200] = torch.nextafter(prior[0, 100], torch.tensor(0.0))   # just under row 100
    u = ft.baseline_scores('ema', prior, IDS[:1], step(5), scalar(1), 600)
    assert torch.equal(u[0, :201], prior[0, :201])
    order = u[0, :201].argsort(descending=True, stable=True).tolist()
    assert order.index(100) + 1 == order.index(200)


def test_ema_scores_randomize_only_the_never_used_experts_and_rank_them_last():
    prior = torch.zeros(1, E)
    prior[0, 0] = 0.1 * 0.9 ** 511   # the smallest prior in the registered 512-token maximum
    u = ft.baseline_scores('ema', prior, IDS[:1], step(5), scalar(1), 600)
    unused = u[0, 1:]
    assert unused.max() < u[0, 0] and unused.max() < 1e-30
    assert unused.unique().numel() > 200    # 255 hash draws of 24 bits: fp32 keeps them apart
    assert unused.min() == 0 or unused.min() >= torch.finfo(torch.float32).tiny   # never a subnormal that a flush would zero
    assert unused.argmax() != 0     # a plain prior would give the kernel's lowest id


def test_ema_ties_above_zero_keep_the_kernel_rule():
    prior = torch.zeros(1, E)
    prior[0, [7, 9, 200]] = 0.25
    u = ft.baseline_scores('ema', prior, IDS[:1], step(5), scalar(1), 600)
    assert u[0, 7] == u[0, 9] == u[0, 200] == 0.25
    assert int(u[0].argmax()) == 7


def test_baselines_repeat_for_one_seed_and_change_with_step_and_request():
    prior = torch.zeros(H, E)
    for kind in ft.PICK_BASELINES:
        a = ft.baseline_scores(kind, prior, IDS, step(9), scalar(2), 600)
        assert torch.equal(a, ft.baseline_scores(kind, prior, IDS, step(9), scalar(2), 600))
        for other in (ft.baseline_scores(kind, prior, IDS, step(10), scalar(2), 600),
                      ft.baseline_scores(kind, prior, IDS, step(9), scalar(3), 600)):
            assert (a != other).float().mean() > 0.99


def test_random_scores_keep_the_top_of_a_section_apart_on_this_step():
    """select_scored takes at most 3 per section from 5 x 256 experts. 24-bit scores can still tie on another step, and
    select_scored then takes the lower gid. This is one seeded draw, not a proof that ties never occur."""
    a = ft.hash_unit(IDS[:5], step(1), scalar(1), 600).flatten()
    assert a.topk(16).values.unique().numel() == 16


@pytest.fixture
def cfg():
    published = Path(os.environ.get('COMMITMOE_PREDICTOR_DIR', '/nonexistent'))
    if not (published / 'config.json').is_file():
        pytest.skip('published folder absent')
    return PredictorConfig.model_validate(json.loads((published / 'config.json').read_text()))


class _StubBank:
    """A fixed bank output, so that a test can see whether the pick replaced it."""

    def __init__(self, cfg, state, device):
        self.nbytes = 0
        self.out = torch.full((len(cfg.target_layers), cfg.base_model.num_experts), 0.5)

    def __call__(self, raw):
        return self.out


@pytest.fixture
def make_hooks(cfg, monkeypatch):
    monkeypatch.setattr(ft, 'Int8Bank', _StubBank)
    monkeypatch.setattr(torch.cuda, 'Stream', lambda **kwargs: None)  # no device on the CPU runner

    def make(**kwargs):
        hooks = ft.PrefetchHooks(cfg, {}, demand_stripes=16, **kwargs)
        hooks.reserve(SimpleNamespace(device=torch.device('cpu')))
        hooks.begin_request(-1)
        hooks.score_inputs['routes'][:, :] = torch.arange(8)
        return hooks
    return make


def test_flag_off_keeps_the_bank_scores(make_hooks):
    hooks = make_hooks()
    assert hooks.pick is None and not hooks.layer_counters
    assert not hasattr(hooks, 'pick_ids') and not hasattr(hooks, 'pick_requests') and not hasattr(hooks, 'layer_published')
    hooks.before_decode(4)
    hooks._score()
    assert hooks.probs.eq(0.5).all()


@pytest.mark.parametrize('kind', ft.PICK_BASELINES)
def test_a_pick_replaces_the_bank_scores_and_the_prior_moves_the_same(make_hooks, kind):
    off, on = make_hooks(), make_hooks(pick=(kind, 600))
    for hooks in (off, on):
        hooks.before_decode(4)
        hooks._score()
    assert not on.probs.eq(0.5).any()
    assert torch.equal(on.raw['prior_before'], off.raw['prior_before'])
    if kind == 'ema':
        moved = on.raw['prior_before'] > 0
        assert moved.any() and torch.equal(on.probs[moved], on.raw['prior_before'][moved])


def test_the_counter_flag_allocates_and_resets_its_buffer(make_hooks):
    hooks = make_hooks(layer_counters=True)
    assert hooks.layer_counters and hooks.layer_published.tolist() == [0] * 40
    hooks.layer_published += 1
    hooks.begin_request(11)
    assert hooks.layer_published.tolist() == [0] * 40


def test_begin_request_counts_requests_for_the_hash_and_resets_the_prior(make_hooks):
    hooks = make_hooks(pick=('ema', 600))
    start = int(hooks.pick_requests)
    hooks.before_decode(4)
    hooks._score()
    assert hooks.raw['prior_before'].any()
    hooks.begin_request(11)
    assert int(hooks.pick_requests) == start + 1 and not hooks.raw['prior_before'].any()


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
