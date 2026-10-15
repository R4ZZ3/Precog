"""Learned eviction (EXP526) wired into the decode hooks, inside a captured CUDA graph.

The victim kernel itself is tested against the replay rule in ft_fork/tests/moe/test_learned_evict.py.
This test checks the wiring that those tests cannot see:

- the keep scores used during token t+1 are the eviction rows applied to the bank trunk on the
  end-of-token-t inputs, bit for bit, and the prefetch probabilities come from the same trunk call;
- valid is 0 on the first token of every request and 1 after it;
- the graph-captured cache decisions equal an eager run of the same kernel on a shadow copy of
  the cache maps, fed the same routes and the same published scores, with layers below J on LRU.
  Together with the kernel tests this makes the live decisions equal to the replay rule;
- every resident slot still holds its expert's bytes (staged and demand copies unaffected).

Needs the published predictor (COMMITMOE_PREDICTOR_DIR) and an eviction bundle from
experiments/exp526_export_evict.py (EXP526_EVICT_BUNDLE).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_gpu_schedule import E, K, L, _cache, _capture, _cfg, _check_resident_rows, _Inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

BUNDLE = Path(os.environ.get('EXP526_EVICT_BUNDLE', '/nonexistent'))
STEPS = 5


def _hooks(start_layer: int, timing: Path | None = None, record: Path | None = None, timing_all: bool = False):
    from commitmoe_prefetch import bank_loader
    from commitmoe_prefetch import freetoken as ft
    from test_gpu_schedule import PUBLISHED
    _cfg()  # skips when the published folder is absent
    if not BUNDLE.is_file():
        pytest.skip('eviction bundle absent')
    cfg, state = bank_loader.load(PUBLISHED)
    release = json.loads((PUBLISHED / 'manifest.json').read_text())['files']['model.safetensors']
    hooks = ft.PrefetchHooks(cfg, state, demand_stripes=16, evict=(BUNDLE, start_layer),
                             timing=timing, timing_all=timing_all, release_sha256=release)
    hooks.reserve(SimpleNamespace(device=torch.device('cuda')))
    cache, hosts = _cache()
    if record is not None:
        from freetoken.moe.routing_recorder import RoutingRecorder
        recorder = RoutingRecorder(hooks, record, top_k=K)
        recorder.bind(None, cache)
        assert cache.learned_evict is hooks.learned
        return recorder, cache, hosts
    hooks.bind(None, cache)
    assert cache.learned_evict is hooks.learned
    return hooks, cache, hosts


def _shadow(cache):
    """The map fields of ``cache`` that the victim kernel reads and writes, as independent copies."""
    return SimpleNamespace(slot_for_id=cache.slot_for_id.clone(), id_of_slot=cache.id_of_slot.clone(),
                           usage=cache.usage.clone(), step=cache.step.clone(),
                           src_indices=torch.empty_like(cache.src_indices),
                           evict_slots=torch.empty_like(cache.evict_slots),
                           num_indices=torch.zeros_like(cache.num_indices),
                           collect_stats=False, lru_stats=None, num_experts=cache.num_experts)


@pytest.mark.parametrize('start_layer', [0, 8])
def test_live_rule_in_graph(start_layer):
    from freetoken.moe.learned_evict import LearnedEvict, learned_ensure
    hooks, cache, hosts = _hooks(start_layer)
    rule = hooks.learned
    inputs = _Inputs()
    graph = _capture(hooks, cache, inputs)
    generator = torch.Generator().manual_seed(2)
    for uid in (1, 2):
        hooks.begin_request(uid)
        assert int(rule.valid) == 0 and not rule.keep.any() and not rule.counters.any()
        shadow = _shadow(cache)
        shadow_rule = LearnedEvict(keep=rule.keep.clone(), valid=rule.valid.clone(),
                                   target_of_layer=rule.target_of_layer, start_layer=start_layer,
                                   counters=torch.zeros_like(rule.counters))
        previous = None
        for step in range(STEPS):
            inputs.draw(generator)
            prior_before = hooks.raw['prior_before'].clone()
            hooks.before_decode(step)
            graph.replay()
            torch.cuda.synchronize()
            assert int(rule.valid) == int(step > 0)
            if previous is not None:
                # The trunk read token t's inputs (saved before this replay), not token t+1's.
                hidden, ids = previous
                lo = hooks.first
                assert torch.equal(hooks.raw['h'], hidden[lo:, 0].half())
                assert torch.equal(hooks.raw['curr'], ids[lo:, 0].long())
                assert torch.equal(hooks.raw['win'][:, 0], ids[lo - 1:-1, 0].long())
                assert int(hooks.raw['tid']) == 100 + step - 1 and int(hooks.raw['upcoming_tid']) == 200 + step - 1
            g = hooks.bank.trunk({**hooks.raw, 'prior_before': prior_before})
            expect = torch.empty_like(rule.keep)
            hooks.evict(g, expect)
            assert torch.equal(rule.keep, expect)
            assert torch.equal(hooks.probs, hooks.bank.head(g))
            # The same token, eagerly, on the shadow maps with the scores the graph published.
            shadow_rule.keep.copy_(rule.keep)
            shadow_rule.valid.copy_(rule.valid)
            for layer in range(L):
                learned_ensure(shadow, layer, inputs.ids[layer].clone(), shadow_rule)
            for name in ('slot_for_id', 'id_of_slot', 'usage', 'step'):
                assert torch.equal(getattr(shadow, name), getattr(cache, name)), (uid, step, name)
            assert _check_resident_rows(cache, hosts) > 0
            hooks.after_decode(step, torch.tensor([100 + step], device='cuda'),
                               torch.tensor([200 + step], device='cuda'))
            previous = (inputs.hidden.clone(), inputs.ids.clone())
        assert torch.equal(rule.counters, shadow_rule.counters)
        events, reranked, rescued = rule.counters.tolist()
        # Events at or above J with a miss, on the scored tokens (all but the first).
        assert events > 0 and reranked > 0 and rescued > 0, rule.counters
        summary = hooks.summary()
        assert summary['evict_events_scored'] == events


def test_timing_stamps(tmp_path):
    """The J diagnostic: per token, main's stamps rise layer by layer and the ready stamp exists."""
    out = tmp_path / 'stamps.pt'
    hooks, cache, _ = _hooks(L - 1, timing=out, timing_all=True)
    inputs = _Inputs()
    graph = _capture(hooks, cache, inputs)
    generator = torch.Generator().manual_seed(3)
    for uid in (1, 2):
        hooks.begin_request(uid)
        for step in range(3):
            inputs.draw(generator)
            hooks.before_decode(step)
            graph.replay()
            hooks.after_decode(step, torch.tensor([1], device='cuda'), torch.tensor([2], device='cuda'))
    torch.cuda.synchronize()
    assert len(torch.load(out)['step']) == 3  # request 2 is written only when a third request begins
    hooks.begin_request(3)
    rows = torch.load(out)
    stamps = rows['stamps']
    assert stamps.shape == (6, L + 3) and rows['step'].tolist() == [0, 1, 2, 0, 1, 2]
    assert rows['start_layer'] == L - 1
    assert bool((stamps[:, 1:L] >= stamps[:, :L - 1]).all())  # main reaches the layers in order
    assert bool((stamps[:, L] >= stamps[:, 0]).all())  # scores are ready after the fork at layer 0
    assert bool((stamps[:, L + 1] >= stamps[:, L - 1]).all())  # main at J (= 39), before the wait
    assert bool((stamps[:, L + 2] >= stamps[:, L + 1]).all())  # after the wait
    assert bool((stamps[:, L + 2] >= stamps[:, L]).all())  # the wait ended after the ready stamp


def test_recorder_keeps_published_scores(tmp_path):
    """The fidelity replay reads keep[r] and keep_valid[r]: the scores decode step r used."""
    recorder, cache, _ = _hooks(8, record=tmp_path)
    hooks = recorder.inner
    inputs = _Inputs()
    graph = _capture(recorder, cache, inputs)
    generator = torch.Generator().manual_seed(4)
    used = []
    for uid in (1, 2):
        recorder.begin_request(uid)
        for step in range(3):
            inputs.draw(generator)
            recorder.before_decode(step)
            graph.replay()
            torch.cuda.synchronize()
            if uid == 1:
                used.append((hooks.learned.keep.clone(), int(hooks.learned.valid)))
            recorder.after_decode(step, torch.tensor([1], device='cuda'), torch.tensor([2], device='cuda'))
    rec = torch.load(next(tmp_path.glob('request_000_*.pt')))
    assert rec['keep_valid'].tolist() == [v for _, v in used] == [0, 1, 1]
    assert all(torch.equal(rec['keep'][r].cuda(), k) for r, (k, _) in enumerate(used))
    assert rec['evict_layers'] == hooks.evict.layers and rec['evict_start_layer'] == 8


def test_nonfinite_scores_fall_back_to_lru():
    """A NaN keep score must never reach the victim kernel: the token publishes valid = 0."""
    hooks, cache, _ = _hooks(0)
    hooks.evict.bias[0, 3] = float('nan')
    inputs = _Inputs()
    graph = _capture(hooks, cache, inputs)
    generator = torch.Generator().manual_seed(5)
    hooks.begin_request(1)
    for step in range(3):
        inputs.draw(generator)
        hooks.before_decode(step)
        graph.replay()
        torch.cuda.synchronize()
        assert int(hooks.learned.valid) == 0
        hooks.after_decode(step, torch.tensor([1], device='cuda'), torch.tensor([2], device='cuda'))
    summary = hooks.summary()
    assert summary['evict_nonfinite_tokens'] == 2  # steps 1 and 2; step 0 is never scored
    assert summary['evict_events_scored'] == 0


def test_bundle_from_other_release_is_refused():
    from commitmoe_prefetch.evict import EvictRows
    if not BUNDLE.is_file():
        pytest.skip('eviction bundle absent')
    cfg = _cfg()
    with pytest.raises(ValueError, match='served release'):
        EvictRows(BUNDLE, list(cfg.target_layers), cfg.architecture.up_dim, E, cfg.serving_reference.int8_group,
                  torch.device('cuda'), release_sha256='0' * 64)


def test_flag_from_env():
    from commitmoe_prefetch.freetoken import EVICT_ENV, EVICT_START_ENV, evict_from_env
    assert evict_from_env({}) is None
    assert evict_from_env({EVICT_ENV: ''}) is None
    assert evict_from_env({EVICT_ENV: '/b.pt', EVICT_START_ENV: '8'}) == (Path('/b.pt'), 8)
    with pytest.raises(ValueError, match=EVICT_START_ENV):
        evict_from_env({EVICT_ENV: '/b.pt'})


def test_shapes_are_the_bank_shapes():
    """One keep row per bundle layer, every bundle layer a predicted target layer."""
    if not BUNDLE.is_file():
        pytest.skip('eviction bundle absent')
    from commitmoe_prefetch.evict import EvictRows
    cfg = _cfg()
    rows = EvictRows(BUNDLE, list(cfg.target_layers), cfg.architecture.up_dim, E,
                     cfg.serving_reference.int8_group, torch.device('cuda'), release_sha256=None)
    assert rows.codes.shape == (len(rows.layers), E, cfg.architecture.up_dim)
    assert all(cfg.target_layers[0] <= x < L for x in rows.layers)
    assert K == cfg.base_model.top_k
