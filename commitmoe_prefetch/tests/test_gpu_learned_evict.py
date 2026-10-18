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
- with a version 2 bundle (EXP540): every live raw input equals the offline definition that the
  borrowed rows were trained on, and rows that share a source head read that head's trunk.

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


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('start_layer', [0, 8])
def test_live_rule_in_graph(start_layer, split):
    """``split``: the overlap schedule (EXP538) must give the same decisions and scores."""
    from freetoken.moe.learned_evict import LearnedEvict, learned_ensure
    hooks, cache, hosts = _hooks(start_layer)
    rule = hooks.learned
    inputs = _Inputs()
    graph = _capture(hooks, cache, inputs, split=split)
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
                  torch.device('cuda'), release_sha256='0' * 64, num_layers=L)


def test_flag_from_env():
    from commitmoe_prefetch.freetoken import EVICT_ENV, EVICT_START_ENV, evict_from_env
    assert evict_from_env({}) is None
    assert evict_from_env({EVICT_ENV: ''}) is None
    assert evict_from_env({EVICT_ENV: '/b.pt', EVICT_START_ENV: '8'}) == (Path('/b.pt'), 8)
    with pytest.raises(ValueError, match=EVICT_START_ENV):
        evict_from_env({EVICT_ENV: '/b.pt'})


def test_shapes_are_the_bank_shapes():
    """One keep row per owner layer, every source layer a predicted target layer."""
    if not BUNDLE.is_file():
        pytest.skip('eviction bundle absent')
    from commitmoe_prefetch.evict import EvictRows
    cfg = _cfg()
    rows = EvictRows(BUNDLE, list(cfg.target_layers), cfg.architecture.up_dim, E,
                     cfg.serving_reference.int8_group, torch.device('cuda'), release_sha256=None, num_layers=L)
    assert rows.codes.shape == (len(rows.layers), E, cfg.architecture.up_dim)
    assert all(0 <= x < L for x in rows.layers)
    assert all(cfg.target_layers[0] <= x < L for x in rows.source_layers)
    assert K == cfg.base_model.top_k


def test_v2_borrowed_rows_equal_the_rows_on_their_source_head(tmp_path):
    """A version 2 row of owner L0-L5 with source L6 gives the keep score of the same row placed on
    owner L6 in a version 1 bundle, bit for bit, also when six rows share the one source head."""
    from commitmoe_prefetch.evict import EvictRows
    cfg = _cfg()
    up, group, targets = cfg.architecture.up_dim, cfg.serving_reference.int8_group, list(cfg.target_layers)
    torch.manual_seed(0)
    w = (torch.randn(7, E, up) * 0.02).bfloat16()
    b = (torch.randn(7, E) * 0.5).bfloat16()
    common = dict(horizon_tokens=8, release_sha256='a' * 64)
    torch.save(dict(version=2, layers=[0, 1, 2, 3, 4, 5, 8], source_layers=[6] * 6 + [8], weight=w, bias=b,
                    **common), tmp_path / 'v2.pt')
    v2 = EvictRows(tmp_path / 'v2.pt', targets, up, E, group, torch.device('cuda'), 'a' * 64, L)
    g = torch.randn(len(targets), up, device='cuda').bfloat16()
    keep = torch.empty(7, E, device='cuda')
    v2(g, keep)
    for m, (owner, source) in enumerate(zip(v2.layers, v2.source_layers)):
        torch.save(dict(version=1, layers=[source], weight=w[m:m + 1].clone(), bias=b[m:m + 1].clone(), **common),
                   tmp_path / f'v1_{m}.pt')
        v1 = EvictRows(tmp_path / f'v1_{m}.pt', targets, up, E, group, torch.device('cuda'), 'a' * 64, L)
        one = torch.empty(1, E, device='cuda')
        v1(g, one)
        assert torch.equal(keep[m], one[0]), (owner, source)
    assert torch.isfinite(keep).all() and not torch.equal(keep[0], keep[1])  # distinct rows, one source


def _offline_raw(cfg, history, t):
    """Raw inputs of token t of a request by the offline definition (experiments/exp537_borrowed.py,
    raw_from_trace), built only from the routes, hidden states and logits the test drew.
    ``history[s]`` = (hidden, logits, ids) of decode step s, on the CPU."""
    feats = cfg.features
    targets = list(cfg.target_layers)
    o = feats.hidden_source_offset
    hidden, logits, ids = history[t]
    prior = torch.zeros(len(targets), E)
    for s in range(t):  # the EMA over tokens 0 .. t-1, in FP32 as the offline builder
        routes = history[s][2][targets, 0].long()
        prior *= feats.ema_beta
        prior.scatter_(1, routes, prior.gather(1, routes) + (1 - feats.ema_beta))
    zeros = torch.zeros(len(targets), K, dtype=torch.long)  # serving resets prevwin to expert id 0
    prev = [history[t - 1][2][[T - d for T in targets], 0].long() if t > 0 else zeros
            for d in range(1, feats.previous_token_window + 1)]
    rl = [torch.stack([torch.zeros(E, dtype=torch.float16) if T - d in feats.router_logit_layers_zeroed
                       else logits[T - d, 0].half() for T in targets])
          for d in range(1, feats.router_logit_window + 1)]
    return dict(h=hidden[[T + o for T in targets], 0].half(), h_prev=hidden[[T + o - 1 for T in targets], 0].half(),
                win=torch.stack([ids[[T - d for T in targets], 0].long() for d in range(1, feats.route_window + 1)], 1),
                prevwin=torch.stack(prev, 1), curr=ids[targets, 0].long(), rl=torch.cat(rl, 1),
                pos=torch.full((len(targets), 1), (t + 1) / feats.position_scale),
                tid=torch.tensor([100 + t]), upcoming_tid=torch.tensor([200 + t]), prior_before=prior)


def test_live_features_equal_the_offline_source_through_resets_and_replays():
    """EXP540 live-feature gate, for the bundle in EXP526_EVICT_BUNDLE (the 36-row version 2 bundle).

    The six L0-L5 rows were trained offline on the L6 head's inputs built by raw_from_trace. Here
    every raw input of every head (h, h_prev, the route window, the previous token's routes with
    zeros at token 0, the current routes, the router logits with L5 zero, position, token ids and
    the EMA prior) must equal that offline definition exactly, after the graph warm-ups, through two
    requests (two resets) and replays that alternate between two graphs captured on the same hooks
    (as serving captures several graphs that share the hooks' buffers and events). The prior
    advances exactly once per token, although six rows share one source head, and each keep row
    equals its own row applied to the trunk output of its source head. J = 1, the serve setting."""
    from commitmoe_prefetch.qbank import qlinear
    hooks, cache, _ = _hooks(1)
    rows, rule, cfg = hooks.evict, hooks.learned, hooks.cfg
    if rows.version != 2 or len(set(rows.source_layers)) == len(rows.source_layers):
        pytest.skip('needs a version 2 bundle in which rows share a source head')
    targets = list(cfg.target_layers)
    inputs = _Inputs()
    # Each capture runs its eager warm-up and leaves reset state; both graphs read the same buffers.
    graphs = [_capture(hooks, cache, inputs) for _ in range(2)]
    generator = torch.Generator().manual_seed(6)
    dev = torch.device('cuda')
    for uid in (1, 2):
        hooks.begin_request(uid)
        history = []
        for step in range(STEPS):
            inputs.draw(generator)
            history.append(tuple(x.cpu().clone() for x in (inputs.hidden, inputs.logits, inputs.ids)))
            prior_before = hooks.raw['prior_before'].clone()
            hooks.before_decode(step)
            graphs[(uid + step) % 2].replay()  # request 2 starts on the other graph
            torch.cuda.synchronize()
            if step == 0:
                assert int(rule.valid) == 0 and torch.equal(hooks.raw['prior_before'], prior_before)
            else:
                t = step - 1  # the replay of step s scores token s-1, the last token fully decoded
                want = _offline_raw(cfg, history, t)
                for name, value in want.items():
                    got = prior_before if name == 'prior_before' else hooks.raw[name]
                    assert torch.equal(got.cpu(), value.to(got.dtype)), (uid, step, name)
                one_step = _offline_raw(cfg, history, t + 1)['prior_before']  # exactly one EMA advance
                assert torch.equal(hooks.raw['prior_before'].cpu(), one_step), (uid, step)
                g = hooks.bank.trunk({k: v.to(dev) for k, v in want.items()})
                assert torch.equal(hooks.probs, hooks.bank.head(g))
                for m, source in enumerate(rows.source_layers):
                    head = targets.index(source)
                    y = qlinear(g[head:head + 1].contiguous(), rows.codes[m:m + 1], rows.scale[m:m + 1],
                                rows.bias[m:m + 1], rows.group)
                    assert torch.equal(rule.keep[m], torch.sigmoid(y.float())[0]), (uid, step, rows.layers[m])
                assert int(rule.valid) == 1
            hooks.after_decode(step, torch.tensor([100 + step], device='cuda'),
                               torch.tensor([200 + step], device='cuda'))


@pytest.mark.parametrize('start_layer', [1, 8])
def test_copy_never_writes_a_resident_route_slot(start_layer):
    """EXP538 pass 1 reads the slots of the routes resident before ensure_experts while the demand
    copy runs. Under the learned rule, no slot that the copy writes (evict_slots[:num_indices])
    may be one of those slots, and every resident route keeps its slot. Eager, scored tokens."""
    hooks, cache, hosts = _hooks(start_layer)
    inputs = _Inputs()
    generator = torch.Generator().manual_seed(3)
    hooks.begin_request(1)
    checked = 0
    for step in range(STEPS):
        inputs.draw(generator)
        hooks.before_decode(step)
        for layer in range(L):
            work = inputs.work[layer]
            work.copy_(inputs.ids[layer])
            hooks.observe(layer, inputs.hidden[layer], inputs.logits[layer], work)
            before = cache.slot_for_id[layer].index_select(0, work.reshape(-1)).clone()
            cache.ensure_experts(layer, work)
            torch.cuda.synchronize()
            resident = before >= 0
            written = cache.evict_slots[:int(cache.num_indices)].long()
            assert torch.equal(work.reshape(-1)[resident], before[resident])  # ids rewritten to the same slots
            assert not torch.isin(before[resident].long(), written).any(), (step, layer)
            checked += int(resident.sum())
            hooks.begin_copy(cache, layer)()
        torch.cuda.synchronize()
        assert _check_resident_rows(cache, hosts) > 0
        hooks.after_decode(step, torch.tensor([1], device='cuda'), torch.tensor([2], device='cuda'))
    assert checked > 0 and hooks.learned.counters[2] > 0  # resident routes seen and slots rescued
