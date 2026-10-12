"""The whole PrefetchHooks schedule inside a captured CUDA graph, on a real offload cache.

One captured "decode token" runs, for every layer, what FreeToken's MoE layer runs:
observe, ensure_experts (the real LRU kernel), then the hooks' copy schedule. A short
sleep per layer stands in for the expert GEMM, so the pump has time to stage experts
before their target layers, as in serving. Two requests of several tokens each replay it.

Checked after every token:
- every resident cache slot holds the bytes of its expert's host row (so every staged
  copy that demand consumed was complete and of the right expert);
- with a real bank: the raw features are token t-1's, the probabilities equal the bank
  on those features, and the prior moved only from step 1 on;
- request state starts from zero on the second request.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

# A local copy of the published predictor (huggingface-cli download RASMUS/qwen3.5-35b-a3b-fp8-expert-predictor
# --local-dir DIR). The tests that need it skip when COMMITMOE_PREDICTOR_DIR is unset.
PUBLISHED = Path(os.environ.get('COMMITMOE_PREDICTOR_DIR', '/nonexistent'))
L, E, K, D = 40, 256, 8, 2048
FEATS = [8192, 512, 256, 4096, 512, 256]  # mxfp4_triton bank row bytes, as tests/moe/test_fused_copy.py
CACHE_SLOTS = 600   # under two tokens of routes (2 x 320), so most routes miss again later
SLEEP_CYCLES = 200_000  # per layer; roughly 0.1 ms, the order of one layer's expert GEMM
STEPS = 6


def _cfg():
    from commitmoe_prefetch.spec import PredictorConfig
    if not (PUBLISHED / 'config.json').is_file():
        pytest.skip('published folder absent')
    return PredictorConfig.model_validate(json.loads((PUBLISHED / 'config.json').read_text()))


def _cache():
    from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache
    dev = torch.device('cuda')
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=CACHE_SLOTS, device=dev, cache_policy='lru',
                            prefill_overlap=False, quant_format='mxfp4_triton')
    hosts = {name: torch.randint(0, 256, (L * E, feat), dtype=torch.uint8, device=dev)
             for name, feat in zip(_BANK_SCHEMAS['mxfp4_triton'], FEATS)}
    cache.set_bank_sources({name: list(t.split(E)) for name, t in hosts.items()})
    return cache, list(hosts.values())


class _Inputs:
    def __init__(self):
        dev = torch.device('cuda')
        self.hidden = torch.zeros(L, 1, D, dtype=torch.bfloat16, device=dev)
        self.logits = torch.zeros(L, 1, E, dtype=torch.bfloat16, device=dev)
        self.ids = torch.zeros(L, 1, K, dtype=torch.int32, device=dev)
        self.work = torch.zeros_like(self.ids)  # ensure_experts rewrites ids into slots in place

    def draw(self, generator):
        self.hidden.copy_(torch.randn(L, 1, D, generator=generator).bfloat16())
        self.logits.copy_(torch.randn(L, 1, E, generator=generator).bfloat16())
        self.ids.copy_(torch.stack([torch.randperm(E, generator=generator)[:K] for _ in range(L)])[:, None]
                       .int())


def _decode(hooks, cache, inputs, sleep=SLEEP_CYCLES):
    for layer in range(L):
        inputs.work[layer].copy_(inputs.ids[layer])
        hooks.observe(layer, inputs.hidden[layer], inputs.logits[layer], inputs.work[layer])
        cache.ensure_experts(layer, inputs.work[layer])
        assert hooks.copy_missing(cache, layer)
        if sleep:
            torch.cuda._sleep(sleep)


def _capture(hooks, cache, inputs, sleep=SLEEP_CYCLES):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _decode(hooks, cache, inputs, sleep)  # eager warmup, as the graph runner does
    torch.cuda.current_stream().wait_stream(side)
    hooks.begin_request(-1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _decode(hooks, cache, inputs, sleep)
    return graph


def _check_resident_rows(cache, hosts):
    gids = cache.id_of_slot.long()
    resident = (gids >= 0).nonzero().flatten()
    for (_, bank), host in zip(cache.banks, hosts):
        assert torch.equal(bank[resident], host[gids[resident]])
    return int(resident.numel())


class _ChosenBank:
    """Probabilities set by the test: high for the experts the next token will route."""

    def __init__(self, cfg, state, device):
        self.nbytes = 0
        self.out = torch.zeros(len(cfg.target_layers), E, device=device)

    def __call__(self, raw):
        return self.out.clone()


def _hooks(monkeypatch, bank_class=None):
    from commitmoe_prefetch import bank_loader
    from commitmoe_prefetch import freetoken as ft
    cfg = _cfg()
    if bank_class is not None:
        monkeypatch.setattr(ft, 'Int8Bank', bank_class)
        state = {}
    else:
        if not (PUBLISHED / 'model.safetensors').is_file():
            pytest.skip('published weights absent')
        cfg, state = bank_loader.load(PUBLISHED)
    hooks = ft.PrefetchHooks(cfg, state, demand_stripes=16)
    hooks.reserve(SimpleNamespace(device=torch.device('cuda')))
    cache, hosts = _cache()
    hooks.bind(None, cache)
    return hooks, cache, hosts


@pytest.mark.parametrize('sleep', [SLEEP_CYCLES, 0])
def test_staged_copies_are_consumed_byte_exact(monkeypatch, sleep):
    """With the sleep every selected expert lands in time. Without it the layers run faster
    than the pump, so pumps yield to the demand copy, sections expire and some experts stay
    partial; demand must then read the host, and the rows must still be exact."""
    hooks, cache, hosts = _hooks(monkeypatch, _ChosenBank)
    inputs = _Inputs()
    graph = _capture(hooks, cache, inputs, sleep)
    generator = torch.Generator().manual_seed(0)
    for uid in (1, 2):
        hooks.begin_request(uid)
        assert not any(hooks.summary().values())
        for step in range(STEPS):
            inputs.draw(generator)
            # the scores used during this token favour exactly the experts it routes
            hooks.bank.out.zero_()
            routes = inputs.ids[hooks.first:, 0].long()
            hooks.bank.out.scatter_(1, routes, torch.linspace(0.9, 0.5, K, device='cuda').expand(len(routes), K))
            hooks.before_decode(step)
            graph.replay()
            hooks.after_decode(step, torch.tensor([7], device='cuda'), torch.tensor([9], device='cuda'))
            torch.cuda.synchronize()
            assert _check_resident_rows(cache, hosts) > 0
            if step == 0:
                assert hooks.summary()['published'] == 0  # first prefetch step is 1
        summary = hooks.summary()
        assert summary['misses_from_stage'] <= summary['published'], summary
        if sleep:
            assert summary['published'] == 16 * (STEPS - 1), summary  # caps sum to 16 per token
            assert summary['misses_from_stage'] == summary['published'], summary
        else:
            assert summary['yields_to_demand'] + summary['expired_calls'] > 0, summary


def test_scores_use_end_of_token_features(monkeypatch):
    from commitmoe_prefetch.reference_scorer import advance_prior
    hooks, cache, hosts = _hooks(monkeypatch)
    inputs = _Inputs()
    graph = _capture(hooks, cache, inputs)
    generator = torch.Generator().manual_seed(1)
    previous = None
    for uid in (1, 2):
        hooks.begin_request(uid)
        previous = None
        for step in range(4):
            inputs.draw(generator)
            prior_before = hooks.raw['prior_before'].clone()
            hooks.before_decode(step)
            graph.replay()
            torch.cuda.synchronize()
            raw = hooks.raw  # checked before after_decode writes the next token's ids
            if previous is None:
                assert torch.equal(raw['prior_before'], prior_before)  # step 0: the prior holds
            else:
                hidden, ids = previous
                lo = hooks.first
                assert torch.equal(raw['h'], hidden[lo:, 0].half())
                assert torch.equal(raw['curr'], ids[lo:, 0].long())
                assert torch.equal(raw['win'][:, 0], ids[lo - 1:-1, 0].long())
                assert int(raw['tid']) == 100 + step - 1 and int(raw['upcoming_tid']) == 200 + step - 1
                assert torch.equal(raw['prior_before'], advance_prior(prior_before, raw['curr'], 0.9))
                # the bank on the same inputs, with the prior as it was when the graph scored
                assert torch.equal(hooks.probs, hooks.bank({**raw, 'prior_before': prior_before}))
            assert _check_resident_rows(cache, hosts) > 0
            hooks.after_decode(step, torch.tensor([100 + step], device='cuda'),
                               torch.tensor([200 + step], device='cuda'))
            previous = (inputs.hidden.clone(), inputs.ids.clone())
