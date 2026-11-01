"""Shared learned eviction (EXP585, R584) on the GPU: the gates of the EXP585 GPU correctness window.

P0  The served kernels against the CPU reference fixtures of experiments/exp585_fixtures.py: the eviction
    state (ema, age) is bit-exact at every fixture event, the age table holds the fixture's bytes, the FP32
    own features and the assembled BF16 inputs equal the CPU fork path, and every qgemv output is within
    the declared tolerance of reference_linear on the same input (exp585_window_checks.qgemv_ratio; any
    nonfinite value fails).
P1  Inside a captured CUDA graph (one-pass and split schedules, J = 1): per token, the residency snapshot,
    keep, valid, the eviction state and the prefetch probabilities equal an eager run on the same frozen
    inputs bit for bit, and the victim kernel on shadow maps makes the same decisions; step 0 of every
    request is invalid and a request starts from the reset state.
P2  GPU keep scores against the emulator scores stored in the fixtures (same exported codes): bottom-16
    resident overlap mean >= 0.99. The real victim interface: the victim kernel fed the GPU scores and fed
    the emulator scores, from the same maps and routes, chooses the same victim for >= 99% of evictions;
    the evictions where the learned choice differs from LRU must exist and are reported by layer.
a2  Engineering gate, the agreed microbenchmark: on an idle GPU, each arm's probe-stream prefix (main's
    observe(0) stamp, the residency snapshot for R584, the raw-feature preparation and the scorer, up to
    the scores-ready stamp) as one captured graph; 256 MiB written before every replay (cold L2); per
    block 20 warm-ups and 200 timed replays; blocks base, candidate, candidate, base. Base: r36 (EvictRows
    on the bank trunk). Candidate: R584. Gate: candidate median <= 150 us.
    Diagnostic only (no gate): the prefix and main's wait at J inside the captured split decode of the
    test harness (600 slots, a sleep kernel stands in for each layer's expert GEMM).

Inputs are bound: EXP585_BUNDLE, EXP585_FIXTURES, EXP526_EVICT_BUNDLE (r36) and the release weights must
hash to EXP585_BUNDLE_SHA256, EXP585_FIXTURES_SHA256, EXP526_EVICT_BUNDLE_SHA256 and EXP585_RELEASE_SHA256,
or the test skips; experiments/exp585_gpu_window.py sets them and refuses any skip. Every test writes its
numbers as one JSON line to EXP585_GPU_REPORT, which must not exist when the session starts.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import exp585_window_checks as wc
from test_gpu_learned_evict import _shadow
from test_gpu_schedule import PUBLISHED, D, E, K, L, _cache, _capture, _cfg, _check_resident_rows, _Inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

BUNDLE = Path(os.environ.get('EXP585_BUNDLE', '/nonexistent'))
FIXTURES = Path(os.environ.get('EXP585_FIXTURES', '/nonexistent'))
R36 = Path(os.environ.get('EXP526_EVICT_BUNDLE', '/nonexistent'))
PINS = {name: os.environ.get(name) for name in ('EXP585_BUNDLE_SHA256', 'EXP585_FIXTURES_SHA256',
                                                'EXP526_EVICT_BUNDLE_SHA256', 'EXP585_RELEASE_SHA256')}
REPORT = os.environ.get('EXP585_GPU_REPORT')
REPORT_EXISTED = bool(REPORT) and Path(REPORT).exists()  # an earlier attempt's report is never appended to
CUDA = torch.device('cuda')
START = 1           # J of the plan: main waits for the scores at layer 1
STEPS = 6           # decode steps per request in the graph tests (step 0 included)
DIAG_STEPS = 40     # per request in the in-schedule diagnostic; 2 requests
OVERLAP_GATE = 0.99
VICTIM_GATE = 0.99
A2_GATE_NS = 150_000
FLUSH_BYTES = 256 * 2**20   # 4x the 64 MiB L2 of the RTX 4080
WARMUP, TIMED = 20, 200


def _report(name: str, **values) -> None:
    if not REPORT:
        return
    assert not REPORT_EXISTED, f'{REPORT} existed before this session'
    with open(REPORT, 'a') as fh:
        fh.write(json.dumps(dict(test=name, **values)) + '\n')


@functools.cache
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def _pinned(path: Path, pin: str) -> Path:
    if not path.is_file() or not PINS[pin]:
        pytest.skip(f'{path} or {pin} absent')
    assert _sha256(path) == PINS[pin], f'{path} does not hash to {pin}'
    return path


def _release() -> tuple[str, torch.Tensor]:
    """The release's model.safetensors hash (its bytes, equal to the manifest and the pin) and its table."""
    from safetensors import safe_open
    _cfg()  # skips when the published folder is absent
    _pinned(BUNDLE, 'EXP585_BUNDLE_SHA256')
    weights = _pinned(PUBLISHED / 'model.safetensors', 'EXP585_RELEASE_SHA256')
    release = json.loads((PUBLISHED / 'manifest.json').read_text())['files']['model.safetensors']
    assert release == PINS['EXP585_RELEASE_SHA256'], 'manifest and release weights differ'
    with safe_open(str(weights), framework='pt') as fh:
        table = fh.get_tensor('table')
    return release, table


def _scorers(record: list | None = None):
    """(GPU scorer with qgemv, CPU scorer with reference_linear, GPU table, CPU table). ``record`` collects
    every qgemv call of the GPU scorer as (x, codes, scale, bias, group, y)."""
    from commitmoe_prefetch import shared_evict as se
    from commitmoe_prefetch.qbank import qlinear
    release, table = _release()
    meta, t = se.load_bundle(BUNDLE, release, L, E, K, D, table, 256, 128)

    def recorded(x, codes, scale, bias, group):
        y = qlinear(x, codes, scale, bias, group)
        record.append((x, codes, scale, bias, group, y))
        return y
    gpu = se.SharedEvict(meta, t, CUDA, recorded if record is not None else qlinear)
    cpu = se.SharedEvict(meta, t, torch.device('cpu'), se.reference_linear)
    return gpu, cpu, table.to(CUDA), table


def _fixtures() -> dict:
    fx = torch.load(_pinned(FIXTURES, 'EXP585_FIXTURES_SHA256'), weights_only=False)
    assert fx['meta']['bundle_sha256'] == PINS['EXP585_BUNDLE_SHA256']
    return fx


def _args(x: dict, hidden_layers, device) -> tuple:
    h = torch.zeros(L, D, dtype=torch.float16, device=device)
    h[list(hidden_layers)] = x['hid'].to(device)
    one = dict(dtype=torch.int64, device=device)
    return (h, torch.tensor([x['tid']], **one), torch.tensor([x['up']], **one),
            torch.tensor([[x['pos']]], dtype=torch.float32, device=device))


def bottom_overlap(z_a: torch.Tensor, z_b: torch.Tensor, resident: torch.Tensor, k: int = 16) -> torch.Tensor:
    """Per layer, the share of A's k lowest-scored resident experts among B's k lowest (as exp585_fixtures)."""
    out = torch.ones(z_a.shape[0])
    for layer in range(z_a.shape[0]):
        ids = resident[layer].nonzero(as_tuple=True)[0]
        n = min(k, ids.numel())
        if n == 0:
            continue
        a = set(ids[torch.argsort(z_a[layer, ids], stable=True)[:n]].tolist())
        b = set(ids[torch.argsort(z_b[layer, ids], stable=True)[:n]].tolist())
        out[layer] = len(a & b) / n
    return out


# --- P0: served kernels against the CPU reference ------------------------------------------------------

def test_p0_state_features_and_kernels():
    from commitmoe_prefetch import shared_evict as se
    fx = _fixtures()
    calls: list = []
    gpu, cpu, table, table_cpu = _scorers(calls)
    assert torch.equal(gpu.age_table.cpu(), fx['age_table']) and torch.equal(fx['age_table'], se.age_table(gpu.far))
    f, bf = gpu.f, torch.bfloat16
    names = ('w_layer', 'h_proj', 'fc1', 'fc2', 'we', 'wb', 'wb64')
    events = scored = exact = total = hn_changed = 0
    worst, hn_diff = 0.0, 0.0
    for e in fx['events']:
        if e['kind'] == 'reset':
            gpu.reset()
            cpu.reset()
        else:
            x = e['inputs']
            gpu.advance(x['routes'].to(CUDA), torch.tensor([e['step']], dtype=torch.int32, device=CUDA))
            cpu.advance(x['routes'], torch.tensor([e['step']], dtype=torch.int32))
        events += 1
        assert torch.equal(gpu.ema.cpu(), e['expect']['ema']), (events, e['kind'])
        assert torch.equal(gpu.age.cpu(), e['expect']['age'].long()), (events, e['kind'])
        if not e['expect']['valid']:
            continue
        scored += 1
        res = x['res'].float()
        own = gpu.own(res.to(CUDA))
        assert torch.equal(own.cpu(), cpu.own(res)), events
        calls.clear()
        gpu.heads(own, *_args(x, fx['meta']['hidden_layers'], CUDA), table)
        assert [c[1].shape for c in calls] == [gpu.t[n + '_q'].shape for n in names]
        # The assembled BF16 inputs: own features bit-exact; the fc1 input is the concatenation of the
        # w_layer output, the two table rows, pos and the h_proj output, bit for bit.
        assert torch.equal(calls[0][0].cpu(), cpu.own(res).permute(0, 2, 1).reshape(L, -1).to(bf)), events
        expect = torch.cat([calls[0][5].reshape(-1).cpu(), table_cpu[x['tid']], table_cpu[x['up']],
                            torch.tensor([x['pos']]).to(bf), calls[1][5].reshape(-1).cpu()])
        assert torch.equal(calls[2][0][0].cpu(), expect), events
        assert expect.numel() == L * f['layer_dim'] + 2 * f['tid_emb_dim'] + 1 + calls[1][5].numel()
        hn_cpu = torch.nn.functional.layer_norm(x['hid'].float(), (D,)).to(bf)
        hn_diff = max(hn_diff, float((calls[1][0].cpu().float() - hn_cpu.float()).abs().max()))
        hn_changed += int((calls[1][0].cpu() != hn_cpu).sum())
        for xin, codes, scale, bias, group, y in calls:
            with wc.fp32_exact():
                ref = se.reference_linear(xin, codes, scale, bias, group)
            ratio = wc.qgemv_ratio(y, ref, xin, codes, scale, group)  # raises on any nonfinite value
            worst = max(worst, float(ratio.max()))
            exact += int((y == ref).sum())
            total += y.numel()
    n_scored = sum(e['expect']['valid'] for e in fx['events'])
    _report('p0', events=events, scored=scored, qgemv_outputs=total, qgemv_exact_share=exact / max(total, 1),
            qgemv_worst_diff_over_tolerance=worst, layer_norm_max_abs_diff_vs_cpu=hn_diff,
            layer_norm_bf16_values_changed_vs_cpu=hn_changed)
    assert scored == n_scored > 0 and events == len(fx['events']) and total > 0
    assert worst <= 1.0, worst


# --- P1: graph against eager -----------------------------------------------------------------------------

def _hooks(start_layer: int = START, timing: Path | None = None, timing_all: bool = False, r36: bool = False):
    from commitmoe_prefetch import bank_loader
    from commitmoe_prefetch import freetoken as ft
    release, _ = _release()
    rule = dict(evict=(_pinned(R36, 'EXP526_EVICT_BUNDLE_SHA256'), start_layer)) if r36 else \
        dict(shared=(BUNDLE, start_layer))
    cfg, state = bank_loader.load(PUBLISHED)
    hooks = ft.PrefetchHooks(cfg, state, demand_stripes=16, timing=timing, timing_all=timing_all,
                             release_sha256=release, **rule)
    before = torch.cuda.memory_allocated()
    hooks.reserve(SimpleNamespace(device=CUDA))
    reserved = torch.cuda.memory_allocated() - before
    cache, hosts = _cache()
    hooks.bind(None, cache)
    assert cache.learned_evict is hooks.learned
    return hooks, cache, hosts, reserved


def _twin(s, ema, age, routed):
    """An eager copy of scorer ``s``: the same device tensors, its own state."""
    twin = copy.copy(s)
    twin.ema, twin.age, twin.routed = ema.clone(), age.clone(), routed.clone()
    return twin


@pytest.mark.parametrize('split', [False, True])
def test_p1_graph_equals_eager(split):
    from freetoken.moe.learned_evict import LearnedEvict, learned_ensure
    hooks, cache, hosts, _ = _hooks()
    s, rule = hooks.shared, hooks.learned
    inputs = _Inputs()
    graph = _capture(hooks, cache, inputs, split=split)
    generator = torch.Generator().manual_seed(7)
    table = hooks.bank.buffers['table']
    for uid in (1, 2):
        hooks.begin_request(uid)
        assert int(rule.valid) == 0 and not s.ema.any() and s.age.eq(s.far).all() and not s.routed.any()
        shadow = _shadow(cache)
        shadow_rule = LearnedEvict(keep=rule.keep.clone(), valid=rule.valid.clone(),
                                   target_of_layer=rule.target_of_layer, start_layer=START,
                                   counters=torch.zeros_like(rule.counters))
        for step in range(STEPS):
            inputs.draw(generator)
            state = (s.ema.clone(), s.age.clone(), s.routed.clone())
            resident = (cache.slot_for_id >= 0).float()
            prior_before = hooks.raw['prior_before'].clone()
            hooks.before_decode(step)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(hooks.resident, resident), (uid, step)
            twin = _twin(s, *state)
            twin.advance(hooks.score_inputs['routes'], hooks.step)
            keep = twin.scores(twin.own(resident), hooks.score_inputs['h'], hooks.raw['tid'],
                               hooks.raw['upcoming_tid'], hooks.raw['pos'][:1], table)
            assert torch.equal(rule.keep, keep), (uid, step)
            assert torch.equal(s.ema, twin.ema) and torch.equal(s.age, twin.age), (uid, step)
            assert bool(torch.isfinite(keep).all()) and int(rule.valid) == int(step > 0), (uid, step)
            if step == 0:
                assert torch.equal(s.ema, state[0]) and torch.equal(s.age, state[1])
            g = hooks.bank.trunk({**hooks.raw, 'prior_before': prior_before})
            assert torch.equal(hooks.probs, hooks.bank.head(g)), (uid, step)
            shadow_rule.keep.copy_(rule.keep)
            shadow_rule.valid.copy_(rule.valid)
            for layer in range(L):
                learned_ensure(shadow, layer, inputs.ids[layer].clone(), shadow_rule)
            for name in ('slot_for_id', 'id_of_slot', 'usage', 'step'):
                assert torch.equal(getattr(shadow, name), getattr(cache, name)), (uid, step, name)
            assert _check_resident_rows(cache, hosts) > 0
            hooks.after_decode(step, torch.tensor([100 + step], device=CUDA), torch.tensor([200 + step], device=CUDA))
        assert torch.equal(rule.counters, shadow_rule.counters)
        events, reranked, rescued = rule.counters.tolist()
        assert events > 0 and reranked > 0, rule.counters
    _report('p1', split=split, requests=2, steps=STEPS, evict_counters_last_request=rule.counters.tolist(),
            nonfinite_tokens=int(hooks.nonfinite_tokens))
    assert int(hooks.nonfinite_tokens) == 0


# --- P2: GPU scores against the emulator, and the victim interface ----------------------------------------

def _rule(keep: torch.Tensor, valid: int):
    from freetoken.moe.learned_evict import LearnedEvict
    return LearnedEvict(keep=keep, valid=torch.full((1,), valid, dtype=torch.int32, device=CUDA),
                        target_of_layer=torch.arange(L, dtype=torch.int32, device=CUDA), start_layer=0,
                        counters=torch.zeros(3, dtype=torch.int64, device=CUDA))


def _follow(dst, src) -> None:
    for name in ('slot_for_id', 'id_of_slot', 'usage', 'step'):
        getattr(dst, name).copy_(getattr(src, name))


def test_p2_scores_and_victims():
    from freetoken.moe.learned_evict import learned_ensure
    fx = _fixtures()
    gpu, _, table, _ = _scorers()
    rows = []
    for e in fx['events']:
        if e['kind'] == 'reset':
            gpu.reset()
            continue
        x = e['inputs']
        gpu.advance(x['routes'].to(CUDA), torch.tensor([e['step']], dtype=torch.int32, device=CUDA))
        if e['expect']['valid']:
            z = gpu.scores(gpu.own(x['res'].float().to(CUDA)), *_args(x, fx['meta']['hidden_layers'], CUDA), table)
            rows.append((z.cpu(), e['expect']['z'], x['res'], x['routes']))
    finite = all(bool(torch.isfinite(z).all()) for z, _, _, _ in rows)
    overlap = torch.stack([bottom_overlap(z, ze, res) for z, ze, res, _ in rows])
    diff = torch.stack([z - ze for z, ze, _, _ in rows])
    ze_all = torch.stack([ze for _, ze, _, _ in rows])
    q = torch.quantile(diff.abs().flatten(), torch.tensor([0.5, 0.99, 0.999]))
    ties = sum(int(z[layer, res[layer]].numel() - z[layer, res[layer]].unique().numel())
               for z, _, res, _ in rows for layer in range(L))
    low, high = ze_all < -10, ze_all > 10
    # The victim kernel, fed each score source (A: GPU, B: emulator) and the LRU rule (C) from the same
    # maps and the same next routes. After each layer B and C continue from A's maps, so every layer
    # compares one choice. 'reranked' counts A's victims that LRU would not have chosen.
    cache, _ = _cache()
    base = _shadow(cache)  # the slot maps only; no expert bytes move in this test
    warm = _Inputs()
    generator = torch.Generator().manual_seed(11)
    lru = _rule(torch.zeros(L, E, device=CUDA), 0)  # valid 0: the plain LRU rule
    ev, agree, reranked, reranked_agree = (torch.zeros(L, dtype=torch.int64) for _ in range(4))
    for i, (z, ze, _, _) in enumerate(rows):
        warm.draw(generator)
        for layer in range(L):  # one LRU token, so the maps move between rows
            learned_ensure(base, layer, warm.ids[layer].clone(), lru)
        nxt = rows[(i + 1) % len(rows)][3]
        a, b, c = _shadow(base), _shadow(base), _shadow(base)
        ra, rb = _rule(z.to(CUDA), 1), _rule(ze.to(CUDA), 1)
        for layer in range(L):
            ids = nxt[layer].to(CUDA).int().view(1, -1)
            before = a.slot_for_id.clone()
            learned_ensure(a, layer, ids.clone(), ra)
            learned_ensure(b, layer, ids.clone(), rb)
            learned_ensure(c, layer, ids.clone(), lru)
            gone_a, gone_b, gone_c = ((before >= 0) & (m.slot_for_id < 0) for m in (a, b, c))
            ev[layer] += int(gone_a.sum())
            agree[layer] += int((gone_a & gone_b).sum())
            reranked[layer] += int((gone_a & ~gone_c).sum())
            reranked_agree[layer] += int((gone_a & ~gone_c & gone_b).sum())
            _follow(b, a)
            _follow(c, a)
    total = int(ev.sum())
    values = dict(rows=len(rows), finite=finite, overlap_mean=float(overlap.mean()), overlap_min=float(overlap.min()),
                  layer_min=float(overlap.mean(0).min()), layer_min_index=int(overlap.mean(0).argmin()),
                  bad_layer_rows=int((overlap < 1).sum()), layer_rows=overlap.numel(),
                  abs_diff_q50_q99_q999=q.tolist(), abs_diff_max=float(diff.abs().max()),
                  signed_mean=float(diff.mean()),
                  low_logits=int(low.sum()), low_abs_diff_max=float(diff[low].abs().max()) if low.any() else 0.0,
                  high_logits=int(high.sum()), high_abs_diff_max=float(diff[high].abs().max()) if high.any() else 0.0,
                  resident_ties=ties, evictions=total, victim_same_share=int(agree.sum()) / max(total, 1),
                  reranked=int(reranked.sum()),
                  reranked_same_share=int(reranked_agree.sum()) / max(int(reranked.sum()), 1),
                  evictions_by_layer=ev.tolist(), disagreements_by_layer=(ev - agree).tolist(),
                  reranked_by_layer=reranked.tolist())
    _report('p2', **values)
    assert finite and len(rows) == sum(e['expect']['valid'] for e in fx['events']) > 0, values
    assert total > 0 and int(reranked.sum()) > 0, values  # the interface was exercised, not only LRU
    assert values['overlap_mean'] >= OVERLAP_GATE, values
    assert values['victim_same_share'] >= VICTIM_GATE, values


# --- a2: the agreed cold-L2 microbenchmark (gate) and the in-schedule diagnostic --------------------------

def _prefix_graph(hooks, cache, snapshot: bool) -> torch.cuda.CUDAGraph:
    """One arm's probe prefix as a graph: main's observe(0) stamp, the snapshot (R584 only) and _score(),
    which writes the scores-ready stamp (column L) before the prefetch head."""
    from commitmoe_prefetch import kernels
    hooks.step.fill_(1)  # a scored token: valid is computed as in decode

    def body():
        kernels.stamp[(1,)](hooks.stamps, IDX=0)
        if snapshot:
            hooks.resident.copy_(cache.slot_for_id >= 0)
        hooks._score()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        body()  # eager warm-up, as before any capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    return graph


def _block(graph, hooks, flush: torch.Tensor) -> list[int]:
    out = []
    for i in range(WARMUP + TIMED):
        flush.fill_(i & 0xFF)  # evict L2 before every replay
        hooks.stamps.zero_()
        graph.replay()
        torch.cuda.synchronize()
        s = hooks.stamps.cpu()
        if i >= WARMUP:
            ready, obs = int(s[L]), int(s[0])
            assert obs > 0 and ready >= obs, (obs, ready)
            out.append(ready - obs)
    return out


def test_a2_cold_l2_prefix_gate(tmp_path):
    base, base_cache, _, base_mem = _hooks(START, timing=tmp_path / 'b.pt', r36=True)
    cand, cand_cache, _, cand_mem = _hooks(START, timing=tmp_path / 'c.pt')
    graphs = dict(base=_prefix_graph(base, base_cache, False), cand=_prefix_graph(cand, cand_cache, True))
    hooks = dict(base=base, cand=cand)
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=CUDA)
    times = dict(base=[], cand=[])
    for arm in ('base', 'cand', 'cand', 'base'):
        times[arm] += _block(graphs[arm], hooks[arm], flush)
    t = {arm: torch.tensor(v, dtype=torch.float64) for arm, v in times.items()}
    values = {f'{arm}_{k}_us': float(fn(t[arm])) / 1e3 for arm in t
              for k, fn in (('median', torch.median), ('p90', lambda v: torch.quantile(v, 0.9)), ('max', torch.max))}
    values.update(replays_per_arm=len(times['cand']), shared_evict_mib=cand.shared.nbytes / 2**20,
                  reserve_allocated_mib=dict(base=base_mem / 2**20, cand=cand_mem / 2**20))
    _report('a2_cold_l2', **values)
    assert len(times['cand']) == len(times['base']) == 2 * TIMED
    assert float(t['cand'].median()) <= A2_GATE_NS, values


def test_a2_in_schedule_diagnostic(tmp_path):
    """Stand-in workload (synthetic 600-slot cache, sleep kernels): reported, not gated."""
    out = tmp_path / 'stamps.pt'
    hooks, cache, _, _ = _hooks(START, timing=out, timing_all=True)
    inputs = _Inputs()
    graph = _capture(hooks, cache, inputs, split=True)
    generator = torch.Generator().manual_seed(13)
    for uid in (1, 2, 3):  # the rows of a request are written when the next request begins
        hooks.begin_request(uid)
        if uid == 3:
            break
        for step in range(DIAG_STEPS):
            inputs.draw(generator)
            hooks.before_decode(step)
            graph.replay()
            hooks.after_decode(step, torch.tensor([1], device=CUDA), torch.tensor([2], device=CUDA))
    torch.cuda.synchronize()
    rows = torch.load(out)
    iv = wc.stamp_intervals(rows['stamps'], list(rows['step']), list(range(DIAG_STEPS)) * 2, L)
    scored = torch.tensor(list(rows['step'])) > 0
    prefix, wait = iv['prefix'][scored].double(), iv['wait'][scored].double()
    _report('a2_in_schedule_diagnostic', tokens=int(scored.sum()), prefix_median_us=float(prefix.median()) / 1e3,
            prefix_p90_us=float(torch.quantile(prefix, 0.9)) / 1e3,
            wait_at_J_median_us=float(wait.median()) / 1e3, wait_at_J_p90_us=float(torch.quantile(wait, 0.9)) / 1e3)
    assert int(scored.sum()) == 2 * (DIAG_STEPS - 1)
