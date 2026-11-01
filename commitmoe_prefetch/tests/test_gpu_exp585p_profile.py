"""EXP585P: GPU profile of the R584 probe prefix after the failed EXP585 a2 gate (Codex 7cae34bd, 28f6621d,
4701e625). A diagnostic: no timing gate; the tests fail only when a measurement cannot be read, its stamps
differ from the declared layout or its intervals do not add up.

Setup of the a2 gate (test_gpu_shared_evict): _hooks(START) on the synthetic 600-slot cache, step set to 1,
repeated graph replays that advance the eviction state; base = r36 (EvictRows on the bank trunk),
candidate = R584. It is not a replay of the fixture rows. Served code is not changed: S graphs wrap
methods of the hook instance with profile stamps (exp585p_profile_checks.instrument) after the U graph of
the same instance is captured.

test_m345_stage_intervals
    U: the a2 prefix graph (start stamp, R584 snapshot, _score with its scores-ready stamp) plus one end
       stamp after _score, that is after the prior update. S: U plus the profile stamps of
       exp585p_profile_checks.layout. Blocks of 20 warm-ups and 200 timed replays, cold L2 (256 MiB written
       before every replay) and then warm L2, in the order base U, cand U, cand S, base S, base S, cand S,
       cand U, base U. Reports per block the prefix (scores-ready - start), the suffix (end - scores-ready)
       and the total probe work through the prior update (end - start), the bracketed stages of S, and per
       condition the signed S - U prefix differences and exp585p_profile_checks.stamped_shares.
test_m2_reference_chains
    A graph of 100 tick kernels (the stamp body with a runtime index; 99 measured intervals between them)
    and a graph of 100 add_ operations on a (40, 256) FP32 tensor between two ticks; elapsed and per
    measured interval, cold and warm. Reference chain timings, not a device-wide cost per node.
test_m1_graph_structure
    The U graphs of both arms captured with keep_graph=True and read node by node through the CUDA driver
    API. The full record (every node with its full kernel name and dependencies, the stamp indices, each
    partition's node indices, the software versions) goes to m1_graph_<arm>.json beside the report.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

import exp585p_profile_checks as pc
from test_gpu_shared_evict import FLUSH_BYTES, REPORT, START, TIMED, WARMUP, _hooks, _report
from test_gpu_schedule import L

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
CUDA = torch.device('cuda')
CHAIN = 100        # operations per reference chain
ORDER = (('base', 'U'), ('cand', 'U'), ('cand', 'S'), ('base', 'S'),
         ('base', 'S'), ('cand', 'S'), ('cand', 'U'), ('base', 'U'))


def _stamps(cand: bool, stages: bool) -> pc.Stamps:
    """Profile stamps into their own buffer; IDX is a compile-time constant of the served stamp kernel."""
    from commitmoe_prefetch import kernels
    out = torch.zeros(64, dtype=torch.int64, device=CUDA)
    st = pc.Stamps(lambda i: kernels.stamp[(1,)](out, IDX=i), pc.layout(cand, stages))
    st.out = out
    return st


def _graph(hooks, cache, cand: bool, st: pc.Stamps, stages: bool, keep_graph: bool = False) -> torch.cuda.CUDAGraph:
    """The a2 prefix body (test_gpu_shared_evict._prefix_graph) plus a final end stamp, captured after one
    eager run on a side stream. stages: also the stamp after the R584 snapshot (S graphs; the stamps of the
    wrapped served calls come from pc.instrument). st.end() refuses a run whose labels differ from the
    declared layout, in the eager run and in the capture."""
    from commitmoe_prefetch import kernels
    hooks.step.fill_(1)

    def body():
        st.begin()
        kernels.stamp[(1,)](hooks.stamps, IDX=0)
        if cand:
            hooks.resident.copy_(cache.slot_for_id >= 0)
            if stages:
                st('snapshot')
        hooks._score()
        st('end')
        st.end()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=keep_graph)
    with torch.cuda.graph(graph):
        body()
    return graph


def _replays(graph, hooks, st: pc.Stamps, flush: torch.Tensor | None) -> torch.Tensor:
    """(TIMED, 2 + stamps) int64 per timed replay: start, scores-ready, then the profile stamps of ``st``."""
    rows = []
    for i in range(WARMUP + TIMED):
        if flush is not None:
            flush.fill_(i & 0xFF)  # evict L2 before every replay, as in a2
        hooks.stamps.zero_()
        st.out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        if i >= WARMUP:
            rows.append(torch.cat([hooks.stamps[[0, L]], st.out[:len(st.labels)]]).cpu())
    return torch.stack(rows)


def _q(v: torch.Tensor) -> dict:
    v = v.double() / 1e3
    return dict(median_us=float(v.median()), p90_us=float(torch.quantile(v, 0.9)), max_us=float(v.max()))


def test_m345_stage_intervals(tmp_path):
    arms = {}
    for arm in ('base', 'cand'):
        cand = arm == 'cand'
        # The served scores-ready stamp needs a timing destination; rows are written only at request
        # boundaries, which these graphs never reach.
        hooks, cache, _, _ = _hooks(START, timing=tmp_path / f'{arm}.pt', r36=not cand)
        u, s = _stamps(cand, stages=False), _stamps(cand, stages=True)
        graph_u = _graph(hooks, cache, cand, u, stages=False)  # captured before instrument
        pc.instrument(hooks, s, cand)
        graph_s = _graph(hooks, cache, cand, s, stages=True)
        arms[arm] = dict(hooks=hooks, U=(graph_u, u), S=(graph_s, s))
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=CUDA)
    for cond in ('cold', 'warm'):
        blocks = []
        for k, (arm, var) in enumerate(ORDER):
            graph, st = arms[arm][var]
            t = pc.intervals(_replays(graph, arms[arm]['hooks'], st, flush if cond == 'cold' else None), st.labels)
            prefix = sum(v for name, v in t.items() if name != 'suffix')  # start to scores-ready
            row = dict(block=k, arm=arm, variant=var, prefix=_q(prefix), suffix=_q(t['suffix']),
                       total_through_prior_update=_q(prefix + t['suffix']), replays=int(prefix.numel()))
            if var == 'S':
                row['stages'] = {name: _q(v) for name, v in t.items()}
                if arm == 'cand':
                    row['qgemv_bracketed'] = _q(sum(t[n] for n in pc.LINEARS))
            blocks.append(row)
            _report('m345_block', condition=cond, **row)
        out = dict(condition=cond)
        for arm in ('base', 'cand'):
            pre = {var: [b['prefix']['median_us'] for b in blocks if b['arm'] == arm and b['variant'] == var]
                   for var in ('U', 'S')}
            out[arm + '_s_minus_u_us'] = [ps - pu for ps in pre['S'] for pu in pre['U']]
        stamped = [(b['qgemv_bracketed']['median_us'] * 1e3, b['prefix']['median_us'] * 1e3)
                   for b in blocks if b['arm'] == 'cand' and b['variant'] == 'S']
        unstamped = [b['prefix']['median_us'] * 1e3 for b in blocks if b['arm'] == 'cand' and b['variant'] == 'U']
        out['stamped_shares'] = pc.stamped_shares(stamped, unstamped)
        _report('m345_summary', **out)
        assert len(blocks) == len(ORDER) and all(b['replays'] == TIMED for b in blocks)


@triton.jit
def _tick(OUT, i):
    """The body of kernels.stamp with a runtime index, so one compiled kernel serves the whole chain."""
    z = tl.zeros([1], tl.int32)
    t = tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;', '=l,r', [z], dtype=tl.int64, is_pure=False, pack=1)
    tl.store(OUT + i + tl.arange(0, 1), t)


def test_m2_reference_chains():
    out = torch.zeros(CHAIN + 1, dtype=torch.int64, device=CUDA)
    x = torch.zeros(40, 256, dtype=torch.float32, device=CUDA)

    def ticks():
        for i in range(CHAIN):
            _tick[(1,)](out, i)

    def adds():
        _tick[(1,)](out, 0)
        for _ in range(CHAIN):
            x.add_(1.0)
        _tick[(1,)](out, CHAIN)
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=CUDA)
    # (name, body, first stamp, last stamp, operations in the chain, measured intervals between the stamps)
    for name, body, first, last, ops, measured in (('ticks', ticks, 0, CHAIN - 1, CHAIN, CHAIN - 1),
                                                    ('adds', adds, 0, CHAIN, CHAIN, CHAIN)):
        body()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        for cond in ('cold', 'warm'):
            elapsed = []
            for i in range(WARMUP + TIMED):
                if cond == 'cold':
                    flush.fill_(i & 0xFF)
                out.zero_()
                graph.replay()
                torch.cuda.synchronize()
                if i >= WARMUP:
                    row = out.cpu()
                    assert bool((row[[first, last]] > 0).all()) and int(row[last]) >= int(row[first])
                    elapsed.append(int(row[last] - row[first]))
            e = torch.tensor(elapsed, dtype=torch.int64)
            _report('m2_chain', chain=name, condition=cond, operations=ops, measured_intervals=measured,
                    elapsed=_q(e), per_measured_interval_median_us=float(e.double().median()) / 1e3 / measured)


def _versions(dr) -> dict[str, str]:
    import cuda.bindings
    err, driver = dr.cuDriverGetVersion()
    assert err == dr.CUresult.CUDA_SUCCESS, err
    return dict(torch=torch.__version__, torch_cuda=str(torch.version.cuda), triton=triton.__version__,
                cuda_bindings=cuda.bindings.__version__, driver=str(driver),
                device=torch.cuda.get_device_name())


def test_m1_graph_structure(tmp_path):
    import cuda.bindings.driver as dr
    folder = Path(REPORT).parent if REPORT else tmp_path
    for arm in ('base', 'cand'):
        cand = arm == 'cand'
        hooks, cache, _, _ = _hooks(START, timing=tmp_path / f'{arm}.pt', r36=not cand)
        st = _stamps(cand, stages=False)
        graph = _graph(hooks, cache, cand, st, stages=False, keep_graph=True)
        nodes = pc.graph_nodes(dr.CUgraph(graph.raw_cuda_graph()), dr)
        split = pc.split_at_ready(nodes)
        with open(folder / f'm1_graph_{arm}.json', 'x') as fh:  # never overwrites an earlier record
            json.dump(pc.graph_evidence(nodes, split, _versions(dr)), fh)
        _report('m1_graph', arm=arm, total_nodes=split['total_nodes'], stamps=split['stamps'],
                **{k: dict(nodes=split[k]['nodes'], by_type=split[k]['by_type'], kernels=split[k]['kernels'])
                   for k in pc.PARTS})
        assert split['unordered']['nodes'] == 0 and split['before_start']['nodes'] == 0, split
        if cand:
            qgemv = sum(c for name, c in split['prefix']['kernels'].items() if pc.QGEMV.match(name))
            assert qgemv == len(pc.LINEARS), split['prefix']['kernels']
