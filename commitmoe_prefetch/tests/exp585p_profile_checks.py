"""Checks of the EXP585P GPU profile (test_gpu_exp585p_profile.py) that hold on any device; their CPU tests
are in test_exp585p_profile_checks.py, so a reader that drops nodes, a stamp layout that differs from the
declared one or a stamp row that does not add up fails before the GPU window, not in it.

graph_nodes      every node of a captured CUDA graph through the driver API; refuses what it cannot read.
split_at_ready   prefix / suffix node partitions around the start, scores-ready and end stamp kernels.
layout           the declared stamp labels of each arm and graph variant.
Stamps           emits the profile stamps and refuses a body run whose labels differ from the layout.
instrument       wraps the served calls of one hook instance with profile stamps.
intervals        bracketed stage times of one block from start, scores-ready and the profile stamps.
stamped_shares   descriptive shares of the stamped prefix (Codex 4701e625, point 1); not an attribution.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable

import torch

# Node types a stream capture of the probe prefix can produce. Any other type (host, child graph, memory
# allocation, conditional) refuses the graph instead of being left out of the counts.
ALLOWED = ('KERNEL', 'MEMSET', 'MEMCPY', 'EMPTY', 'EVENT_RECORD', 'WAIT_EVENT')
TYPE_PREFIX = 'CU_GRAPH_NODE_TYPE_'
STAMP = re.compile(r'stamp(_|$)')  # the Triton stamp kernel; Triton names a kernel after its function
QGEMV = re.compile(r'qgemv(_|$)')
LINEARS = ('w_layer', 'h_proj', 'fc1', 'fc2', 'we', 'wb', 'wb64')  # call order in SharedEvict.heads


def _call(dr, fn, *args) -> tuple:
    res = fn(*args)
    if res[0] != dr.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f'{fn.__name__}: {res[0]!r}')
    return res[1:]


def graph_nodes(graph, dr) -> list[dict]:
    """Every node of the CUDA graph ``graph`` (a CUgraph handle) as dict(type, name, deps): the type without
    the CU_GRAPH_NODE_TYPE_ prefix, the full kernel name for a kernel node (None otherwise) and the indices
    of the node's dependencies. ``dr`` is cuda.bindings.driver (a stand-in in the CPU tests). Raises
    RuntimeError on any driver error, a type outside ALLOWED, a kernel without a name, or a node or a
    dependency count that changes between the two queries."""
    _, n = _call(dr, dr.cuGraphGetNodes, graph, 0)
    nodes, got = _call(dr, dr.cuGraphGetNodes, graph, n)
    if got != n or len(nodes) != n:
        raise RuntimeError(f'graph nodes: {n} counted, {got} returned')
    index = {int(node): i for i, node in enumerate(nodes)}
    if len(index) != n:
        raise RuntimeError('graph nodes: duplicate handles')
    out = []
    for node in nodes:
        typ, = _call(dr, dr.cuGraphNodeGetType, node)
        kind = typ.name.removeprefix(TYPE_PREFIX)
        if kind not in ALLOWED:
            raise RuntimeError(f'graph nodes: unsupported node type {typ.name}')
        name = None
        if kind == 'KERNEL':
            params, = _call(dr, dr.cuGraphKernelNodeGetParams, node)
            if int(params.func):
                raw, = _call(dr, dr.cuFuncGetName, params.func)
            elif int(params.kern):
                raw, = _call(dr, dr.cuKernelGetName, params.kern)
            else:
                raise RuntimeError('graph nodes: a kernel node has neither a function nor a kernel handle')
            name = raw.decode() if isinstance(raw, bytes) else str(raw)
            if not name:
                raise RuntimeError('graph nodes: a kernel node has an empty name')
        _, _, m = _call(dr, dr.cuGraphNodeGetDependencies, node, 0)
        deps, _, got = _call(dr, dr.cuGraphNodeGetDependencies, node, m)
        if got != m or len(deps) != m:
            raise RuntimeError(f'graph nodes: {m} dependencies counted, {got} returned')
        if any(int(d) not in index for d in deps):
            raise RuntimeError('graph nodes: a dependency outside the graph')
        out.append(dict(type=kind, name=name, deps=sorted(index[int(d)] for d in deps)))
    return out


def _ancestors(nodes: list[dict], i: int) -> set[int]:
    seen, todo = set(), list(nodes[i]['deps'])
    while todo:
        j = todo.pop()
        if j not in seen:
            seen.add(j)
            todo.extend(nodes[j]['deps'])
    return seen


def _summary(nodes: list[dict], ids: set[int]) -> dict:
    """Node indices, counts by type and by full kernel name of one partition."""
    return dict(nodes=len(ids), indices=sorted(ids), by_type=dict(Counter(nodes[i]['type'] for i in ids)),
                kernels=dict(Counter(nodes[i]['name'] for i in ids if nodes[i]['type'] == 'KERNEL')))


PARTS = ('before_start', 'prefix', 'suffix', 'unordered')


def split_at_ready(nodes: list[dict]) -> dict:
    """Partitions of a probe graph with exactly three stamp kernels in a total order: start, scores ready
    and end. prefix = ancestors of ready after start; suffix = ancestors of end after ready; before_start
    = ancestors of start; unordered = nodes that are not ancestors of end. Stamps belong to no partition;
    their indices are returned. Raises ValueError unless there are three totally ordered stamps, or when
    the partitions overlap or do not cover every node together with the stamps."""
    stamps = [i for i, n in enumerate(nodes) if n['type'] == 'KERNEL' and STAMP.match(n['name'])]
    if len(stamps) != 3:
        raise ValueError(f'probe graph: {len(stamps)} stamp kernels, expected 3 (start, ready, end)')
    anc = {i: _ancestors(nodes, i) for i in stamps}
    rank = {i: len(anc[i] & set(stamps)) for i in stamps}
    if sorted(rank.values()) != [0, 1, 2]:
        raise ValueError(f'probe graph: the stamps are not in a total order ({sorted(rank.values())})')
    start, ready, end = sorted(stamps, key=rank.get)
    s = set(stamps)
    parts = dict(before_start=anc[start] - s, prefix=anc[ready] - anc[start] - s,
                 suffix=anc[end] - anc[ready] - s, unordered=set(range(len(nodes))) - anc[end] - s)
    seen = s.copy()
    for ids in parts.values():
        if seen & ids:
            raise ValueError('probe graph: the partitions overlap')
        seen |= ids
    if seen != set(range(len(nodes))):
        raise ValueError('probe graph: the partitions do not cover every node')
    return dict(total_nodes=len(nodes), stamps=dict(start=start, ready=ready, end=end),
                **{k: _summary(nodes, v) for k, v in parts.items()})


def graph_evidence(nodes: list[dict], split: dict, versions: dict[str, str]) -> dict:
    """The JSON record of one M1 graph: every node (type, full name, dependencies), the split with its
    node indices and the software versions used to read it. split_at_ready(record['nodes']) gives the
    record's split again (tested on the CPU)."""
    return dict(versions=dict(versions), nodes=[dict(n) for n in nodes], split=split)


def layout(cand: bool, stages: bool) -> list[str]:
    """The declared profile stamp labels, in program order, of one graph. U graphs (stages False) have
    only the end stamp. S graphs: for R584 the snapshot, the raw-feature block (entry of _score_shared),
    advance, own, before and after each linear call in LINEARS order, the rest of heads and mix; for r36
    the raw-feature block (entry of the bank trunk), the trunk and EvictRows. The served scores-ready stamp
    is not a profile stamp; it comes between the last label before 'end' and 'end'."""
    if not stages:
        return ['end']
    if cand:
        return (['snapshot', 'raw_copies', 'advance', 'own'] + [x for n in LINEARS for x in (n + '_pre', n)]
                + ['heads_tail', 'mix', 'end'])
    return ['raw_copies', 'trunk', 'evict', 'end']


class Stamps:
    """Profile stamps at fixed indices: the n-th stamp of a body run goes to index n (``emit(n)`` launches
    the stamp kernel), so the eager warm-up and the capture write the same columns. end() raises
    ValueError when the labels of the run differ from the declared layout."""

    def __init__(self, emit: Callable[[int], None], labels: list[str]):
        self.emit, self.labels, self.seen = emit, list(labels), []

    def begin(self) -> None:
        self.seen = []

    def __call__(self, label: str) -> None:
        self.emit(len(self.seen))
        self.seen.append(label)

    def end(self) -> None:
        if self.seen != self.labels:
            raise ValueError(f'profile stamps {self.seen}, declared {self.labels}')


def _after(st: Stamps, label: str, fn):
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        st(label)
        return out
    return wrapped


def instrument(hooks, st: Stamps, cand: bool) -> None:
    """Wrap the served calls of this hook instance with profile stamps; instance attributes shadow the
    methods, and SharedEvict.heads reads self.linear at call time. The seven linear calls are told apart
    by the data pointer of their INT8 codes; an unknown pointer raises KeyError."""
    if cand:
        s = hooks.shared
        names = {s.t[n + '_q'].data_ptr(): n for n in LINEARS}
        if len(names) != len(LINEARS):
            raise ValueError('instrument: two linear stages share their codes tensor')
        linear = s.linear

        def stamped_linear(x, codes, scale, bias, group):
            name = names[codes.data_ptr()]
            st(name + '_pre')
            y = linear(x, codes, scale, bias, group)
            st(name)
            return y
        score_shared = hooks._score_shared

        def stamped_score_shared():
            st('raw_copies')
            score_shared()
        hooks._score_shared = stamped_score_shared
        s.advance = _after(st, 'advance', s.advance)
        s.own = _after(st, 'own', s.own)
        s.heads = _after(st, 'heads_tail', s.heads)
        s.mix = _after(st, 'mix', s.mix)
        s.linear = stamped_linear
    else:
        trunk = hooks.bank.trunk

        def stamped_trunk(raw):
            st('raw_copies')
            g = trunk(raw)
            st('trunk')
            return g
        hooks.bank.trunk = stamped_trunk
        hooks.evict = _after(st, 'evict', hooks.evict)


def intervals(rows: torch.Tensor, labels: list[str]) -> dict[str, torch.Tensor]:
    """Bracketed intervals of one block. rows (n, 2 + len(labels)) int64: the served start and scores-ready
    stamps, then the profile stamps of ``labels`` (a layout(), ending with 'end'). Interval 'ready' ends at
    scores-ready (R584 S: keep copy and validity; U: the whole prefix), 'suffix' is scores-ready to end.
    Raises ValueError for a layout without a final 'end' and everything stage_intervals refuses, so a
    scores-ready stamp outside its place gives a negative interval and is refused."""
    if not labels or labels[-1] != 'end':
        raise ValueError(f'profile labels {labels} do not end with end')
    if rows.dim() != 2 or rows.shape[1] != 2 + len(labels):
        raise ValueError(f'stamp rows {tuple(rows.shape)}, expected (n, {2 + len(labels)})')
    cut = len(labels) - 1
    order = torch.cat([rows[:, :1], rows[:, 2:2 + cut], rows[:, 1:2], rows[:, 2 + cut:]], 1)
    return stage_intervals(order, labels[:cut] + ['ready', 'suffix'])


def stage_intervals(rows: torch.Tensor, labels: list[str]) -> dict[str, torch.Tensor]:
    """Per replay, the time in ns of each bracketed stage: rows (n, len(labels) + 1) int64 globaltimer
    stamps in program order, column 0 the start; labels[k] names the interval that ends at column k + 1.
    Raises ValueError on another dtype or shape, a missing stamp (<= 0), a stamp earlier than the one
    before it, or stages that do not add up to last minus first."""
    if rows.dtype != torch.int64 or rows.dim() != 2 or rows.shape[1] != len(labels) + 1 or rows.shape[0] == 0:
        raise ValueError(f'stamp rows {rows.dtype} {tuple(rows.shape)}, expected int64 (n, {len(labels) + 1})')
    if len(set(labels)) != len(labels):
        raise ValueError('stamp rows: repeated stage label')
    if not bool((rows > 0).all()):
        raise ValueError('stamp rows: a stamp is missing (<= 0)')
    d = rows[:, 1:] - rows[:, :-1]
    if not bool((d >= 0).all()):
        bad = int((d < 0).any(0).nonzero()[0])
        raise ValueError(f'stamp rows: {labels[bad]} ends before it starts')
    if not torch.equal(d.sum(1), rows[:, -1] - rows[:, 0]):
        raise ValueError('stamp rows: the stages do not add up to the stamped total')
    return {name: d[:, k] for k, name in enumerate(labels)}


def stamped_shares(stamped: list[tuple[float, float]], unstamped: list[float]) -> dict:
    """Descriptive shares of the STAMPED prefix, per L2 condition. ``stamped`` holds per S block (median
    bracketed qgemv time Q, median S prefix P), ``unstamped`` the median U prefix of each U block, in ns.
    Returns Q / P and (P - Q) / P per S block, every signed S - U difference, and a stamped-majority label:
    'matrix' if Q / P >= 0.5 in every S block, 'surrounding' if Q / P < 0.5 in every S block, else
    'mixed'. The stamps can slow one stage and speed another with cancellation in the total, so this is
    not an attribution of the unstamped prefix (that stays undetermined) and does not select the next
    change by itself. Raises ValueError on empty input, a non-finite value, P <= 0, U <= 0 or Q outside
    [0, P]."""
    values = [v for q, p in stamped for v in (q, p)] + list(unstamped)
    if (not stamped or not unstamped or not all(math.isfinite(v) for v in values)
            or any(u <= 0 for u in unstamped) or any(p <= 0 or q < 0 or q > p for q, p in stamped)):
        raise ValueError(f'stamped_shares: invalid blocks {stamped}, {unstamped}')
    matrix = [q / p for q, p in stamped]
    label = 'matrix' if min(matrix) >= 0.5 else 'surrounding' if max(matrix) < 0.5 else 'mixed'
    return dict(stamped_majority=label, matrix_share=matrix, other_share=[(p - q) / p for q, p in stamped],
                s_minus_u_ns=[p - u for _, p in stamped for u in unstamped], unstamped_attribution='undetermined')
