"""CPU tests of exp585p_profile_checks: the graph reader refuses errors, unknown node types, nameless kernels
and changing counts; the split puts every node in exactly one partition and its JSON record gives the same
split again, with full kernel names; on real PrefetchHooks (CPU, stub bank and stamp kernel) the profile
stamps of both arms follow the declared layout, the served scores-ready stamp falls between mix or EvictRows
and end, and the intervals add up; other layouts and misplaced stamps are refused; the stamped shares are
descriptive only and refuse non-finite input.

The hooks tests need the published config (COMMITMOE_PREDICTOR_DIR) and skip without it.
Run:  CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider tests/test_exp585p_profile_checks.py
"""
from __future__ import annotations

import enum
import json
from types import SimpleNamespace

import pytest
import torch

import exp585p_profile_checks as pc
from commitmoe_prefetch import evict as evict_module
from commitmoe_prefetch import freetoken as ft
from commitmoe_prefetch import shared_evict as se
from test_shared_evict_cpu import SMALL, cfg, feed, make_bundle, make_hooks, stub_device  # noqa: F401 (cfg fixture)


class _Res(enum.IntEnum):
    CUDA_SUCCESS = 0
    CUDA_ERROR_INVALID_VALUE = 1


class _Type(enum.IntEnum):
    CU_GRAPH_NODE_TYPE_KERNEL = 0
    CU_GRAPH_NODE_TYPE_MEMCPY = 1
    CU_GRAPH_NODE_TYPE_MEMSET = 2
    CU_GRAPH_NODE_TYPE_HOST = 3


class _H(int):
    """A handle: int() gives its address, as for cuda.bindings handles."""


class _Driver:
    """The five driver calls of graph_nodes over a list of (type, name, deps); name None = no handle."""

    CUresult = _Res

    def __init__(self, spec, fail=None, shrink=False):
        self.spec, self.fail, self.shrink = spec, fail, shrink
        self.handles = [_H(100 + i) for i in range(len(spec))]

    def _node(self, h):
        return self.spec[int(h) - 100]

    def cuGraphGetNodes(self, graph, n):
        if self.fail == 'nodes':
            return (_Res.CUDA_ERROR_INVALID_VALUE, [], 0)
        if n == 0:
            return (_Res.CUDA_SUCCESS, [], len(self.handles))
        got = self.handles[:n - 1] if self.shrink else self.handles[:n]
        return (_Res.CUDA_SUCCESS, got, len(got))

    def cuGraphNodeGetType(self, h):
        return (_Res.CUDA_SUCCESS, self._node(h)[0])

    def cuGraphKernelNodeGetParams(self, h):
        name = self._node(h)[1]
        return (_Res.CUDA_SUCCESS, SimpleNamespace(func=_H(int(h)) if name else _H(0), kern=_H(0)))

    def cuFuncGetName(self, func):
        return (_Res.CUDA_SUCCESS, self._node(func)[1].encode())

    def cuKernelGetName(self, kern):
        raise AssertionError('not used: every stand-in kernel has a function handle')

    def cuGraphNodeGetDependencies(self, h, m):
        if self.fail == 'deps':
            return (_Res.CUDA_ERROR_INVALID_VALUE, [], [], 0)
        deps = [self.handles[j] for j in self._node(h)[2]]
        return (_Res.CUDA_SUCCESS, deps[:m] if m else [], [None] * (len(deps[:m]) if m else 0), len(deps))


K, C, S, HOST = (_Type.CU_GRAPH_NODE_TYPE_KERNEL, _Type.CU_GRAPH_NODE_TYPE_MEMCPY,
                 _Type.CU_GRAPH_NODE_TYPE_MEMSET, _Type.CU_GRAPH_NODE_TYPE_HOST)
# pre -> start -> memset -> qgemv -> ready -> copy -> end, and a branch from qgemv that end does not wait for.
CHAIN = [(K, 'pre', []), (K, 'stamp', [0]), (S, None, [1]), (K, 'qgemv', [2]), (K, 'stamp', [3]),
         (C, None, [4]), (K, 'stamp', [5]), (K, 'side', [3])]


def test_graph_nodes_reads_types_names_and_dependencies():
    nodes = pc.graph_nodes('g', _Driver(CHAIN))
    assert [n['type'] for n in nodes] == ['KERNEL', 'KERNEL', 'MEMSET', 'KERNEL', 'KERNEL', 'MEMCPY', 'KERNEL',
                                          'KERNEL']
    assert [n['name'] for n in nodes] == ['pre', 'stamp', None, 'qgemv', 'stamp', None, 'stamp', 'side']
    assert nodes[4]['deps'] == [3] and nodes[0]['deps'] == []


@pytest.mark.parametrize('driver,message', [
    (_Driver(CHAIN, fail='nodes'), 'cuGraphGetNodes'),
    (_Driver(CHAIN, fail='deps'), 'cuGraphNodeGetDependencies'),
    (_Driver(CHAIN + [(HOST, None, [])]), 'unsupported node type'),
    (_Driver([(K, None, [])]), 'neither a function nor a kernel'),
    (_Driver(CHAIN, shrink=True), 'counted'),
])
def test_graph_nodes_refuses(driver, message):
    with pytest.raises(RuntimeError, match=message):
        pc.graph_nodes('g', driver)


def test_split_counts_every_node_once():
    out = pc.split_at_ready(pc.graph_nodes('g', _Driver(CHAIN)))
    assert out['prefix']['nodes'] == 2 and out['prefix']['by_type'] == {'MEMSET': 1, 'KERNEL': 1}
    assert out['prefix']['kernels'] == {'qgemv': 1}
    assert out['suffix']['nodes'] == 1 and out['suffix']['by_type'] == {'MEMCPY': 1}
    assert out['before_start']['nodes'] == 1 and out['unordered']['nodes'] == 1
    assert sum(out[k]['nodes'] for k in pc.PARTS) + 3 == out['total_nodes']
    assert out['stamps'] == dict(start=1, ready=4, end=6)
    ids = [i for k in pc.PARTS for i in out[k]['indices']] + list(out['stamps'].values())
    assert sorted(ids) == list(range(out['total_nodes']))


def test_evidence_round_trip_keeps_full_kernel_names():
    long = 'k' * 60
    spec = CHAIN[:3] + [(K, long + '_a', [2]), (K, long + '_b', [3])] + [(K, 'stamp', [4]), (C, None, [5]),
                                                                          (K, 'stamp', [6])]
    nodes = pc.graph_nodes('g', _Driver(spec))
    split = pc.split_at_ready(nodes)
    assert split['prefix']['kernels'] == {long + '_a': 1, long + '_b': 1}
    record = json.loads(json.dumps(pc.graph_evidence(nodes, split, dict(torch='x'))))
    assert record['nodes'] == nodes and record['versions'] == dict(torch='x')
    assert pc.split_at_ready(record['nodes']) == record['split'] == json.loads(json.dumps(split))


def test_split_refuses_other_stamp_layouts():
    two = [n for n in CHAIN if n != CHAIN[6]][:6]
    with pytest.raises(ValueError, match='2 stamp kernels'):
        pc.split_at_ready(pc.graph_nodes('g', _Driver(two)))
    parallel = [(K, 'stamp', []), (K, 'stamp', []), (K, 'stamp', [0])]  # stamps 0 and 1 are unordered
    with pytest.raises(ValueError, match='total order'):
        pc.split_at_ready(pc.graph_nodes('g', _Driver(parallel)))


def _rows():
    return torch.tensor([[100, 150, 400, 420], [200, 260, 500, 530]], dtype=torch.int64)


def test_stage_intervals_telescope():
    out = pc.stage_intervals(_rows(), ['a', 'b', 'c'])
    assert out['a'].tolist() == [50, 60] and out['b'].tolist() == [250, 240] and out['c'].tolist() == [20, 30]


@pytest.mark.parametrize('change,labels,message', [
    (lambda r: r.__setitem__((0, 2), 0), ['a', 'b', 'c'], 'missing'),
    (lambda r: r.__setitem__((1, 2), 250), ['a', 'b', 'c'], 'b ends before it starts'),
    (lambda r: None, ['a', 'b'], 'expected int64'),
    (lambda r: None, ['a', 'a', 'c'], 'repeated'),
])
def test_stage_intervals_refuse(change, labels, message):
    r = _rows()
    change(r)
    with pytest.raises(ValueError, match=message):
        pc.stage_intervals(r, labels)


def test_stage_intervals_refuse_other_dtypes():
    with pytest.raises(ValueError, match='expected int64'):
        pc.stage_intervals(_rows().double(), ['a', 'b', 'c'])


def test_stamped_shares_are_descriptive_and_flag_mixed_blocks():
    out = pc.stamped_shares([(300.0, 400.0), (310.0, 405.0)], [390.0, 395.0])
    assert out['stamped_majority'] == 'matrix' and out['matrix_share'] == [0.75, 310 / 405]
    assert out['other_share'] == [0.25, 95 / 405] and out['s_minus_u_ns'] == [10.0, 5.0, 15.0, 10.0]
    assert pc.stamped_shares([(100.0, 400.0)], [395.0])['stamped_majority'] == 'surrounding'
    assert pc.stamped_shares([(300.0, 400.0), (100.0, 400.0)], [400.0])['stamped_majority'] == 'mixed'


def test_stamped_shares_claim_no_unstamped_attribution():
    """Codex 4701e625: S with Q 300, other 100 and U 400 is consistent with an unstamped Q 100, other 300
    (the stamps can speed one stage and slow another). The result is a stamped majority only."""
    out = pc.stamped_shares([(300.0, 400.0)], [400.0])
    assert out['stamped_majority'] == 'matrix' and out['unstamped_attribution'] == 'undetermined'
    assert out['s_minus_u_ns'] == [0.0]
    assert set(out) == {'stamped_majority', 'matrix_share', 'other_share', 's_minus_u_ns', 'unstamped_attribution'}


@pytest.mark.parametrize('stamped,unstamped', [([], [1.0]), ([(1.0, 2.0)], []), ([(3.0, 2.0)], [2.0]),
                                               ([(1.0, 2.0)], [0.0]), ([(-1.0, 2.0)], [2.0]),
                                               ([(float('nan'), 2.0)], [2.0]), ([(1.0, float('inf'))], [2.0]),
                                               ([(1.0, 2.0)], [float('nan')]), ([(1.0, 2.0)], [float('-inf')])])
def test_stamped_shares_refuse_invalid_blocks(stamped, unstamped):
    with pytest.raises(ValueError):
        pc.stamped_shares(stamped, unstamped)


def test_layout_has_every_linear_once_in_order():
    cand = pc.layout(True, True)
    lin = [x for x in cand if x in pc.LINEARS]
    assert lin == list(pc.LINEARS) and len(set(cand)) == len(cand)
    assert cand.index('own') < cand.index('w_layer_pre') and cand[-3:] == ['heads_tail', 'mix', 'end']
    assert pc.layout(False, True) == ['raw_copies', 'trunk', 'evict', 'end']
    assert pc.layout(True, False) == pc.layout(False, False) == ['end']


def test_stamps_refuse_missing_duplicate_and_reordered_labels():
    emitted = []
    for labels in (['a', 'b'], ['a', 'end', 'end'], ['end', 'a'], ['a', 'end', 'x']):
        st = pc.Stamps(emitted.append, ['a', 'end'])
        st.begin()
        for x in labels:
            st(x)
        with pytest.raises(ValueError, match='declared'):
            st.end()
    st = pc.Stamps(emitted.append, ['a', 'end'])
    st.begin()
    st('a')
    st('end')
    st.end()
    assert emitted[-2:] == [0, 1]


class _Timeline:
    """One clock for the served stamp kernel (stand-in for kernels.stamp) and the profile stamps."""

    def __init__(self, L: int):
        self.L, self.events = L, []

    def __getitem__(self, grid):
        return lambda out, IDX: self.events.append(('served', IDX))

    def profile(self, i: int) -> None:
        self.events.append(('profile', i))

    def labels(self, st: pc.Stamps) -> list[str]:
        names = {0: 'start', self.L: 'ready'}
        return [names[i] if kind == 'served' else st.labels[i] for kind, i in self.events]

    def rows(self) -> torch.Tensor:
        """One stamp row, the time of an event = its position + 1: start, ready, then the profile stamps."""
        t = {e: k + 1 for k, e in enumerate(self.events)}
        prof = sorted(i for kind, i in self.events if kind == 'profile')
        return torch.tensor([[t[('served', 0)], t[('served', self.L)]] + [t[('profile', i)] for i in prof]],
                            dtype=torch.int64)


def _body(hooks, tl_: _Timeline, st: pc.Stamps, cand: bool, stages: bool) -> None:
    """The body of test_gpu_exp585p_profile._graph, with the stand-in stamp kernel."""
    st.begin()
    tl_[(1,)](hooks.stamps, IDX=0)
    if cand:
        hooks.resident.copy_(hooks.cache.slot_for_id >= 0)
        if stages:
            st('snapshot')
    hooks._score()
    st('end')
    st.end()


def _arm(cfg, monkeypatch, tmp_path, cand: bool):  # noqa: F811 (cfg is the fixture value)
    stub_device(monkeypatch, ft)
    monkeypatch.setattr(evict_module, 'qlinear', se.reference_linear)
    L = cfg.base_model.num_layers
    if cand:
        path = make_bundle(tmp_path / 's.pt', SMALL, group=cfg.serving_reference.int8_group)
        hooks = make_hooks(ft, cfg, shared=(path, 1), timing=tmp_path / 't.pt')
        hooks.cache = SimpleNamespace(slot_for_id=torch.full((L, 256), -1, dtype=torch.int32))
    else:
        torch.manual_seed(0)
        torch.save(dict(version=2, layers=[0, 1, 8, 39], source_layers=[6, 6, 8, 39], horizon_tokens=8,
                        release_sha256='a' * 64, weight=torch.randn(4, 256, cfg.architecture.up_dim).bfloat16(),
                        bias=torch.randn(4, 256).bfloat16()), tmp_path / 'r.pt')
        hooks = make_hooks(ft, cfg, evict=(tmp_path / 'r.pt', 1), timing=tmp_path / 't.pt')
        H, up = len(cfg.target_layers), cfg.architecture.up_dim
        hooks.bank.trunk = lambda raw, b=hooks.bank: (b(raw)[:, :1] + torch.linspace(-1, 1, H * up).view(H, up)
                                                      ).bfloat16()
        hooks.bank.head = lambda g, b=hooks.bank: b.out
    tl_ = _Timeline(L)
    monkeypatch.setattr(ft.kernels, 'stamp', tl_)
    feed(hooks, 3, 0)
    return hooks, tl_


@pytest.mark.parametrize('cand', [True, False])
def test_profile_stamps_follow_the_layout_on_the_served_hooks(cfg, monkeypatch, tmp_path, cand):  # noqa: F811
    hooks, tl_ = _arm(cfg, monkeypatch, tmp_path, cand)
    u = pc.Stamps(tl_.profile, pc.layout(cand, False))
    _body(hooks, tl_, u, cand, stages=False)
    assert tl_.labels(u) == ['start', 'ready', 'end']
    t = pc.intervals(tl_.rows(), u.labels)
    assert list(t) == ['ready', 'suffix'] and int(t['ready'] + t['suffix']) == 2

    tl_.events.clear()
    s = pc.Stamps(tl_.profile, pc.layout(cand, True))
    pc.instrument(hooks, s, cand)
    _body(hooks, tl_, s, cand, stages=True)
    want = s.labels[:-1] + ['ready', 'end']
    assert tl_.labels(s) == ['start'] + want  # scores-ready between mix (or evict) and end
    rows = tl_.rows()
    t = pc.intervals(rows, s.labels)
    assert list(t) == want[:-2] + ['ready', 'suffix'] and all(int(v) == 1 for v in t.values())
    assert int(sum(t.values())) == int(rows[0, -1] - rows[0, 0])


def test_instrument_refuses_an_unknown_linear(cfg, monkeypatch, tmp_path):  # noqa: F811
    hooks, tl_ = _arm(cfg, monkeypatch, tmp_path, True)
    pc.instrument(hooks, pc.Stamps(tl_.profile, pc.layout(True, True)), True)
    with pytest.raises(KeyError):
        hooks.shared.linear(torch.zeros(1, 4), torch.zeros(1, 2, 4, dtype=torch.int8), None, None, 4)


def test_intervals_refuse_a_misplaced_ready_stamp_and_other_layouts():
    labels = ['a', 'mix', 'end']
    good = torch.tensor([[1, 4, 2, 3, 5]], dtype=torch.int64)   # start, ready, a, mix, end
    assert pc.intervals(good, labels)['ready'].tolist() == [1]
    with pytest.raises(ValueError, match='mix ends before it starts|ready ends before it starts'):
        pc.intervals(torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64), labels)  # ready before a and mix
    with pytest.raises(ValueError, match='suffix ends before it starts'):
        pc.intervals(torch.tensor([[1, 6, 2, 3, 5]], dtype=torch.int64), labels)  # ready after end
    with pytest.raises(ValueError, match='do not end with end'):
        pc.intervals(good, ['a', 'end', 'mix'])
    with pytest.raises(ValueError, match='expected'):
        pc.intervals(good[:, :4], labels)
