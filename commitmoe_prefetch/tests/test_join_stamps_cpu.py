"""CPU tests of the EXP586 final-join stamps (freetoken.py, timing mode only).

The streams, events and Triton kernels are replaced by fakes that log every launch and every
stream dependency in program order. The tests compare the logs of the current freetoken.py with
the EXP586 step C bytes (JOIN_BASE) on the same calls, on both copy paths (copy_missing, and
begin_copy + finish of the split path), for the r36 and the R584 rule:
  - timing off: the log is unchanged.
  - timing on: the log is unchanged except for two stamps on main into the new 'join' tensor,
    right before and right after main's wait for the probe stream at the last layer.
The saved timing file keeps the old (tokens, L + 3) stamp layout and adds 'join' (tokens, 2).
"""
from __future__ import annotations

import contextlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from commitmoe_prefetch import evict as ev
from commitmoe_prefetch import freetoken as ft
from commitmoe_prefetch import kernels as real_kernels
from test_shared_evict_cpu import HERE, SMALL, cfg, make_bundle, make_hooks, stub_device  # noqa: F401
from test_shared_evict_cpu import se

JOIN_BASE = '8c40673'  # freetoken.py of EXP586 step C (sha256 3569c527...), before the join stamps
KERNELS = ('stamp', 'retire', 'select_scored', 'publishing_pump', 'phase')
LAST = 39
SECTION_LAST = 5 * 7 - 1  # SECTION_LAYERS 5, 7 sections: the last layer with section work
SECTION_FIRST = SECTION_LAST - 4  # the first layer of the last section: retire and select run there


def load_at(commit: str, monkeypatch):
    root = HERE.parents[1]
    src = subprocess.run(['git', '-C', str(root), 'show', f'{commit}:commitmoe_prefetch/commitmoe_prefetch/freetoken.py'],
                         check=True, capture_output=True, text=True).stdout
    spec = importlib.util.spec_from_loader('commitmoe_prefetch.freetoken_join_base', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'commitmoe_prefetch'
    monkeypatch.setitem(sys.modules, spec.name, mod)
    exec(compile(src, f'{commit}:freetoken.py', 'exec'), mod.__dict__)
    return mod


class Recorder:
    """The fake device of one hooks instance: a log, a stream stack and fake kernels."""

    def __init__(self):
        self.log: list[tuple] = []
        self.main = FakeStream('main', self)
        self.stack = [self.main]

    @contextlib.contextmanager
    def stream(self, s):
        self.stack.append(s)
        try:
            yield
        finally:
            self.stack.pop()

    def kernels(self) -> SimpleNamespace:
        consts = {k: getattr(real_kernels, k) for k in dir(real_kernels) if k.isupper()}
        return SimpleNamespace(**consts, **{k: Launcher(k, self) for k in KERNELS})


class FakeStream:
    def __init__(self, name: str, rec: Recorder):
        self.name, self.rec = name, rec

    def wait_stream(self, other):
        self.rec.log.append(('wait_stream', self.name, other.name))

    def wait_event(self, event):
        self.rec.log.append(('wait_event', self.name, event.name))


class FakeEvent:
    def __init__(self, name: str, rec: Recorder):
        self.name, self.rec = name, rec

    def record(self, stream=None):
        on = (stream or self.rec.stack[-1]).name
        self.rec.log.append(('record', self.name, on))


class Launcher:
    def __init__(self, name: str, rec: Recorder):
        self.name, self.rec = name, rec

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            consts = tuple(sorted((k, v) for k, v in kwargs.items() if isinstance(v, (int, bool, str))))
            # A stamp names its target by shape: (L + 3,) the old stamps, (2,) the join stamps.
            target = tuple(args[0].shape) if self.name == 'stamp' else ()
            self.rec.log.append((self.name, self.rec.stack[-1].name, target, consts))
        return launch


def device_hooks(module, cfg, monkeypatch, rule: str, timing: Path | None, tmp_path: Path):  # noqa: F811
    """A hooks instance of ``module`` on the fake device; its Recorder is hooks.rec."""
    rec = Recorder()
    stub_device(monkeypatch, module)
    kwargs = dict(timing=timing)
    if rule == 'r584':
        kwargs['shared'] = (make_bundle(tmp_path / 's.pt', SMALL, group=cfg.serving_reference.int8_group), 1)
    else:
        torch.manual_seed(0)
        torch.save(dict(version=2, layers=[0, 1, 8, 39], source_layers=[6, 6, 8, 39], horizon_tokens=8,
                        release_sha256='a' * 64, weight=torch.randn(4, 256, cfg.architecture.up_dim).bfloat16(),
                        bias=torch.randn(4, 256).bfloat16()), tmp_path / 'r.pt')
        kwargs['evict'] = (tmp_path / 'r.pt', 1)
    hooks = make_hooks(module, cfg, **kwargs)
    hooks.rec = rec
    hooks.probe_stream = FakeStream('probe', rec)
    hooks.copy_events = [FakeEvent(f'copy{i}', rec) for i in range(len(hooks.copy_events))]
    hooks.scores_ready = FakeEvent('scores_ready', rec)
    hooks.demand.stream = FakeStream('demand', rec)
    hooks.demand.launch = lambda cache, layer_id, stages: rec.log.append(('demand', layer_id))
    # Set by bind(); the fake kernels do not read them. Fixed values, so that both logs carry them.
    hooks.hosts = hooks.stage_ptrs = hooks.feat_bytes = None
    hooks.num_banks, hooks.quota = 2, 64
    return hooks


def run_layer(module, hooks, path: str, layer_id: int, monkeypatch) -> list[tuple]:
    """The log of one layer's copy on ``path``: 'copy' (copy_missing) or 'split' (begin_copy + finish)."""
    rec = hooks.rec
    monkeypatch.setattr(module, 'kernels', rec.kernels())
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *a, **k: rec.stack[-1])
    monkeypatch.setattr(torch.cuda, 'stream', rec.stream)
    cache = SimpleNamespace(slot_for_id=torch.full((40, 256), -1, dtype=torch.int32), num_indices=None,
                            _copy_feat_bytes=None)
    rec.log.clear()
    if path == 'copy':
        assert hooks.copy_missing(cache, layer_id) is True
    else:
        finish = hooks.begin_copy(cache, layer_id)
        rec.log.append(('resident routes',))
        finish()
    return list(rec.log)


@pytest.fixture
def pair(cfg, monkeypatch, tmp_path):  # noqa: F811
    base = load_at(JOIN_BASE, monkeypatch)

    def make(rule: str, timing: Path | None):
        return [device_hooks(m, cfg, monkeypatch, rule, timing, tmp_path / name)
                for m, name in ((base, 'base'), (ft, 'new'))]
    for name in ('base', 'new'):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(ev, 'qlinear', se.reference_linear)
    return base, make


@pytest.mark.parametrize('rule', ['r36', 'r584'])
@pytest.mark.parametrize('path', ['copy', 'split'])
@pytest.mark.parametrize('layer_id', [SECTION_FIRST, SECTION_LAST, LAST])
def test_timing_off_launches_are_unchanged(pair, rule, path, layer_id, monkeypatch):
    base, make = pair
    old, new = make(rule, None)
    a = run_layer(base, old, path, layer_id, monkeypatch)
    b = run_layer(ft, new, path, layer_id, monkeypatch)
    assert a == b
    assert not any(e[0] == 'stamp' for e in b)
    # The fakes see the real launches: the pump in every section layer, retire and select at the
    # first layer of a section, the join only at LAST.
    names = {e[0] for e in b}
    assert ('publishing_pump' in names) == (layer_id != LAST)
    assert ({'retire', 'select_scored'} <= names) == (layer_id == SECTION_FIRST)
    assert (('wait_stream', 'main', 'probe') in b) == (layer_id == LAST)


@pytest.mark.parametrize('rule', ['r36', 'r584'])
@pytest.mark.parametrize('path', ['copy', 'split'])
def test_timing_on_adds_only_the_join_stamps_around_the_last_join(pair, rule, path, monkeypatch, tmp_path):
    base, make = pair
    old, new = make(rule, tmp_path / 't.pt')
    for layer_id in (SECTION_FIRST, SECTION_LAST, LAST - 1):  # no join: the logs are equal
        assert run_layer(base, old, path, layer_id, monkeypatch) == run_layer(ft, new, path, layer_id, monkeypatch)
    a = run_layer(base, old, path, LAST, monkeypatch)
    b = run_layer(ft, new, path, LAST, monkeypatch)
    join = ('wait_stream', 'main', 'probe')
    assert a.count(join) == 1 and a[-1] == join  # the join ends the layer in the base
    before = ('stamp', 'main', (2,), (('IDX', 0),))
    after = ('stamp', 'main', (2,), (('IDX', 1),))
    assert b == a[:-1] + [before, join, after]


@pytest.mark.parametrize('rule', ['r36', 'r584'])
def test_timing_file_keeps_the_stamp_layout_and_adds_join(pair, rule, monkeypatch, tmp_path):
    _, make = pair
    hooks = make(rule, tmp_path / 't.pt')[1]
    L = 40
    hooks.begin_request(1)
    for step in range(3):
        hooks.before_decode(step)
        hooks.stamps.copy_(torch.arange(L + 3) + 100 * step)
        hooks.join_stamps.copy_(torch.tensor([7, 9]) + 100 * step)
        hooks.after_decode(step, torch.tensor([11 + step]), torch.tensor([12 + step]))
    hooks.begin_request(2)  # the rows of request 1 are written now
    saved = torch.load(tmp_path / 't.pt')
    assert set(saved) == {'stamps', 'step', 'join', 'start_layer'}
    assert saved['stamps'].dtype == torch.int64 and tuple(saved['stamps'].shape) == (3, L + 3)
    assert torch.equal(saved['stamps'][2], torch.arange(L + 3) + 200)
    assert saved['join'].dtype == torch.int64 and saved['join'].tolist() == [[7, 9], [107, 109], [207, 209]]
    assert saved['step'].tolist() == [0, 1, 2] and saved['start_layer'] == 1
