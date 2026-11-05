"""CPU tests of the EXP603 switches on PrefetchHooks and StripedDemandHooks: the no-copy switch, the oracle record and picks, the F0
miss log, their refusals, and that every flag off leaves the hooks as they were. The kernels and the graph are not run here."""
from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from commitmoe_prefetch import freetoken as ft
from commitmoe_prefetch import kernels
from freetoken.moe import striped_demand
from freetoken.moe.striped_demand import StripedDemandHooks
from test_shared_evict_cpu import SMALL, StubBank, cfg, feed, make_bundle, make_hooks, stub_device  # noqa: F401  (cfg is a fixture)


@pytest.fixture
def stubbed(monkeypatch):
    stub_device(monkeypatch, ft)
    StubBank.log.clear()


@pytest.fixture
def bundle(cfg, stubbed, tmp_path) -> Path:
    return make_bundle(tmp_path / 's.pt', SMALL, group=cfg.serving_reference.int8_group)


def routes(steps: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.stack([torch.randperm(256, generator=g)[:8] for _ in range(40)]) for _ in range(steps)])


def run_request(hooks, uid: int, rows: torch.Tensor) -> None:
    """The engine's order for one request: begin_request at prefill, then before_decode, the graph, after_decode."""
    hooks.begin_request(uid)
    one = torch.zeros(1, dtype=torch.int64)
    for step, row in enumerate(rows):
        hooks.before_decode(step)
        hooks.live['routes'].copy_(row.to(torch.int64))
        hooks.after_decode(step, one, one)


@pytest.fixture
def plain(cfg, stubbed):
    return make_hooks(ft, cfg)


def test_the_record_writes_one_file_per_finished_request_and_keeps_the_order(cfg, stubbed, tmp_path):
    folder = tmp_path / 'routes'
    hooks = make_hooks(ft, cfg, oracle_record=folder)
    a, b, tail = routes(3, 1), routes(5, 2), routes(2, 3)
    run_request(hooks, 11, a)
    run_request(hooks, 12, b)
    run_request(hooks, 13, tail)     # no later request: the tail stays unwritten
    hooks.begin_request(13)          # a second prefill chunk of one uid ends nothing
    assert sorted(p.name for p in folder.iterdir()) == ['routes_0000.pt', 'routes_0001.pt']
    got = [torch.load(folder / f'routes_{i:04d}.pt', weights_only=True) for i in range(2)]
    assert got[0].dtype == torch.int16 and torch.equal(got[0], a.to(torch.int16)) and torch.equal(got[1], b.to(torch.int16))


def test_a_request_with_no_decode_step_takes_no_file_number(cfg, stubbed, tmp_path):
    folder = tmp_path / 'routes'
    hooks = make_hooks(ft, cfg, oracle_record=folder)
    hooks.begin_request(7)   # the prefill warm-up request: no decode step
    run_request(hooks, 8, routes(2, 1))
    run_request(hooks, 9, routes(2, 2))
    assert [p.name for p in folder.iterdir()] == ['routes_0000.pt']


def write_table(folder: Path, *rows: torch.Tensor) -> None:
    folder.mkdir()
    for i, r in enumerate(rows):
        torch.save(r.to(torch.int16), folder / f'routes_{i:04d}.pt')


def test_oracle_marks_only_the_routes_of_the_step_and_scores_the_rest_under_min_score(cfg, stubbed, tmp_path):
    a = routes(4, 1)
    write_table(tmp_path / 't', a)
    hooks = make_hooks(ft, cfg, oracle_picks=tmp_path / 't')
    hooks.begin_request(1)
    hooks.before_decode(0)
    hooks.before_decode(2)   # the step index selects the row
    want = torch.full((hooks.heads, 256), ft.ORACLE_OTHER)
    want.scatter_(1, a[2, hooks.first:].to(torch.int64), ft.ORACLE_NEEDED)
    assert torch.equal(hooks.oracle_mask, want)
    # Trap: select_scored treats a score at MIN_SCORE or above as eligible.
    assert ft.ORACLE_OTHER < ft.MIN_SCORE <= ft.ORACLE_NEEDED
    assert int((hooks.oracle_mask == ft.ORACLE_NEEDED).sum()) == hooks.heads * 8


def test_oracle_score_overwrites_the_bank_result_and_keeps_the_bank_call(cfg, stubbed, tmp_path):
    write_table(tmp_path / 't', routes(4, 1))
    hooks = make_hooks(ft, cfg, oracle_picks=tmp_path / 't')
    hooks.begin_request(1)
    hooks.before_decode(0)
    hooks.before_decode(1)
    hooks._score()
    assert 'bank' in StubBank.log and torch.equal(hooks.probs, hooks.oracle_mask)


def test_identity_counts_stay_zero_on_the_recorded_routes(cfg, stubbed, tmp_path):
    a, b = routes(3, 1), routes(2, 2)
    write_table(tmp_path / 't', a, b)
    hooks = make_hooks(ft, cfg, oracle_picks=tmp_path / 't')
    run_request(hooks, 1, a)
    s = hooks.summary()
    assert s['oracle_mismatch_steps'] == 0 and s['oracle_missing_steps'] == 0
    run_request(hooks, 2, b)
    s = hooks.summary()
    assert s['oracle_mismatch_steps'] == 0 and s['oracle_missing_steps'] == 0


def test_identity_counts_a_differing_route_and_a_missing_step(cfg, stubbed, tmp_path):
    a = routes(3, 1)
    write_table(tmp_path / 't', a)
    hooks = make_hooks(ft, cfg, oracle_picks=tmp_path / 't')
    other = a.clone()
    other[1, 20, 3] = (other[1, 20, 3] + 1) % 256
    run_request(hooks, 1, other)
    assert hooks.summary()['oracle_mismatch_steps'] == 1
    longer = torch.cat([a, routes(2, 9)])
    run_request(hooks, 2, longer)                     # request 2 has no recorded file at all: every step is missing
    s = hooks.summary()
    assert s['oracle_missing_steps'] == 5 and s['oracle_mismatch_steps'] == 0


def test_the_kernel_rule_takes_only_the_needed_experts_from_the_mask(cfg, stubbed, tmp_path):
    """select_scored keeps a score at MIN_SCORE or above. The rule on the mask must give exactly the routes of the step."""
    a = routes(2, 4)
    write_table(tmp_path / 't', a)
    hooks = make_hooks(ft, cfg, oracle_picks=tmp_path / 't')
    hooks.begin_request(1)
    hooks.before_decode(0)
    hooks.before_decode(1)
    eligible = (hooks.oracle_mask >= ft.MIN_SCORE).nonzero()
    got = {(int(h) + hooks.first, int(e)) for h, e in eligible}
    assert got == {(layer, int(e)) for layer in range(hooks.first, 40) for e in a[1, layer]}


def test_a_missing_row_leaves_no_expert_eligible(cfg, stubbed, tmp_path):
    a = routes(1, 1)
    write_table(tmp_path / 't', a)
    hooks = make_hooks(ft, cfg, oracle_picks=tmp_path / 't')
    hooks.begin_request(1)
    hooks.before_decode(0)
    hooks.before_decode(1)   # past the recorded row
    assert hooks.oracle_mask.eq(ft.ORACLE_OTHER).all()


def test_the_identity_counts_reset_per_request(cfg, stubbed, tmp_path):
    a = routes(1, 1)
    write_table(tmp_path / 't', a)
    hooks = make_hooks(ft, cfg, oracle_picks=tmp_path / 't')
    run_request(hooks, 1, routes(3, 2))
    assert hooks.summary()['oracle_missing_steps'] == 2 and hooks.summary()['oracle_mismatch_steps'] == 1
    hooks.begin_request(2)
    s = hooks.summary()
    assert s['oracle_missing_steps'] == 0 and s['oracle_mismatch_steps'] == 0 and hooks.oracle_mask.eq(ft.ORACLE_OTHER).all()


@pytest.mark.parametrize('names', [(), ('routes_0001.pt',), ('routes_0000.pt', 'routes_0002.pt')])
def test_a_table_with_a_gap_is_refused(cfg, stubbed, tmp_path, names):
    (tmp_path / 't').mkdir()
    for n in names:
        torch.save(routes(1).to(torch.int16), tmp_path / 't' / n)
    with pytest.raises(ValueError, match='without gaps'):
        make_hooks(ft, cfg, oracle_picks=tmp_path / 't')


def test_the_record_folder_must_be_new(cfg, stubbed, tmp_path):
    (tmp_path / 'r').mkdir()
    with pytest.raises(FileExistsError):
        make_hooks(ft, cfg, oracle_record=tmp_path / 'r')


def test_refusals(cfg):
    with pytest.raises(ValueError, match=ft.NO_COPIES_ENV):
        ft.PrefetchHooks(cfg, {}, 16, no_copies=True)
    with pytest.raises(ValueError, match='two pick rules'):
        ft.PrefetchHooks(cfg, {}, 16, pick=('ema', 1), oracle_picks=Path('/t'))
    with pytest.raises(ValueError, match='records or replays'):
        ft.PrefetchHooks(cfg, {}, 16, oracle_record=Path('/a'), oracle_picks=Path('/b'))
    for kwargs in (dict(oracle_picks=Path('/b')), dict(pick=('ema', 1))):
        with pytest.raises(ValueError, match='no picks to replace'):
            ft.PrefetchHooks(cfg, {}, 16, shared=(Path('/b.pt'), 1), shared_fast='blockn', no_copies=True, **kwargs)


def test_every_flag_off_keeps_the_hooks_as_they_were(plain):
    assert plain.no_copies is False and plain.oracle_record is None and plain.oracle_picks is None
    for name in ('oracle_mask', 'oracle_table', 'oracle_mismatch', 'oracle_missing'):
        assert not hasattr(plain, name)
    run_request(plain, 1, routes(2, 1))
    s = plain.summary()
    assert 'oracle_mismatch_steps' not in s and 'oracle_missing_steps' not in s
    assert plain.oracle_log == []    # no route is held without the record flag


class _Recorder:
    """Stands for one kernel table: any variant call writes the kernel name in the log."""

    def __init__(self, name: str, log: list[str]):
        self.name, self.log = name, log

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.log.append(self.name)


@pytest.fixture
def section_log(monkeypatch) -> list[str]:
    """The names of the section kernels and of the last-layer join, in call order. The stream calls need no device here."""
    log: list[str] = []
    for name in ('retire', 'select_scored', 'publishing_pump'):
        monkeypatch.setattr(kernels, name, _Recorder(name, log))
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: SimpleNamespace(wait_stream=lambda s: log.append('join')))
    monkeypatch.setattr(torch.cuda, 'stream', lambda s: contextlib.nullcontext())
    return log


def bind_cache(hooks) -> SimpleNamespace:
    """The attributes of bind that the section kernels name. The recorders ignore the values."""
    cache = SimpleNamespace(slot_for_id=torch.full((40, 256), -1, dtype=torch.int32))
    hooks.cache = cache
    for name in ('hosts', 'stage_ptrs', 'feat_bytes', 'busy', 'layer', 'num_banks', 'quota'):
        if not hasattr(hooks, name):
            setattr(hooks, name, None)
    return cache


def test_the_no_copy_switch_runs_no_section_work_and_keeps_the_last_join(cfg, bundle, section_log):
    hooks = make_hooks(ft, cfg, shared=(bundle, 1), shared_fast='blockn', no_copies=True)
    cache, waited = bind_cache(hooks), []
    for layer in (0, 7, 39):
        hooks._after_demand(cache, layer, lambda layer=layer: waited.append(layer))
    assert section_log == ['join'] and waited == []


def test_the_same_recorder_sees_the_section_work_with_the_switch_off(cfg, bundle, section_log):
    """The control of the test above: without it, a recorder that sees nothing would pass that test too."""
    hooks = make_hooks(ft, cfg, shared=(bundle, 1), shared_fast='blockn')
    cache, waited = bind_cache(hooks), []
    hooks._after_demand(cache, 0, lambda: waited.append(0))
    assert section_log == ['retire', 'select_scored', 'publishing_pump'] and waited == [0]


def test_the_no_copy_score_publishes_keep_scores_and_runs_no_bank(cfg, bundle):
    hooks = make_hooks(ft, cfg, shared=(bundle, 1), shared_fast='blockn', no_copies=True)
    hooks.cache = SimpleNamespace(slot_for_id=torch.full((40, 256), -1, dtype=torch.int32))
    copied = []
    copy = hooks._copy_raw
    hooks._copy_raw = lambda copy=copy: (copied.append(1), copy())
    feed(hooks, 3, 0)
    before, probs = hooks.raw['prior_before'].clone(), hooks.probs.clone()
    StubBank.log.clear()
    hooks._score()
    assert StubBank.log == ['scores_ready'] and copied == []     # no 'bank' and no raw copy
    assert int(hooks.learned.valid) == 1
    assert torch.equal(hooks.raw['prior_before'], before) and torch.equal(hooks.probs, probs)


def test_the_no_copy_keep_scores_equal_the_full_serve(cfg, bundle):
    """The eviction of arm E must be the eviction of R584: the same keep scores on the same inputs."""
    on = make_hooks(ft, cfg, shared=(bundle, 1), shared_fast='blockn', no_copies=True)
    off = make_hooks(ft, cfg, shared=(bundle, 1), shared_fast='blockn')
    for hooks in (on, off):
        hooks.cache = SimpleNamespace(slot_for_id=torch.full((40, 256), -1, dtype=torch.int32))
    for step in (1, 2, 3):
        for hooks in (on, off):
            feed(hooks, step, step + 70)
            hooks._score()
        assert torch.equal(on.learned.keep, off.learned.keep) and int(on.learned.valid) == int(off.learned.valid)


# ---- F0 miss log ---------------------------------------------------------------------------------------------------


def striped(log_misses: bool) -> StripedDemandHooks:
    hooks = StripedDemandHooks(40, 256, 16, log_misses=log_misses)
    demand = SimpleNamespace(counters=torch.zeros((40, 3), dtype=torch.int64))
    demand.reset = lambda: demand.counters.zero_()   # the kernel object zeroes its counters in reset
    hooks.demand = demand
    return hooks


def one_step(hooks, uid: int, misses: list[int], steps: int) -> None:
    hooks.begin_request(uid)
    hooks.demand.counters[:, 1] = torch.tensor(misses)
    one = torch.zeros(1)
    for s in range(steps):
        hooks.after_decode(s, one, one)


@pytest.fixture
def miss_lines(monkeypatch) -> list[str]:
    """The info lines of the hooks. The logger of the fork does not propagate, so caplog would need a change of global state."""
    lines: list[str] = []
    monkeypatch.setattr(striped_demand.logger, 'info', lambda text, *args: lines.append(text % args))
    return lines


def test_f0_miss_log_writes_one_line_per_finished_request(miss_lines):
    hooks = striped(True)
    misses = list(range(40))
    one_step(hooks, 5, misses, 255)
    hooks.begin_request(5)     # another prefill chunk of the same uid is no new request and logs nothing
    hooks.demand.counters[:, 1] = torch.tensor(misses)
    hooks.after_decode(254, None, None)
    assert miss_lines == []
    one_step(hooks, 6, misses, 3)
    assert miss_lines == [f'striped demand: request 5, 255 decode steps, misses per layer: {misses}']


def test_f0_miss_log_skips_a_request_with_no_decode_step(miss_lines):
    hooks = striped(True)
    hooks.begin_request(4)
    one_step(hooks, 5, list(range(40)), 2)
    assert miss_lines == []


def test_f0_miss_log_off_writes_nothing(miss_lines):
    hooks = striped(False)
    one_step(hooks, 5, list(range(40)), 3)
    one_step(hooks, 6, list(range(40)), 3)
    assert miss_lines == []


def test_the_engine_reads_the_f0_flag_only_at_value_one():
    text = (Path(ft.__file__).parents[2] / 'python/freetoken/engine/engine.py').read_text()
    assert 'log_misses=os.environ.get("FREETOKEN_DEMAND_MISS_LOG") == "1"' in text
