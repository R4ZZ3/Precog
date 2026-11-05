"""CPU tests of the EXP604 and EXP605 switches: FREETOKEN_MIN_SCORE (the copy threshold of PrefetchHooks) and
FREETOKEN_POSITION_COUNTERS (cumulative counters per decode step, in PrefetchHooks and StripedDemandHooks). They also test
that both flags off leave the hooks as they were. The kernels and the graph are not run here: a test stands for the graph
by changing the counters between before_decode and after_decode."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from commitmoe_prefetch import freetoken as ft
from commitmoe_prefetch import kernels
from freetoken.moe import position_counters as pc
from freetoken.moe import striped_demand
from freetoken.moe.striped_demand import StripedDemandHooks
from test_exp603_switches_cpu import bind_cache, miss_lines, section_log  # noqa: F401  (fixtures)
from test_shared_evict_cpu import SMALL, StubBank, cfg, make_bundle, make_hooks, stub_device  # noqa: F401  (cfg is a fixture)

PUBLISHED = kernels.STATS_FIELDS.index('published')


@pytest.fixture
def stubbed(monkeypatch):
    stub_device(monkeypatch, ft)
    StubBank.log.clear()


@pytest.fixture
def info_lines(monkeypatch) -> list[str]:
    """The info lines of the predictor hooks. The logger of the fork does not propagate, so caplog does not see them."""
    lines: list[str] = []
    monkeypatch.setattr(ft.logger, 'info', lambda text, *args: lines.append(text % args))
    return lines


# ---- FREETOKEN_MIN_SCORE -------------------------------------------------------------------------------------------


@pytest.mark.parametrize('value, want', [(None, 0.0), ('', 0.0), ('0', 0.0), ('0.3', 0.3), ('0.5', 0.5), ('1', 1.0)])
def test_min_score_from_env_reads_a_value_in_range(value, want):
    env = {} if value is None else {ft.MIN_SCORE_ENV: value}
    assert ft.min_score_from_env(env) == want


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf', '-0.1', '1.01', 'abc', '0.5x'])
def test_min_score_from_env_refuses_a_bad_value(value):
    with pytest.raises(ValueError, match=ft.MIN_SCORE_ENV):
        ft.min_score_from_env({ft.MIN_SCORE_ENV: value})


@pytest.mark.parametrize('score', [float('nan'), -0.01, 1.5])
def test_the_hooks_refuse_a_min_score_outside_the_range(cfg, stubbed, score):
    with pytest.raises(ValueError, match='outside'):
        make_hooks(ft, cfg, min_score=score)


def test_the_flag_off_keeps_the_module_threshold(cfg, stubbed):
    hooks = make_hooks(ft, cfg)
    assert hooks.min_score == ft.MIN_SCORE == 0.0 and hooks.positions is None and hooks.position_folder is None


class _ScoreRecorder:
    """Stands for select_scored: records the MIN_SCORE constexpr of each launch."""

    def __init__(self, seen: list):
        self.seen = seen

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.seen.append(kwargs['MIN_SCORE'])


@pytest.mark.parametrize('score', [0.0, 0.3, 0.7])
def test_select_scored_gets_the_hook_threshold(cfg, stubbed, section_log, monkeypatch, score):
    seen: list[float] = []
    monkeypatch.setattr(kernels, 'select_scored', _ScoreRecorder(seen))
    hooks = make_hooks(ft, cfg, min_score=score)
    cache = bind_cache(hooks)
    hooks._after_demand(cache, 0, lambda: None)
    assert seen == [score]


@pytest.mark.parametrize('score, want', [(0.0, []), (0.3, ['expert predictor: min score 0.3']),
                                         (0.5, ['expert predictor: min score 0.5']), (0.7, ['expert predictor: min score 0.7'])])
def test_the_threshold_log_line_appears_only_with_the_flag(cfg, stubbed, info_lines, score, want):
    """The pass A gate reads this line (reserve() writes it): T3, T5 and T7 name their value, T0 logs as earlier serves."""
    make_hooks(ft, cfg, min_score=score)
    assert [line for line in info_lines if 'min score' in line] == want


# ---- FREETOKEN_POSITION_COUNTERS: the counter object ---------------------------------------------------------------


def test_the_counter_folder_must_be_new(tmp_path):
    (tmp_path / 'p').mkdir()
    with pytest.raises(FileExistsError):
        pc.PositionCounters(tmp_path / 'p', torch.device('cpu'))


def test_the_counter_folder_parent_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        pc.PositionCounters(tmp_path / 'missing' / 'p', torch.device('cpu'))


def test_a_skipped_step_stops_the_counters(tmp_path):
    p = pc.PositionCounters(tmp_path / 'p', torch.device('cpu'), max_rows=8)
    misses = torch.zeros(2, dtype=torch.int64)
    p.snap(0, None, misses)
    p.snap(1, None, misses)
    with pytest.raises(RuntimeError, match='skipped'):
        p.snap(3, None, misses)


def test_a_request_longer_than_the_rows_stops_the_counters(tmp_path):
    p = pc.PositionCounters(tmp_path / 'p', torch.device('cpu'), max_rows=3)
    misses = torch.zeros(2, dtype=torch.int64)
    for row in range(3):
        p.snap(row, None, misses)
    with pytest.raises(RuntimeError, match='outside'):
        p.snap(3, None, misses)


def test_a_write_with_the_wrong_step_count_stops_the_counters(tmp_path):
    p = pc.PositionCounters(tmp_path / 'p', torch.device('cpu'), max_rows=8)
    misses = torch.zeros(2, dtype=torch.int64)
    for row in range(3):
        p.snap(row, None, misses)
    with pytest.raises(RuntimeError, match='rows for'):
        p.write(1, 3, [0, 0, 0])


def test_the_snapshot_is_a_copy_not_a_view(tmp_path):
    """The rows must keep the value at the snapshot, not follow the live counter."""
    p = pc.PositionCounters(tmp_path / 'p', torch.device('cpu'), max_rows=4)
    published, misses = torch.tensor(5), torch.tensor([1, 2])
    p.snap(0, published, misses)
    published.add_(10)
    misses.add_(10)
    p.snap(1, published, misses)
    path = p.write(9, 1, [15, 11, 12])
    got = torch.load(path, weights_only=True)
    assert got['rows'].tolist() == [[5, 1, 2], [15, 11, 12]] and got['end'].tolist() == [15, 11, 12]
    assert got['uid'] == 9 and got['steps'] == 1 and tuple(got['columns']) == pc.COLUMNS
    assert not list((tmp_path / 'p').glob('*.tmp'))


def test_folder_from_env_reads_only_a_non_empty_value():
    assert pc.folder_from_env({}) is None and pc.folder_from_env({'FREETOKEN_POSITION_COUNTERS': ''}) is None
    assert pc.folder_from_env({'FREETOKEN_POSITION_COUNTERS': '/x/y'}) == Path('/x/y')


def test_the_rows_cover_the_longest_request_of_the_serves():
    """8448 is the --max-seq-len-override of the EXP604 and EXP605 serves: no request can decode more steps."""
    assert pc.MAX_ROWS == 8448 + 1


# ---- FREETOKEN_POSITION_COUNTERS: the predictor hooks --------------------------------------------------------------


def predictor_request(hooks, uid: int, steps: int, per_step: list[tuple[int, int, int]], prefill=(0, 0)) -> None:
    """The engine's order for one request. ``prefill``: (stage, host) misses of the prefill. ``per_step``: per decode step,
    the (published, stage, host) increments that the graph of that step makes."""
    hooks.begin_request(uid)
    hooks.demand.counters[0, 0] += prefill[0]
    hooks.demand.counters[1, 1] += prefill[1]
    one = torch.zeros(1, dtype=torch.int64)
    for step in range(steps):
        hooks.before_decode(step)
        published, stage, host = per_step[step]
        hooks.stats[PUBLISHED] += published
        hooks.demand.counters[2, 0] += stage
        hooks.demand.counters[3, 1] += host
        hooks.demand.counters[4, 2] += 7   # host-while-pending: not a column, must not leak into one
        hooks.after_decode(step, one, one)


def test_predictor_counters_write_one_cumulative_file_per_finished_request(cfg, stubbed, tmp_path):
    folder = tmp_path / 'pos'
    hooks = make_hooks(ft, cfg, position_counters=folder)
    a = [(1, 0, 2), (0, 1, 0), (3, 0, 1)]
    b = [(2, 2, 2), (0, 0, 0)]
    predictor_request(hooks, 11, 3, a, prefill=(4, 5))
    predictor_request(hooks, 12, 2, b)
    predictor_request(hooks, 13, 2, b)     # the tail: no later request, so no file
    hooks.begin_request(13)                # a second prefill chunk of one uid ends nothing
    assert sorted(p.name for p in folder.iterdir()) == ['positions_0000.pt', 'positions_0001.pt']
    first = torch.load(folder / 'positions_0000.pt', weights_only=True)
    assert first['uid'] == 11 and first['steps'] == 3
    # Row 0 holds the prefill misses; each later row adds that step's increments.
    assert first['rows'].tolist() == [[0, 4, 5], [1, 4, 7], [1, 5, 7], [4, 5, 8]]
    assert first['end'].tolist() == first['rows'][-1].tolist()
    second = torch.load(folder / 'positions_0001.pt', weights_only=True)
    assert second['rows'].tolist() == [[0, 0, 0], [2, 2, 2], [2, 2, 2]] and second['end'].tolist() == [2, 2, 2]


def test_predictor_counters_end_equals_the_request_summary(cfg, stubbed, tmp_path, monkeypatch):
    """The proof gate compares the last row with the request summary that the same hook logs."""
    folder = tmp_path / 'pos'
    hooks = make_hooks(ft, cfg, position_counters=folder)
    logged: list[dict] = []
    summary = hooks.summary
    monkeypatch.setattr(hooks, 'summary', lambda: (logged.append(summary()), logged[-1])[1])
    predictor_request(hooks, 1, 4, [(1, 1, 1)] * 4, prefill=(2, 3))
    hooks.begin_request(2)
    got = torch.load(folder / 'positions_0000.pt', weights_only=True)
    s = logged[0]
    assert got['end'].tolist() == [s['published'], s['misses_from_stage'], s['misses_from_host']] == [4, 6, 7]


def test_predictor_counters_skip_a_request_with_no_decode_step(cfg, stubbed, tmp_path):
    folder = tmp_path / 'pos'
    hooks = make_hooks(ft, cfg, position_counters=folder)
    hooks.begin_request(4)   # a prefill-only request
    predictor_request(hooks, 5, 2, [(0, 0, 1)] * 2)
    predictor_request(hooks, 6, 1, [(0, 0, 0)])
    assert [p.name for p in folder.iterdir()] == ['positions_0000.pt']
    assert torch.load(folder / 'positions_0000.pt', weights_only=True)['uid'] == 5


def test_predictor_counters_log_their_folder(cfg, stubbed, tmp_path, info_lines):
    hooks = make_hooks(ft, cfg, position_counters=tmp_path / 'pos')
    assert hooks.positions is not None and hooks.positions.folder == tmp_path / 'pos' and (tmp_path / 'pos').is_dir()
    assert f"expert predictor: position counters to {tmp_path / 'pos'}" in info_lines


def test_predictor_counters_off_make_no_folder_and_no_rows(cfg, stubbed, tmp_path):
    hooks = make_hooks(ft, cfg)
    predictor_request(hooks, 1, 2, [(1, 1, 1)] * 2)
    hooks.begin_request(2)
    assert hooks.positions is None and list(tmp_path.iterdir()) == []


def test_create_decode_hooks_passes_both_flags():
    """The env wiring: the two readers feed the two new constructor arguments, in that order."""
    text = Path(ft.__file__).read_text()
    assert ('_folder_from_env(ORACLE_PICKS_ENV), min_score_from_env(os.environ),\n'
            '                         position_folder_from_env(os.environ, POSITION_COUNTERS_ENV))') in text
    assert ft.POSITION_COUNTERS_ENV == 'FREETOKEN_POSITION_COUNTERS' and ft.MIN_SCORE_ENV == 'FREETOKEN_MIN_SCORE'


# ---- FREETOKEN_POSITION_COUNTERS: the no-predictor hooks -----------------------------------------------------------


def striped(folder: Path | None) -> StripedDemandHooks:
    hooks = StripedDemandHooks(40, 256, 16, position_counters=folder)
    demand = SimpleNamespace(counters=torch.zeros((40, 3), dtype=torch.int64), check_cache=lambda cache: None)
    demand.reset = lambda: demand.counters.zero_()   # the kernel object zeroes its counters in reset
    hooks.demand = demand
    if folder is not None:
        hooks.positions = pc.PositionCounters(folder, torch.device('cpu'))   # what reserve() makes on the engine device
    return hooks


def striped_request(hooks, uid: int, per_step: list[tuple[int, int]], prefill=(0, 0)) -> None:
    hooks.begin_request(uid)
    hooks.demand.counters[0, :2] += torch.tensor(prefill)
    for step, (stage, host) in enumerate(per_step):
        hooks.before_decode(step)
        hooks.demand.counters[5, 0] += stage
        hooks.demand.counters[6, 1] += host
        hooks.demand.counters[7, 2] += 3
        hooks.after_decode(step, None, None)


def test_striped_counters_write_misses_and_a_zero_published_column(tmp_path):
    folder = tmp_path / 'pos'
    hooks = striped(folder)
    striped_request(hooks, 1, [(0, 2), (1, 0)], prefill=(0, 9))
    striped_request(hooks, 2, [(0, 1)])
    striped_request(hooks, 3, [(0, 0)])   # the tail
    assert sorted(p.name for p in folder.iterdir()) == ['positions_0000.pt', 'positions_0001.pt']
    first = torch.load(folder / 'positions_0000.pt', weights_only=True)
    assert first['rows'].tolist() == [[0, 0, 9], [0, 0, 11], [0, 1, 11]] and first['end'].tolist() == [0, 1, 11]
    assert torch.load(folder / 'positions_0001.pt', weights_only=True)['rows'].tolist() == [[0, 0, 0], [0, 0, 1]]


def test_striped_counters_end_equals_the_miss_log(tmp_path, miss_lines):
    """Pass B checks the F0QC files against the FREETOKEN_DEMAND_MISS_LOG line of the same request."""
    hooks = striped(tmp_path / 'pos')
    hooks.log_misses = True
    striped_request(hooks, 1, [(0, 2), (0, 3)], prefill=(0, 1))
    striped_request(hooks, 2, [(0, 0)])
    end = torch.load(tmp_path / 'pos' / 'positions_0000.pt', weights_only=True)['end'].tolist()
    assert end == [0, 0, 6] and miss_lines[-1].startswith('striped demand: request 1, 2 decode steps')


def test_striped_reserve_makes_the_counters_on_the_engine_device(tmp_path, monkeypatch):
    made = []
    monkeypatch.setattr(striped_demand, 'StripedDemand', lambda *a, **k: made.append(1) or SimpleNamespace())
    monkeypatch.setattr(striped_demand, 'PositionCounters', lambda folder, device: ('pc', folder, device))
    hooks = StripedDemandHooks(40, 256, 16, position_counters=tmp_path / 'pos')
    hooks.reserve(SimpleNamespace(device='dev'))
    assert hooks.positions == ('pc', tmp_path / 'pos', 'dev')


def test_striped_counters_off_change_nothing(tmp_path):
    hooks = striped(None)
    striped_request(hooks, 1, [(1, 1)] * 3)
    striped_request(hooks, 2, [(1, 1)])
    assert hooks.positions is None and list(tmp_path.iterdir()) == []


def test_the_engine_passes_the_counter_folder_to_the_no_predictor_hooks():
    text = (Path(ft.__file__).parents[2] / 'python/freetoken/engine/engine.py').read_text()
    assert 'position_counters=folder_from_env(os.environ))' in text
    assert 'from freetoken.moe.position_counters import folder_from_env' in text
