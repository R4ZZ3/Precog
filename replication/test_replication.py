"""CPU tests for run_bench.py and report.py. No GPU, no server.

Run from the repository root: .venv/bin/python -m pytest -q replication/test_replication.py

Two kinds of test:
  - synthetic: the gate and report logic on small made-up inputs, including each failure the gates must catch.
  - study records: when the RTX 4080 records of the CommitMoE study exist on this machine (STUDY below), the gates read
    the real EXP609 server logs, and the report statistics must reproduce the study's EXP609 estimate and the reference
    numbers in reference_rtx4080.json. On another machine these tests skip.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report  # noqa: E402
import run_bench as rb  # noqa: E402

STUDY = Path('/mnt/e/CommoditizeTeraModel/commitmoe_runs')
EXP609 = STUDY / 'exp609_a1' / 'bench'
needs_study = pytest.mark.skipif(not (EXP609 / 'pass1.json').is_file(), reason='study records not on this machine')


# ---- streaming values ------------------------------------------------------------------------------------------------

def request(n: int, decode: int, stamps: list[float]) -> dict:
    return dict(number=n, decode=decode, completion_tokens=decode, finish_reason='length', text='x' * len(stamps),
                pieces=['x'] * len(stamps), piece_kinds=['c'] * len(stamps), events=len(stamps), stamps_raw=stamps)


def test_ms_per_token_uses_decode_minus_one():
    q = request(1, 5, [10.0, 10.01, 10.02, 10.03, 10.04])
    assert rb.ms_per_token(q) == pytest.approx(1000 * 0.04 / 4)
    assert report.ms_per_token(q) == rb.ms_per_token(q)


@pytest.mark.parametrize('stamps', [[1.0], [1.0, 0.5], [1.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
def test_stamps_rejected(stamps):
    q = request(1, 5, stamps)
    assert not rb.stamps_ok(q)
    with pytest.raises(rb.Stop):
        rb.ms_per_token(q)


# ---- prompts ---------------------------------------------------------------------------------------------------------

def test_prompt_files_match_pins():
    for spec in rb.PASSES.values():
        assert rb.sha256(spec['prompts']) == spec['sha256']
    assert rb.sha256(rb.BUNDLE) == rb.BUNDLE_SHA256


def test_prompt_rows_full_and_smoke():
    cfg = dict(rb.CONFIG)
    code = rb.prompt_rows('code', cfg)
    assert [r.number for r in code] == list(range(41)) and {r.decode for r in code[1:]} == {1024}
    mq = rb.prompt_rows('math_qa', cfg)
    assert len(mq) == 81 and {r.group for r in mq[1::2]} == {'math'} and {r.group for r in mq[2::2]} == {'qa'}
    assert mq[0].group == 'warmup' and mq[0].decode == 256
    cfg['max_scored_rows'] = 2
    smoke = rb.prompt_rows('math_qa', cfg)
    assert [r.number for r in smoke] == [0, 1, 2, 3, 4]   # both groups keep two rows


# ---- command and environment -----------------------------------------------------------------------------------------

def fake_paths(tmp: Path, model: str = 'snap', predictor: str = 'pred') -> rb.LocalPaths:
    return rb.LocalPaths.model_construct(python=tmp / 'py', cuda_home=tmp / 'cuda', model=tmp / model,
                                         predictor=tmp / predictor)


def test_serve_cmd_differs_only_in_slots_and_predictor(tmp_path):
    p = fake_paths(tmp_path)
    f0q, r584 = rb.serve_cmd('F0Q', 1, p, rb.CONFIG), rb.serve_cmd('R584', 1, p, rb.CONFIG)
    assert r584[:-2] == [str(rb.CONFIG['slots']['R584']) if x == str(rb.CONFIG['slots']['F0Q']) else x for x in f0q]
    assert r584[-2:] == ['--expert-predictor', str(p.predictor)]
    for flag, value in (('--num-pages', '8284'), ('--memory-ratio', '0.9'), ('--max-seq-len-override', '8448'),
                        ('--moe-demand-copy-stripes', '16'), ('--moe-hybrid-max-fetch', '-1')):
        assert f0q[f0q.index(flag) + 1] == value


def test_env_is_a_whitelist(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    monkeypatch.setenv('HF_TOKEN', 'secret')
    env = rb.serve_env('R584', fake_paths(tmp_path), rb.CONFIG, tmp_path)
    assert 'PYTORCH_CUDA_ALLOC_CONF' not in env and 'HF_TOKEN' not in env
    assert env['FREETOKEN_SHARED_EVICT'] == str(rb.BUNDLE) and env['FREETOKEN_SHARED_EVICT_FAST'] == 'blockn'
    assert env['FREETOKEN_PIN_BUDGET_GB'] == '32' and env[rb.CACHE_VAR] == str(tmp_path)
    f0q = rb.serve_env('F0Q', fake_paths(tmp_path), rb.CONFIG, None)
    assert not any(k.startswith('FREETOKEN_') and k != 'FREETOKEN_PIN_BUDGET_GB' for k in f0q) and rb.CACHE_VAR not in f0q


def test_free_port_pair():
    import socket
    port = rb.free_port()
    for p in (port, port + 1):
        with socket.socket() as s:
            s.bind(('127.0.0.1', p))


# ---- log gates on synthetic logs -------------------------------------------------------------------------------------

def server_args_line(arm: str, paths: rb.LocalPaths) -> str:
    args = dict(rb.OFFLOAD_ARGS, moe_cache_size=str(rb.CONFIG['slots'][arm]),
                expert_predictor=f"'{paths.predictor}'" if arm == 'R584' else 'None', model_path=f"'{paths.model}'")
    return 'ServerArgs(' + ', '.join(f'{k}={v}' for k, v in args.items()) + ')'


def synthetic_log(arm: str, paths: rb.LocalPaths) -> str:
    slots = rb.CONFIG['slots'][arm]
    lines = [server_args_line(arm, paths),
             'MEMORY_PHASE {"phase": "before_load", "device_free": 30000000000}',
             f'cache config: moe_cache_size={slots}, moe_cache_rate=None',
             'Allocating 8284 tokens for KV cache',
             'Embedding tables in pinned host RAM (--embed-weights host): 0.95 GiB',
             'Free memory after initialization: 15.10 GiB',
             'Free GPU memory after capturing CUDA graphs: 15.00 GiB',
             'MEMORY_PHASE {"phase": "after_capture", "device_free": 16000000000}']
    if arm == 'R584':
        lines += ['expert predictor: 34 heads L6-L39, 1.2 MiB, caps [2, 2, 2, 2, 2, 3, 3]',
                  f'shared learned eviction: all 40 layers from layer 1, R584 44.7 MiB ({rb.BUNDLE}), fast blockn']
    return '\n'.join(lines) + '\n'


def cache_calls(arm: str) -> list[dict]:
    slots = rb.CONFIG['slots'][arm]
    base = dict(decode_target='gpu', cpu_layer_ids=[], locked_hybrid=[], pool_sizes=[slots], cache_size=slots,
                distinct_layouts=1)
    return [dict(base, source='check'), dict(base, source='init', **rb.IDENTITY[arm])]


GEOMETRY = {'F0Q': dict(num_pages=8284, page_size=1, moe_cache_size=3571),
            'R584': dict(num_pages=8284, page_size=1, moe_cache_size=3496)}


@pytest.mark.parametrize('arm', ['F0Q', 'R584'])
def test_pre_gates_pass_on_good_log(arm, tmp_path):
    p = fake_paths(tmp_path)
    facts = rb.facts_of(synthetic_log(arm, p))
    gates = rb.pre_gates(arm, facts, GEOMETRY[arm], 'snap', cache_calls(arm), p, rb.CONFIG)
    assert all(gates.values()), [k for k, v in gates.items() if not v]
    assert rb.device_use(facts) == 14_000_000_000


MUTATIONS = {
    'free_after_capture': lambda t: t.replace('capturing CUDA graphs: 15.00', 'capturing CUDA graphs: 0.30'),
    'graphs_on': lambda t: t + 'CUDA graph is disabled.\n',
    'kv_pages': lambda t: t.replace('Allocating 8284 tokens', 'Allocating 8000 tokens'),
    'page_size_kept': lambda t: t + 'Page size 1 is auto-adjusted to 16\n',
    'embed_host': lambda t: t.replace('Embedding tables in pinned host RAM', 'Embedding tables on GPU'),
    'server_args': lambda t: t.replace("cache_type='naive'", "cache_type='radix'"),
    'no_log_error': lambda t: t + 'RuntimeError: CUDA error: an illegal memory access was encountered\n',
    'no_diagnostic_switch': lambda t: t + 'expert predictor: no prefetch copies\n',
    'no_rows_rule': lambda t: t + 'learned eviction: layers 1-39\n',
    'memory_phases': lambda t: t + 'MEMORY_PHASE {"phase": "after_capture", "device_free": 1}\n',
    'logged_slots': lambda t: t + 'cache config: moe_cache_size=3000, moe_cache_rate=None\n',
}


@pytest.mark.parametrize('gate', sorted(MUTATIONS))
def test_each_log_mutation_fails_its_gate(gate, tmp_path):
    p = fake_paths(tmp_path)
    facts = rb.facts_of(MUTATIONS[gate](synthetic_log('R584', p)))
    gates = rb.pre_gates('R584', facts, GEOMETRY['R584'], 'snap', cache_calls('R584'), p, rb.CONFIG)
    assert gates[gate] is False


def test_predictor_and_cache_mixups_fail(tmp_path):
    p = fake_paths(tmp_path)
    # F0Q with a predictor line, R584 with the F0Q hook class, a CPU layer, a second shared bundle row, wrong caps
    f0q = rb.pre_gates('F0Q', rb.facts_of(synthetic_log('R584', p).replace('3496', '3571')), GEOMETRY['F0Q'], 'snap',
                       cache_calls('F0Q'), p, rb.CONFIG)
    assert not f0q['predictor_loads'] and not f0q['no_shared_rule'] and not f0q['server_args']
    calls = cache_calls('R584')
    calls[1]['hooks_type'] = 'StripedDemandHooks'
    assert not rb.cache_ok(calls, 3496, 'R584')
    calls = cache_calls('R584')
    calls[0]['cpu_layer_ids'] = [7]
    assert not rb.cache_ok(calls, 3496, 'R584')
    assert not rb.cache_ok([c for c in cache_calls('R584') if c['source'] != 'init'], 3496, 'R584')
    log = synthetic_log('R584', p)
    twice = log + log.splitlines()[-1] + '\n'
    assert not rb.pre_gates('R584', rb.facts_of(twice), GEOMETRY['R584'], 'snap', cache_calls('R584'), p,
                            rb.CONFIG)['shared_bundle']
    caps = log.replace('caps [2, 2, 2, 2, 2, 3, 3]', 'caps [2, 2, 2, 2, 2, 2, 2]')
    assert not rb.pre_gates('R584', rb.facts_of(caps), GEOMETRY['R584'], 'snap', cache_calls('R584'), p,
                            rb.CONFIG)['section_caps']
    assert not rb.pre_gates('R584', rb.facts_of(log), GEOMETRY['R584'], 'other-model', cache_calls('R584'), p,
                            rb.CONFIG)['right_model']


def summary_line(uid: int, steps: int, **over) -> str:
    s = dict(copied_bytes=10, published=5, misses_from_stage=1, misses_from_host=2, expired_calls=0,
             host_misses_while_pending=0, evict_nonfinite_tokens=0)
    s.update(over)
    return f'expert predictor: request {uid}, {steps} decode steps: {s!r}'


def test_predictor_health():
    rows = [rb.PromptRow(number=n, group='g', decode=8, prompt='p') for n in range(3)]
    good = '\n'.join(summary_line(u, 7) for u in range(3))
    assert rb.predictor_health('R584', good, rows)['passed']
    assert not rb.predictor_health('R584', '\n'.join(summary_line(u, 7) for u in range(2)), rows)['passed']
    assert not rb.predictor_health('R584', good.replace('7 decode', '6 decode', 1), rows)['passed']
    assert not rb.predictor_health('R584', '\n'.join(summary_line(0, 7) for _ in range(3)), rows)['passed']
    bad = '\n'.join(summary_line(u, 7, evict_nonfinite_tokens=1 if u == 1 else 0) for u in range(3))
    assert not rb.predictor_health('R584', bad, rows)['passed']
    bad = '\n'.join(summary_line(u, 7, published=True) for u in range(3))
    assert not rb.predictor_health('R584', bad, rows)['passed']
    assert rb.predictor_health('F0Q', 'nothing', rows)['passed']
    assert not rb.predictor_health('F0Q', good, rows)['passed']
    assert not rb.predictor_health('F0Q', 'striped demand: request 3, 7 decode steps, misses per layer: [1]', rows)['passed']


# ---- gates on the real RTX 4080 logs ---------------------------------------------------------------------------------

def study_paths(serve: dict) -> rb.LocalPaths:
    args = serve['log']['server_args']
    pred = args['expert_predictor'].strip("'") if args['expert_predictor'] != 'None' else '/none'
    return rb.LocalPaths.model_construct(python=Path('/py'), cuda_home=Path('/cuda'),
                                         model=Path(args['model_path'].strip("'")), predictor=Path(pred))


@needs_study
@pytest.mark.parametrize('name', ['F0Q_1', 'R584_1'])
def test_pre_gates_pass_on_study_logs(name, monkeypatch):
    rec = json.loads((EXP609 / 'pass1.json').read_text())
    serve = rec['serves'][name]
    arm = serve['arm']
    paths = study_paths(serve)
    if arm == 'R584':   # the study kept the bundle in exp585/; the replication ships the same file
        monkeypatch.setattr(rb, 'BUNDLE', Path(serve['log']['shared_loads'][0][3]))
        paths = paths.model_copy(update=dict(predictor=Path(serve['log']['server_args']['expert_predictor'].strip("'"))))
    folder = EXP609 / f'serve_{name}'
    text = (folder / 'server.log').read_text(errors='replace')
    facts = rb.facts_of(text)
    gates = rb.pre_gates(arm, facts, serve['geometry'], serve['model_id'], rb.cache_records(folder / 'cache_state'),
                         paths, rb.CONFIG)
    assert all(gates.values()), [k for k, v in gates.items() if not v]
    assert rb.device_use(facts) == serve['device_use_bytes']
    rows = [rb.PromptRow(number=n, group='g', decode=256, prompt='p') for n in range(81)]
    assert rb.predictor_health(arm, text, rows)['passed']


@needs_study
def test_ms_per_token_matches_study_record():
    rec = json.loads((EXP609 / 'pass1.json').read_text())
    serve = rec['serves']['R584_2']
    got = [rb.ms_per_token(json.loads((EXP609 / 'serve_R584_2' / f'request_{n:02d}.json').read_text()))
           for n in range(1, 81)]
    assert got == serve['describe']['ms_per_token_by_prompt']


# ---- report statistics against the study ---------------------------------------------------------------------------

@needs_study
@pytest.mark.parametrize('group', ['math', 'qa'])
def test_report_reproduces_exp609(group):
    rec = json.loads((EXP609 / 'pass1.json').read_text())
    est = json.loads((STUDY / 'exp609_estimate_a1.json').read_text())
    rows = np.asarray(est['estimator']['groups'][group]) - 1   # 1-based row numbers inside the 80-row lists
    table = {n: np.asarray(rec['serves'][n]['describe']['ms_per_token_by_prompt'])[rows]
             for pair in report.BLOCKS for n in pair}
    got = report.contrast_ms(table, report.SEEDS[group])
    want = est['groups'][group]
    assert got['pooled_ms'] == pytest.approx(want['pooled_ms'], abs=1e-12)
    assert got['pooled_ci_99'] == pytest.approx(want['pooled_ci'], abs=1e-12)
    assert got['verdict'] == want['verdict'] and got['label'] == want['label']
    assert got['half_width_ms'] == pytest.approx(want['serve_level']['half_width_ms'], abs=1e-12)
    ref = json.loads(report.REFERENCE.read_text())['groups'][group]['rate']
    rate = report.contrast_rate(table, report.SEEDS[group])
    assert round(rate['x_tok_s'], 2) == ref['x_tok_s'] and round(rate['y_tok_s'], 2) == ref['y_tok_s']
    assert [round(v, 1) for v in rate['gain_pct_ci_99']] == ref['gain_pct_ci_99']
    assert rate['prompts_x_faster'] == ref['prompts_x_faster']


def test_verdict_rules():
    assert report.verdict([-2, -1], [-1, -1, -1]) == 'faster'
    assert report.verdict([-2, -1], [-1, -1, 0.1]) == 'unresolved'   # one block disagrees
    assert report.verdict([-2, 0.1], [-1, -1, -1]) == 'unresolved'
    assert report.verdict([1, 2], [1, 1, 1]) == 'slower'


def test_label_needs_serve_level():
    rng = np.random.default_rng(0)
    base = rng.uniform(13, 15, 20)
    # a small, consistent per-prompt gain, but large serve-to-serve drift within each arm
    table = {'F0Q_1': base, 'F0Q_2': base + 1.0, 'F0Q_3': base + 2.0,
             'R584_1': base - 0.05, 'R584_2': base + 0.95, 'R584_3': base + 1.95}
    out = report.contrast_ms(table, 605)
    assert out['verdict'] == 'faster' and out['label'] == 'unresolved'


# ---- report on synthetic runs ----------------------------------------------------------------------------------------

def fake_run(tmp: Path, state: str) -> Path:
    run = tmp / 'run'
    run.mkdir()
    info = dict(run='t', start_wall='s', end_wall='e', engine_commit=rb.ENGINE_COMMIT, state=state,
                config=dict(passes=['math_qa'], max_scored_rows=None), smoke=False,
                hardware=dict(gpus=[dict(name='NVIDIA GeForce RTX 5090', memory_total_mib='32607', pcie_gen_max='5',
                                         pcie_width_max='16', driver='580')], cpu='cpu', mem_total_bytes=64 << 30,
                              kernel='k', os='Ubuntu', wsl=False),
                software=dict(torch='2.14.1', cuda='13.0', triton='3', capability=[12, 0]))
    if state == 'C':
        info['failure'] = 'GPU 0 is X'
        (run / 'run.json').write_text(json.dumps(info))
        return run
    (run / 'run.json').write_text(json.dumps(info))
    pass_dir = run / 'math_qa'
    pass_dir.mkdir()
    rows = list(range(1, 9))
    rng = np.random.default_rng(1)
    serves = {}
    for name, _ in rb.CONFIG['serves']:
        folder = pass_dir / f'serve_{name}'
        folder.mkdir()
        speed = 0.0125 if 'R584' in name else 0.0140
        by_prompt = {}
        for n in [0] + rows:
            gaps = speed * (1 + 0.05 * rng.random(255))
            stamps = list(np.concatenate([[100.0], 100.0 + np.cumsum(gaps)]))
            q = request(n, 256, stamps)
            (folder / f'request_{n:02d}.json').write_text(json.dumps(q))
            if n:
                by_prompt[str(n)] = rb.ms_per_token(q)
        serves[name] = dict(passed=True, ms_per_token_by_prompt=by_prompt, device_use_bytes=14 << 30,
                            ms_per_token_mean=float(np.mean(list(by_prompt.values()))))
    record = dict(status='complete', rows=[0] + rows, decode={str(n): 256 for n in [0] + rows},
                  groups=dict(math=rows[0::2], qa=rows[1::2]), serves=serves, cross_gates={'x': True},
                  equal_memory=dict(difference_slots=0.5))
    (pass_dir / 'record.json').write_text(json.dumps(record))
    return run


def test_report_state_a(tmp_path):
    page = report.build(fake_run(tmp_path, 'A'))
    text = page.read_text()
    assert 'End state A' in text and 'RTX 5090' in text and 'What we can conclude' in text
    s = json.loads((page.parent / 'summary.json').read_text())
    assert s['groups']['math']['rate']['x_tok_s'] > s['groups']['math']['rate']['y_tok_s']


def test_report_rejects_edited_request(tmp_path):
    run = fake_run(tmp_path, 'A')
    path = run / 'math_qa' / 'serve_R584_1' / 'request_03.json'
    q = json.loads(path.read_text())
    q['stamps_raw'][-1] += 0.5
    path.write_text(json.dumps(q))
    with pytest.raises(report.ReportError):
        report.build(run)


def test_report_state_c(tmp_path):
    text = report.build(fake_run(tmp_path, 'C')).read_text()
    assert 'End state C' in text and 'GPU 0 is X' in text and 'Decode speed</h2>' not in text


# ---- server process group --------------------------------------------------------------------------------------------

def test_server_stop_reaches_children(tmp_path):
    # a parent that ignores SIGTERM and a child: stop() must SIGKILL the whole group and leave nothing behind
    cmd = [sys.executable, '-c', 'import signal, subprocess, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
           'subprocess.Popen(["sleep", "300"]); time.sleep(300)']
    s = rb.Server(cmd, dict(os.environ), tmp_path / 'log', tmp_path)
    time.sleep(1.0)
    out = s.stop(grace_s=1)
    assert out['sigkill'] and out['clean']
    with pytest.raises(ProcessLookupError):
        os.killpg(s.pgid, 0)


def test_main_state_c_without_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, 'HERE', tmp_path)   # no local_paths.json there
    cfg = dict(rb.CONFIG, results=tmp_path / 'results', run_name='c')
    assert rb.main(cfg) == rb.EXIT['C']
    info = json.loads((tmp_path / 'results' / 'c' / 'run.json').read_text())
    assert info['state'] == 'C' and 'setup.sh' in info['failure']
    assert (tmp_path / 'results' / 'c' / 'report.html').is_file()


def test_no_subprocess_kill_of_foreign_pids():
    # The script must never signal a process it did not start: only Server.stop calls killpg, on its own group.
    src = (HERE / 'run_bench.py').read_text()
    assert src.count('os.killpg(') == 3 and 'os.kill(' not in src and 'pkill' not in src and 'fuser' not in src


# ---- request deadline (the exp609_port_bench client) -------------------------------------------------------------------

def sse_server(delays: list[float]):
    """A local HTTP server that streams one SSE piece per delay (seconds before that piece), then usage and [DONE]."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Connection', 'close')
            self.end_headers()
            try:
                for i, d in enumerate(delays):
                    time.sleep(d)
                    msg = dict(choices=[dict(delta=dict(content=f'p{i} '), finish_reason='length' if i == len(delays) - 1 else None)])
                    self.wfile.write(b'data: ' + json.dumps(msg).encode() + b'\n\n')
                    self.wfile.flush()
                self.wfile.write(b'data: ' + json.dumps(dict(choices=[], usage=dict(completion_tokens=len(delays)))).encode()
                                 + b'\n\ndata: [DONE]\n\n')
                self.wfile.flush()
            except OSError:
                pass   # the client shut the socket at its deadline

    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_stream_completes_inside_deadline():
    srv = sse_server([0.01] * 4)
    try:
        client: dict = {}
        q = rb.stream(srv.server_address[1], 'm', 'hi', 4, dict(rb.CONFIG, request_timeout_s=5, read_timeout_s=5), client)
    finally:
        srv.shutdown()
    assert q['events'] == 4 and q['completion_tokens'] == 4 and all(s < client['hard_raw'] for s in q['stamps_raw'])
    assert client['kind'] is None and not client['timer_fired']


def test_stream_rejects_pieces_after_deadline():
    # the server is still streaming at the 1 s cutoff: the request must fail, never return late stamps
    srv = sse_server([0.05, 0.05, 1.5, 0.05])
    try:
        client: dict = {}
        with pytest.raises(rb.Stop) as e:
            rb.stream(srv.server_address[1], 'm', 'hi', 4, dict(rb.CONFIG, request_timeout_s=1, read_timeout_s=30), client)
    finally:
        srv.shutdown()
    assert e.value.state == 'B' and 'deadline' in str(e.value) and client['kind'] == 'deadline'
    assert client['raw_at_end'] >= client['hard_raw'] or client['timer_fired']
    assert client['raw_at_end'] - client['hard_raw'] < 1.0   # the cutoff ends the wait; the read timeout does not


def test_stream_stall_is_not_a_deadline():
    srv = sse_server([0.05, 2.0])
    try:
        client: dict = {}
        with pytest.raises(rb.Stop) as e:
            rb.stream(srv.server_address[1], 'm', 'hi', 2, dict(rb.CONFIG, request_timeout_s=30, read_timeout_s=0.5), client)
    finally:
        srv.shutdown()
    assert 'broke' in str(e.value) and client['kind'] == 'stream_broke' and client['read_error']


# ---- GPU watch ----------------------------------------------------------------------------------------------------------

def test_gpu_watch_own_and_foreign(monkeypatch):
    me = os.getpid()
    monkeypatch.setattr(rb, 'compute_apps', lambda: [f'{me}, python, 100 MiB'])
    own = rb.GpuWatch(5.0, os.getpgid(me), pcie=False)
    own.sample()
    assert own.summary()['own_seen'] == 1 and not own.summary()['foreign']
    other = rb.GpuWatch(5.0, -1, pcie=False)   # the same process seen from another group's server
    other.sample()
    assert other.summary()['foreign'] and other.summary()['own_seen'] == 0


def test_gpu_watch_counts_a_vanished_pid_as_foreign(monkeypatch):
    monkeypatch.setattr(rb, 'compute_apps', lambda: ['999999999, gone, 1 MiB'])
    w = rb.GpuWatch(5.0, os.getpgid(os.getpid()), pcie=False)
    w.sample()
    assert w.summary()['foreign'] and w.summary()['foreign'][0]['pgid'] is None



def test_gpu_watch_failed_query_is_not_an_empty_list(monkeypatch):
    def broken():
        rb.need(False, 'nvidia-smi failed: no devices')   # what compute_apps raises on a nonzero exit
    monkeypatch.setattr(rb, 'compute_apps', broken)
    w = rb.GpuWatch(60.0, os.getpgid(os.getpid()), pcie=False)
    with w:
        pass
    got = w.summary()
    assert got['errors'] == 1 and got['samples'] == 0 and got['empty'] == 0 and not got['foreign']
    assert got['error_types'][0].startswith('Stop') and got['ok'] is False


def test_gpu_watch_empty_list_counts_and_covers(monkeypatch):
    monkeypatch.setattr(rb, 'compute_apps', lambda: [])
    w = rb.GpuWatch(60.0, os.getpgid(os.getpid()), pcie=False)   # period longer than the test: the start sample counts
    with w:
        pass
    got = w.summary()
    assert got == dict(samples=1, empty=1, errors=0, error_types=[], joined=True, own_seen=0, foreign=[], ok=True)


def test_gpu_watch_unexpected_exception_is_counted(monkeypatch):
    monkeypatch.setattr(rb, 'compute_apps', lambda: ['not-a-pid, x, 1 MiB'])   # int() raises ValueError
    w = rb.GpuWatch(60.0, os.getpgid(os.getpid()), pcie=False)
    with w:
        pass
    assert w.summary()['errors'] == 1 and w.summary()['ok'] is False and w.summary()['joined'] is True


def test_run_serve_stops_the_watch_before_the_server():
    # A sample taken while the server exits could call the server's own vanished process foreign.
    src = inspect.getsource(rb.run_serve)
    final = src[src.index('        finally:\n            # The watch stops first'):]
    assert final.index('watch.__exit__()') < final.index('server.stop(')


# ---- input evidence -----------------------------------------------------------------------------------------------------

def good_evidence() -> dict:
    return dict(engine_equals_commit=True,
                import_origins=[str(rb.REPO / 'python/freetoken/__init__.py'),
                                str(rb.REPO / 'commitmoe_prefetch/commitmoe_prefetch/freetoken.py'),
                                str(rb.HERE / 'cache_hook/sitecustomize.py'), 'False'],
                predictor=dict(rb.PREDICTOR_RELEASE), bundle=rb.BUNDLE_SHA256,
                prompts={p: s['sha256'] for p, s in rb.PASSES.items()}, run_code={'run_bench.py': 'x'})


@pytest.mark.parametrize('edit', ['engine', 'config', 'loader', 'missing', 'origin', 'usercustomize', 'bundle', 'prompt'])
def test_pin_problems(edit):
    ev = good_evidence()
    assert rb.pin_problems(ev) == []
    if edit == 'engine':
        ev['engine_equals_commit'] = False
    elif edit == 'config':
        ev['predictor']['config.json'] = '0' * 64
    elif edit == 'loader':
        ev['predictor']['bank_loader.py'] = '0' * 64
    elif edit == 'missing':
        ev['predictor']['spec.py'] = None
    elif edit == 'origin':
        ev['import_origins'][0] = '/elsewhere/freetoken/__init__.py'
    elif edit == 'usercustomize':
        ev['import_origins'][3] = 'True'
    elif edit == 'bundle':
        ev['bundle'] = '0' * 64
    else:
        ev['prompts']['code'] = '0' * 64
    assert len(rb.pin_problems(ev)) == 1


def test_check_evidence_stops_on_a_change(monkeypatch):
    snap = good_evidence()
    changed = dict(snap, run_code={'run_bench.py': 'y'})
    monkeypatch.setattr(rb, 'evidence', lambda paths, cfg: changed)
    with pytest.raises(rb.Stop) as e:
        rb.check_evidence(snap, None, rb.CONFIG)
    assert e.value.state == 'B' and 'run_code' in str(e.value)
    monkeypatch.setattr(rb, 'evidence', lambda paths, cfg: dict(snap))
    rb.check_evidence(snap, None, rb.CONFIG)


def test_setup_pins_equal_run_bench_pins():
    text = (HERE / 'setup.sh').read_text()
    block = text.split('PREDICTOR_SHA256="', 1)[1].split('"', 1)[0]
    table = {name: h for h, name in (line.split() for line in block.strip().splitlines())}
    assert table == rb.PREDICTOR_RELEASE
    for spec in rb.PASSES.values():
        assert f"{spec['sha256']}  prompts/{spec['prompts'].name}" in text
    assert f'{rb.BUNDLE_SHA256}  assets/{rb.BUNDLE.name}' in text and f'ENGINE_COMMIT="{rb.ENGINE_COMMIT}"' in text


# ---- failed pass: no timing in the report -------------------------------------------------------------------------------

def test_report_hides_timings_of_a_failed_pass(tmp_path):
    run = fake_run(tmp_path, 'B')
    path = run / 'math_qa' / 'record.json'
    record = json.loads(path.read_text())
    record.update(status='failed', error='R584_3: misses per row differ from warmup_R584')
    path.write_text(json.dumps(record))
    page = report.build(run)
    a = json.loads((page.parent / 'summary.json').read_text())
    assert a['groups'] == {} and all(s['mean_ms'] is None for s in a['passes']['math_qa']['serves'].values())
    gates = page.read_text().split('<h2>Gates</h2>')[1].split('</table>')[0]
    assert gates.count('<td>–</td></tr>') == len(rb.CONFIG['serves'])   # every mean cell is empty


@pytest.mark.parametrize('errors', [0, 1])
def test_report_claims_hidden_process_ids_only_after_clean_queries(tmp_path, errors):
    run = fake_run(tmp_path, 'A')
    rec_path = run / 'math_qa' / 'record.json'
    rec = json.loads(rec_path.read_text())
    for s in rec['serves'].values():
        s['gpu_watch'] = dict(samples=4, empty=4 - errors, errors=errors, error_types=[], joined=True, own_seen=0,
                              foreign=[], ok=errors == 0)
    rec_path.write_text(json.dumps(rec))
    text = report.build(run).read_text()
    assert f'{errors * len(rec["serves"])} failed queries' in text
    assert ('listed no process, not even the server' in text) == (errors == 0)
