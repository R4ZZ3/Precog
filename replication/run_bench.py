"""Replication bench: FreeToken with no predictor (F0Q) against the full CommitMoE method (R584), on one GPU.

This script repeats the decode-speed tests of the CommitMoE study (RTX 4080, 16 GB) on another GPU, by default an RTX
5090, with the same expert-cache slots, KV pages and prompts. It runs two passes:

  code     40 held-out coding prompts, 1024 generated tokens each (as EXP605 of the study).
  math_qa  40 MATH-500 problems (odd rows) and 40 NQ-open questions (even rows), 256 tokens each (as EXP609).

Each pass starts seven servers, one after the other, in this registered order:
  warmup_R584, F0Q_1, R584_1, R584_2, F0Q_2, F0Q_3, R584_3.
The warm-up serve is not timed. The blocks are the adjacent pairs (F0Q_1, R584_1), (R584_2, F0Q_2), (F0Q_3, R584_3).
Every serve sends row 0 (an unscored warm-up prompt), then the scored rows in order, then one unscored 16-token request
that flushes the predictor summary of the last row.

Arms (the flags are those of the study; only the slot count and the predictor switches differ):
  F0Q   FreeToken offload, striped demand copies, bounded load, pre-touched host banks, no predictor, 3571 expert slots.
  R584  the same engine with the published expert predictor (copy-early), copy overlap and the learned shared eviction
        bundle (keep-for-later), 3496 expert slots. On the RTX 4080, the 75 fewer slots paid for the predictor's device
        memory, so both arms used the same GPU memory. The slot counts stay the same here on purpose: the question is
        how the same configuration behaves on another GPU, not what the best configuration of that GPU is.

Gates. The first failed gate stops the run at once (end state B). No gate is relaxed and nothing is retried.
  - inputs, at preflight against the pins (end state C), and before every serve and at every pass end against the
    preflight snapshot: the engine code equals the tested commit, the engine and the cache hook import from this
    checkout, every predictor release file, the R584 bundle, the prompts and this run's code are unchanged.
  - during a serve: no compute process outside the server's process group appears on the GPU (sampled every
    gpu_watch_s; checked when the serve ends), and the watch itself covered the serve: at least one sample, no failed
    query, and its thread ended. Every request ends before its RAW-clock deadline.
  - before the first request of a serve: the right model, at least 0.4 GiB GPU memory free after init and after CUDA
    graph capture, graphs on, page size 1, 8284 KV pages, the registered slot count in the live engine and in the log,
    host embeddings, the ServerArgs line equal to the registered flags, no error line, the predictor loaded exactly once
    in R584 and never in F0Q, the R584 section caps and eviction bundle, none of the study's diagnostic switches, and the
    engine's resolved cache (one GPU slot pool, no CPU layer) with the arm's hook class, read from the live objects.
  - after a serve: every request in order, exactly `decode` completion tokens, finish reason 'length', 2 to `decode`
    streamed pieces with non-decreasing stamps, no error line in the server log, the predictor summaries complete
    (R584) or absent (F0Q), the tail request complete, and the server process group gone.
  - across serves: every serve streams the same pieces as the warm-up (greedy decoding, so any difference means the
    output changed); every R584 serve has the same misses per row as the R584 warm-up; and the mean device memory used by
    the timed F0Q serves minus that of the timed R584 serves lies in [-5, +2] slots of 3146112 bytes (equal GPU memory).

End states and exit codes:
  A (0)  both passes complete with every gate passed. report.html holds the result.
  B (2)  a gate failed or the run stopped. failure details are in run.json and the report.
  C (3)  the run did not start: a preflight check failed. Nothing was timed.
  130    interrupted with Ctrl-C.

Outputs (under replication/results/<run name>/): run.json (hardware, software, state), <pass>/record.json (all gates and
per-prompt values), <pass>/serve_<name>/ (server.log, command.json, request_NN.json), and report.html (from report.py).

Usage, from the repository root, after `bash replication/setup.sh`:
    .venv/bin/python replication/run_bench.py
Edit CONFIG below to change the expected GPU name or to run a short smoke (max_scored_rows).

This script never stops a process it did not start. If another process uses the GPU, it refuses to start (state C) or
stops (state B), and leaves that process alone.
"""
from __future__ import annotations

import ast
import hashlib
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# ---- configuration (edit here; the script takes no command-line options) ---------------------------------------------
SLOT_BYTES = 3146112   # one FP8 expert of Qwen3.5-35B-A3B in the engine's slot layout
CONFIG = dict(
    # A substring of the nvidia-smi name of GPU 0. The run refuses another GPU, so a report never mislabels its hardware.
    expect_gpu='RTX 5090',
    passes=('code', 'math_qa'),
    # None runs every scored row. An integer runs only the first rows of each group (a smoke; its report says so).
    max_scored_rows=None,
    serves=(('warmup_R584', 'R584'), ('F0Q_1', 'F0Q'), ('R584_1', 'R584'), ('R584_2', 'R584'), ('F0Q_2', 'F0Q'),
            ('F0Q_3', 'F0Q'), ('R584_3', 'R584')),
    slots={'F0Q': 3571, 'R584': 3496},   # the frozen values of the RTX 4080 runs (EXP605 and EXP609)
    pages=8284, page_size=1, memory_ratio='0.9', pin_budget_gb='32', stripes=16,
    max_seq_len=8192 + 256,   # 8448, as in every study serve (the base was the 256-token default row)
    section_caps=[2, 2, 2, 2, 2, 3, 3],   # R584 copy budget per decode step by layer section; the sum is 16
    tail_decode=16,
    min_free_gib=0.4,   # GPU memory free after init and after graph capture (study value)
    # Device use of the R584 serves on the RTX 4080 was 13.72 GiB (EXP593). With 0.1 GiB margin and the 0.4 GiB floor,
    # a serve needs about 14.22 GiB = 14562 MiB free on the idle GPU.
    min_gpu_free_mib=14562,
    # Host memory the server pins or holds at load: 40 x 256 expert banks, the host embedding tables (0.95 GiB), the
    # bounded reader's transient upper bound (two peaks, from the checkpoint headers) and 6 GB for the process, the
    # CUDA context and page cache. About 46.6 GB of MemAvailable.
    host_need_bytes=40 * 256 * SLOT_BYTES + int(0.95 * 1024 ** 3) + 3645493248 + 3759603840 + 6_000_000_000,
    min_disk_free_bytes=2_000_000_000,   # about 60 MB of logs and request files per serve, 14 serves
    device_band_bytes=(-5 * SLOT_BYTES, 2 * SLOT_BYTES),   # mean F0Q minus mean R584 device use (study band)
    startup_timeout_s=1800,   # the RTX 4080 took about 210 s from start to ready; a slow disk can take much longer
    request_timeout_s=900,    # RAW-clock deadline per request; one 1024-token request took about 15 s on the RTX 4080
    read_timeout_s=180,       # longest silence on the socket before the request counts as stalled (also capped by
                              # the time left to the request deadline)
    stop_grace_s=30,          # SIGTERM, then SIGKILL after this many seconds (own process group only)
    vram_release_s=120,       # wait for the previous server's GPU memory to return before the next start
    poll_s=2.0,
    # GPU samples from server start to server stop: compute processes (every serve) and the PCIe link (first warm-up
    # serve only). 5 s gives several samples in the shortest serve (about 2 minutes) and costs one nvidia-smi call, the
    # same in both arms.
    gpu_watch_s=5.0,
    results=HERE / 'results',
    run_name=None,            # None: the start time, e.g. 20261108-1530
)
PASSES = {
    'code': dict(prompts=HERE / 'prompts' / 'code40.jsonl',
                 sha256='7c0225333c0e4b7a5d5b808711770f9b81bf49e0f34c30127969b273ddce9af6', groups=('code',)),
    'math_qa': dict(prompts=HERE / 'prompts' / 'math_qa80.jsonl',
                    sha256='3ff3d8bb381ab9f3556d814f69df3f81c2db0efd2468164dd3a1d7bf60eb9084', groups=('math', 'qa')),
}
ENGINE_COMMIT = '16d7cd485d68599972e3b9c2ae6798c32d72cb49'
ENGINE_PATHS = ('python', 'commitmoe_prefetch', 'setup.py', 'pyproject.toml')
BUNDLE = HERE / 'assets' / 'r584_shared_evict_v1.pt'
BUNDLE_SHA256 = '87421c2054eca01450ba34b88e7fad5e8c5ff206a8ac5616938f6b6df59344f7'
# Every file of the predictor release at its pinned revision. setup.sh holds the same table (a test keeps them equal):
# the server reads config.json and the loader code as well as the weights, so all of them are pinned.
PREDICTOR_RELEASE = {
    'LICENSE': '6d1d968fb225eca367cb7f0b8831ab012a35d92b547e945e17ef8e7b05c3e5cc',
    'NOTICE': '862f73e7a0d143e22e853e58de31bf772d9b51dfb4a7b71da2bff3a28a09440a',
    'README.md': '54fc0b913f44bd6a19554e956254850071cc38795d68735934d8ec98f818db1a',
    'bank_loader.py': '198b03800f11c9d04b6e10e01a810b58777d0045c13eccd17a14e8d825247193',
    'config.json': 'bd0a3ea3b6c8d9a1099ce2f42543e8831a7dade8137d5b3216ed7aa75e2a791b',
    'manifest.json': 'a794508507c48884a3c6bc008eb5a51e963e50b61d5b8e3d462322bb706f4718',
    'model.safetensors': '32c631a276fcf1d34cb079e253c371c3745f05f59ea3c217c855f85fd7cccecf',
    'parity_fixture.safetensors': '09718946bf69e8f3c77b5af45b4ad664c1ccc631435a14c1c6858a903e1832d1',
    'reference_scorer.py': '60e60a1706cf14260dbfb962081c218cb87c51d329af3e61422583d23737d2f8',
    'spec.py': '8e65f523884fccba36fa7f7aad643675badaff570bea8a3b8d5ad7b9512a9eed',
}
# The code of this run. Its hashes go into the evidence, so an edit during the run stops it.
RUN_CODE = {'run_bench.py': HERE / 'run_bench.py', 'report.py': HERE / 'report.py',
            'sitecustomize.py': HERE / 'cache_hook' / 'sitecustomize.py'}
PREDICTOR_ARMS = ('R584',)
# What each arm's engine holds at the return of Engine._init_offload_moe_cache (cache_hook/sitecustomize.py).
IDENTITY = {'F0Q': dict(hooks_type='StripedDemandHooks', learned=False, overlap=False),
            'R584': dict(hooks_type='PrefetchHooks', learned=True, overlap=True)}
CACHE_VAR = 'REPLICATION_CACHE_DIR'
OFFLOAD_ARGS = dict(moe_collect_stats='False', embed_weights="'host'", cache_type="'naive'", num_page_override='8284',
                    memory_ratio='0.9', moe_strategy="'offload'", moe_cache_scope="'unified'", max_running_req='1',
                    cuda_graph_max_bs='1', max_seq_len_override='8448', moe_hybrid_max_fetch='-1',
                    expert_load="'bounded'", pretouch_host_banks='True', moe_demand_copy_stripes='16')
LOG_ERRORS = re.compile(r'Traceback \(most recent call last\)|CUDA error|cudaError|illegal memory access|\bNaN\b'
                        r'|non-finite|\bnonfinite\b')
SUMMARY = re.compile(r'expert predictor: request (-?\d+), (\d+) decode steps: (\{.*\})')
MISS_LINE = re.compile(r'striped demand: request (-?\d+), (\d+) decode steps, misses per layer:')
SUMMARY_COUNTERS = ('copied_bytes', 'published', 'misses_from_stage', 'misses_from_host', 'expired_calls',
                    'host_misses_while_pending')
EXIT = dict(A=0, B=2, C=3)


class Stop(Exception):
    """A failed check. state 'C' before anything was timed (preflight), 'B' after."""

    def __init__(self, state: str, what: str):
        super().__init__(what)
        self.state = state


def need(ok: bool, what: str, state: str = 'B') -> None:
    if not ok:
        raise Stop(state, what)


class LocalPaths(BaseModel):
    """replication/local_paths.json, written by setup.sh."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    python: Path
    cuda_home: Path
    model: Path
    predictor: Path

    @field_validator('python', 'cuda_home', 'model', 'predictor')
    @classmethod
    def exists(cls, p: Path) -> Path:
        if not p.exists():
            raise ValueError(f'{p} does not exist; run replication/setup.sh again')
        return p


class PromptRow(BaseModel):
    model_config = ConfigDict(extra='ignore', frozen=True)
    number: int
    group: str
    decode: int
    prompt: str

    @field_validator('decode')
    @classmethod
    def at_least_two(cls, v: int) -> int:
        if v < 2:
            raise ValueError('decode must be at least 2: the timing span needs two stamps')
        return v


# ---- small helpers ---------------------------------------------------------------------------------------------------

def raw() -> float:
    """CLOCK_MONOTONIC_RAW: not slewed by NTP, so a time step during a request cannot change its timing."""
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def wall() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S %z')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str], timeout: float = 60, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str) + '\n')


def mem_available() -> int:
    m = re.search(r'^MemAvailable:\s+(\d+) kB$', Path('/proc/meminfo').read_text(), re.M)
    need(m is not None, 'MemAvailable missing from /proc/meminfo', 'C')
    return int(m.group(1)) * 1024


def nvsmi(fields: str) -> list[list[str]]:
    """nvidia-smi --query-gpu rows of GPU 0..n, as lists of stripped strings."""
    out = run(['nvidia-smi', f'--query-gpu={fields}', '--format=csv,noheader,nounits'])
    need(out.returncode == 0, f'nvidia-smi failed: {out.stderr.strip()[:200]}')
    return [[v.strip() for v in line.split(',')] for line in out.stdout.strip().splitlines()]


def compute_apps() -> list[str]:
    """Compute processes on any GPU. Under WSL2 this list is always empty (the driver hides it); see the report."""
    out = run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'])
    need(out.returncode == 0, f'nvidia-smi failed: {out.stderr.strip()[:200]}')
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def gpu_free_mib() -> int:
    return int(nvsmi('memory.free')[0][0])


def free_port() -> int:
    """A free HTTP port whose next port is also free: the engine binds port + 1 for its rendezvous."""
    for _ in range(100):
        with socket.socket() as http_sock:
            http_sock.bind(('127.0.0.1', 0))
            port = http_sock.getsockname()[1]
            if port >= 65535:
                continue
            try:
                with socket.socket() as rendezvous:
                    rendezvous.bind(('127.0.0.1', port + 1))
                    return port
            except OSError:
                continue
    raise Stop('B', 'no pair of adjacent free ports')


def get_json(port: int, path: str, timeout: float = 10) -> dict:
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        conn.request('GET', path)
        resp = conn.getresponse()
        body = resp.read()
        if resp.status != 200:
            raise ValueError(f'{path} answered HTTP {resp.status}')
        return json.loads(body)
    finally:
        conn.close()


# ---- prompts ---------------------------------------------------------------------------------------------------------

def prompt_rows(pass_name: str, cfg: dict) -> list[PromptRow]:
    """Row 0 and the scored rows of a pass, in send order. A smoke keeps the first max_scored_rows rows of each group."""
    spec = PASSES[pass_name]
    rows = [PromptRow.model_validate_json(line) for line in Path(spec['prompts']).read_text().splitlines() if line]
    need([r.number for r in rows] == list(range(len(rows))), f'{pass_name}: rows are not numbered 0..n in order')
    need(rows[0].group == 'warmup' and all(r.group in spec['groups'] for r in rows[1:]),
         f'{pass_name}: unexpected groups')
    limit = cfg['max_scored_rows']
    if limit is None:
        return rows
    kept, seen = [rows[0]], {g: 0 for g in spec['groups']}
    for r in rows[1:]:
        if seen[r.group] < limit:
            kept.append(r)
            seen[r.group] += 1
    return kept


# ---- server process --------------------------------------------------------------------------------------------------

class Server:
    """One engine server in its own session (process group), so stop() reaches every child and nothing else."""

    def __init__(self, cmd: list[str], env: dict, log_path: Path, cwd: Path):
        self.log = open(log_path, 'wb')
        self.proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=self.log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        self.pgid = self.proc.pid

    def exited(self) -> int | None:
        return self.proc.poll()

    def group_alive(self) -> bool:
        try:
            os.killpg(self.pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:   # a member exists that we may not signal: still alive
            return True

    def stop(self, grace_s: float) -> dict:
        """SIGTERM to the group, SIGKILL after grace_s, then reap. 'clean' means no member of the group is left."""
        out = dict(exit_before_stop=self.exited(), sigkill=False)
        try:
            os.killpg(self.pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        end = time.monotonic() + grace_s
        while time.monotonic() < end and self.group_alive():
            self.exited()
            time.sleep(0.5)
        if self.group_alive():
            out['sigkill'] = True
            try:
                os.killpg(self.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            end = time.monotonic() + 10
            while time.monotonic() < end and self.group_alive():
                self.exited()
                time.sleep(0.2)
        try:
            out['exit'] = self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            out['exit'] = None
        self.log.close()
        out['clean'] = not self.group_alive()
        return out


def serve_cmd(arm: str, port: int, paths: LocalPaths, cfg: dict) -> list[str]:
    """The study's serve command (EXP605/EXP609 flag order). Only the slots and the predictor flag differ by arm."""
    return [str(paths.python), '-s', '-m', 'freetoken.cli', 'serve',
            '--model', str(paths.model), '--host', '127.0.0.1', '--port', str(port), '--text-model-only',
            '--cache-type', 'naive', '--memory-ratio', cfg['memory_ratio'], '--num-pages', str(cfg['pages']),
            '--expert-load', 'bounded', '--pretouch-host-banks',
            '--moe-strategy', 'offload', '--moe-cache-scope', 'unified', '--moe-cache-size', str(cfg['slots'][arm]),
            '--embed-weights', 'host', '--max-running-requests', '1', '--max-seq-len-override', str(cfg['max_seq_len']),
            '--cuda-graph-max-bs', '1', '--moe-demand-copy-stripes', str(cfg['stripes']),
            '--moe-hybrid-max-fetch', '-1', *(['--expert-predictor', str(paths.predictor)] if arm in PREDICTOR_ARMS else [])]


def serve_env(arm: str, paths: LocalPaths, cfg: dict, cache_dir: Path | None) -> dict:
    """A whitelist: no inherited allocator, numerical or thread setting and no secret reaches the server."""
    env = dict(HOME=str(Path.home()), USER=os.environ.get('USER', 'user'), LANG='C.UTF-8', TERM='dumb',
               PATH=f"{paths.python.parent}:{paths.cuda_home}/bin:/usr/bin:/bin", CUDA_HOME=str(paths.cuda_home),
               CUDA_VISIBLE_DEVICES='0', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PIP_CONFIG_FILE='/dev/null',
               PIP_DISABLE_PIP_VERSION_CHECK='1',
               PYTHONPATH=os.pathsep.join(str(p) for p in (REPO / 'python', REPO / 'commitmoe_prefetch', HERE / 'cache_hook')),
               FREETOKEN_PIN_BUDGET_GB=cfg['pin_budget_gb'])
    if 'HF_HOME' in os.environ:
        env['HF_HOME'] = os.environ['HF_HOME']
    if cache_dir is not None:
        env[CACHE_VAR] = str(cache_dir)
    if arm in PREDICTOR_ARMS:
        env.update(FREETOKEN_OVERLAP_RESIDENT='1', FREETOKEN_LEARNED_EVICT_START='1',
                   FREETOKEN_SHARED_EVICT=str(BUNDLE), FREETOKEN_SHARED_EVICT_FAST='blockn')
    return env


# ---- streaming -------------------------------------------------------------------------------------------------------

def stream(port: int, model_id: str, prompt: str, decode: int, cfg: dict, client: dict | None = None) -> dict:
    """One greedy chat request with streaming. One RAW stamp per nonempty piece (reasoning or content text). The
    request must end with usage and [DONE] before its RAW deadline (request_timeout_s after the start).

    The deadline (the client of exp609_port_bench): a timer shuts the socket at the cutoff and sets a flag, every read
    waits at most the time left (and at most read_timeout_s), and a chunk read at or after the cutoff fails the request,
    so no piece after the cutoff is ever stamped. Trap: threading.Timer waits on CLOCK_MONOTONIC and the cutoff is on
    CLOCK_MONOTONIC_RAW; the two drifted 4.5 % apart in one study serve (EXP607), so the timer is only a best-effort
    abort and the RAW checks decide. `client` (filled on every path, also on failure) is the client-side record: bytes
    read, the read error, the flag, the cutoff and the RAW time at the end."""
    client = {} if client is None else client
    body = dict(model=model_id, messages=[dict(role='user', content=prompt)], max_tokens=decode, ignore_eos=True,
                stream=True, stream_options=dict(include_usage=True), chat_template_kwargs=dict(enable_thinking=True),
                temperature=0.0)
    hard = raw() + cfg['request_timeout_s']
    client.update(hard_raw=hard, bytes_read=0, timer_fired=False, read_error=None, raw_at_end=None, kind=None)
    fired = threading.Event()
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=min(cfg['read_timeout_s'], cfg['request_timeout_s']))
    sock: socket.socket | None = None
    timer: threading.Timer | None = None

    def shut() -> None:
        fired.set()   # set before the shutdown, so a reader that sees the end of stream also sees the flag
        try:
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def late() -> bool:
        return fired.is_set() or raw() >= hard

    def deadline(where: str) -> Stop:
        client['kind'] = 'deadline'
        return Stop('B', f"request reached its {cfg['request_timeout_s']} s deadline {where} "
                         f"(timer fired: {fired.is_set()})")

    stamps, pieces, kinds, usage, finish, complete = [], [], [], None, None, False
    try:
        conn.connect()
        # Kept here: http.client drops conn.sock when a response will close the connection; the response still reads it.
        sock = conn.sock
        timer = threading.Timer(max(0.0, hard - raw()), shut)
        timer.daemon = True
        timer.start()
        conn.request('POST', '/v1/chat/completions', body=json.dumps(body).encode(),
                     headers={'Content-Type': 'application/json'})
        resp = conn.getresponse()
        if late():
            raise deadline('before the response')
        need(resp.status == 200, f'chat request answered HTTP {resp.status}')
        buf = b''
        while not complete:
            left = hard - raw()
            if left <= 0:
                raise deadline('between reads')
            sock.settimeout(min(left, cfg['read_timeout_s']))
            chunk = resp.read1(65536)
            client['bytes_read'] += len(chunk)   # every byte received, also a chunk rejected at the cutoff
            if late():
                raise deadline('during a read')
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b'\n')
            for line in lines:
                line = line.strip()
                if not line.startswith(b'data:'):
                    continue
                payload = line[5:].strip()
                if payload == b'[DONE]':
                    complete = True
                    break
                msg = json.loads(payload)
                usage = msg.get('usage') or usage
                for choice in msg.get('choices', []):
                    finish = choice.get('finish_reason') or finish
                    delta = choice.get('delta') or {}
                    reasoning, content = delta.get('reasoning_content') or '', delta.get('content') or ''
                    if reasoning or content:
                        stamps.append(raw())
                        pieces.append(reasoning + content)
                        kinds.append(('r' if reasoning else '') + ('c' if content else ''))
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
        client['read_error'] = repr(e)
        if late():
            raise deadline(f'(read error {e!r})') from e
        client['kind'] = 'stream_broke'
        raise Stop('B', f'request stream broke after {len(pieces)} pieces: {e!r}') from e
    finally:
        if timer is not None:
            timer.cancel()
        client.update(timer_fired=fired.is_set(), raw_at_end=raw())
        conn.close()
    if stamps and stamps[-1] >= hard:   # a stamp is taken just after the read check; reject even that margin
        raise deadline('at the last stamp')
    if not (complete and usage is not None):
        client['kind'] = 'incomplete'
    need(complete and usage is not None, f'incomplete stream ([DONE] {complete}, usage {usage is not None})')
    return dict(decode=decode, completion_tokens=usage.get('completion_tokens'), finish_reason=finish,
                text=''.join(pieces), pieces=pieces, piece_kinds=kinds, events=len(pieces), stamps_raw=stamps)


def stamps_ok(q: dict) -> bool:
    s = q['stamps_raw']
    return (len(s) == q['events'] and 2 <= len(s) <= q['decode'] and all(b >= a for a, b in zip(s, s[1:]))
            and s[-1] > s[0])


def ms_per_token(q: dict) -> float:
    """Milliseconds per generated token: the span from the first to the last streamed piece over decode - 1 tokens.

    Trap: this is the study's client-side proxy. The first and last pieces need not be the first and last tokens (the
    detokenizer can hold a partial character), but the text gate makes both arms stream the same pieces, so a paired
    difference compares one span. The denominator stays the token count, so the meaning does not change per prompt."""
    need(stamps_ok(q), f"request {q.get('number')}: invalid stamps")
    s = q['stamps_raw']
    return 1000 * (s[-1] - s[0]) / (q['decode'] - 1)


def pgid_of(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None


class GpuWatch:
    """Samples the GPU every period_s on a thread, from server start to server stop.

    Compute processes (every serve): a process outside the server's process group is interference, and the serve fails
    its 'no_foreign_gpu_process' gate. A process that ended between the nvidia-smi call and the group lookup also
    counts as foreign (the conservative side). run_serve stops this watch before it stops the server, so the
    server's own processes cannot vanish under a sample. Where the driver hides process ids (WSL2, many
    containers), a successful query returns an empty list and the gate sees nothing: 'empty' counts those
    queries, 'own_seen' = 0 records it, and the report states it. Inside a container that shows host process
    ids, every sample looks foreign: run on the host.

    Coverage: a failed query is not an empty list. Any exception in a sample counts in 'errors' (with its type in
    'error_types'), and the 'gpu_watch_ok' gate needs errors == 0, at least one sample, and a joined thread. The
    first sample runs at start, so a serve of any length has one.

    PCIe link (first warm-up serve only, pcie=True): generation and width. The link trains down at idle, so the
    report uses the maximum over the samples, most of which fall while requests run."""

    def __init__(self, period_s: float, own_pgid: int, pcie: bool):
        self.period_s, self.own_pgid, self.pcie = period_s, own_pgid, pcie
        self.samples, self.empty, self.own_seen, self.errors = 0, 0, 0, 0
        self.foreign: list[dict] = []
        self.links: list[dict] = []
        self.error_types: list[str] = []
        self.joined = False
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self) -> None:
        try:
            apps = compute_apps()
            if not apps:
                self.empty += 1
            for line in apps:
                pid = int(line.split(',')[0])
                if pgid_of(pid) == self.own_pgid:
                    self.own_seen += 1
                elif len(self.foreign) < 20:   # the first 20 are enough evidence; the gate needs one
                    self.foreign.append(dict(line=line, pgid=pgid_of(pid), raw=raw()))
            self.samples += 1
            if self.pcie:
                gc, gm, wc, wm = nvsmi('pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,'
                                       'pcie.link.width.max')[0]
                self.links.append(dict(gen_current=gc, gen_max=gm, width_current=wc, width_max=wm))
        except Exception as e:   # broad on purpose: an uncaught error would end the thread unseen; the gate fails
            self.errors += 1
            if len(self.error_types) < 20:
                self.error_types.append(f'{type(e).__name__}: {str(e)[:160]}')

    def _run(self) -> None:
        self.sample()   # at start, so a serve shorter than period_s still has a sample
        while not self.done.wait(self.period_s):
            self.sample()

    def __enter__(self) -> GpuWatch:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.done.set()
        self.thread.join(timeout=130)   # one sample is at most two nvidia-smi calls; run() caps each at 60 s
        self.joined = not self.thread.is_alive()

    def ok(self) -> bool:
        """The coverage rule of the 'gpu_watch_ok' gate."""
        return self.joined and self.errors == 0 and self.samples >= 1

    def summary(self) -> dict:
        return dict(samples=self.samples, empty=self.empty, errors=self.errors, error_types=self.error_types,
                    joined=self.joined, own_seen=self.own_seen, foreign=self.foreign, ok=self.ok())

    def pcie_summary(self) -> dict:
        def best(k):
            vals = [int(r[k]) for r in self.links if r[k].isdigit()]
            return max(vals) if vals else None
        return dict(samples=len(self.links), gen_current_max=best('gen_current'), gen_max=best('gen_max'),
                    width_current_max=best('width_current'), width_max=best('width_max'))


# ---- log facts and gates ---------------------------------------------------------------------------------------------

def server_args(text: str) -> dict:
    """The key=value pairs of the single ServerArgs line. Empty if there is no such line or more than one."""
    lines = re.findall(r'^ServerArgs\((.*)\)\s*$', text, re.M)
    return dict(re.findall(r"(?:^|, )(\w+)=([^,()]*)(?=, |$)", lines[0])) if len(lines) == 1 else {}


def facts_of(text: str) -> dict:
    """What the server log says, as the study's bench read it (exp587a_run.parse_log and exp609_port_bench.facts_of)."""
    def last(pattern):
        found = re.findall(pattern, text)
        return found[-1] if found else None
    init = last(r'Free memory after initialization: ([\d.]+) GiB')
    cap = last(r'Free GPU memory after capturing CUDA graphs: ([\d.]+) GiB')
    return dict(
        free_after_init_gib=float(init) if init else None,
        free_after_capture_gib=float(cap) if cap else None,
        graph_disabled='CUDA graph is disabled.' in text,
        page_adjusted=last(r'(Page size \d+ is auto-adjusted[^\n]*)'),
        kv_tokens=last(r'Allocating (\d+) tokens for KV cache'),
        embed_host=last(r'(Embedding tables in pinned host RAM \(--embed-weights host\)[^\n]*)'),
        log_errors=sorted({m.group(0) for m in LOG_ERRORS.finditer(text)}),
        server_args=server_args(text),
        configured_slots=sorted({int(x) for x in re.findall(r'moe_cache_size=(\d+), moe_cache_rate', text)}),
        predictor_loads=len(re.findall(r'expert predictor: \d+ heads L\d+-L\d+', text)),
        predictor_caps=[ast.literal_eval(v) for v in re.findall(r'expert predictor: \d+ heads .*?, caps (\[[\d, ]*\])', text)],
        # The study's diagnostic switches log a line only when set. None may appear in a replication serve.
        switch_lines=len(re.findall(r'expert predictor: (?:baseline picks|layer counters on|no prefetch copies|oracle '
                                    r'record to|oracle picks from|min score)|striped demand: miss log on|position '
                                    r'counters to', text)),
        rows_rule=bool(re.search(r'(?<!shared )learned eviction: layers', text)),
        shared_loads=re.findall(r'shared learned eviction: all (\d+) layers from layer (\d+), '
                                r'R584 ([0-9.]+) MiB \((.*?)\), fast (\S+)', text),
        memory_phases=[json.loads(m) for m in re.findall(r'MEMORY_PHASE (\{.*?\})\s*$', text, re.M)],
    )


def device_use(facts: dict) -> int | None:
    """Device bytes in use: free before load minus free after graph capture, from the engine's MEMORY_PHASE lines
    (torch.cuda.mem_get_info, device-wide). None unless there is exactly one line of each phase."""
    before = [m.get('device_free') for m in facts['memory_phases'] if m.get('phase') == 'before_load']
    after = [m.get('device_free') for m in facts['memory_phases'] if m.get('phase') == 'after_capture']
    if len(before) != 1 or len(after) != 1 or not all(type(v) is int and v >= 0 for v in before + after):
        return None
    return before[0] - after[0]


def args_differ(arm: str, args: dict, paths: LocalPaths, cfg: dict) -> dict:
    """ServerArgs values that differ from the registered flags: what the engine got, not what we sent."""
    want = dict(OFFLOAD_ARGS, moe_cache_size=str(cfg['slots'][arm]),
                expert_predictor=f"'{paths.predictor}'" if arm in PREDICTOR_ARMS else 'None')
    return {k: dict(want=v, got=args.get(k)) for k, v in want.items() if args.get(k) != v}


def cache_records(folder: Path) -> list[dict]:
    return [c for f in sorted(folder.glob('cache_*.json')) for c in json.loads(f.read_text())['calls']]


def cache_ok(calls: list[dict], slots: int, arm: str) -> bool:
    """At least one 'init' record. Every record: GPU decode, no CPU or locked layer, one pool with all slots, one bank
    layout. Every 'init' record: the arm's hook class, learned eviction and overlap switch."""
    inits = [c for c in calls if c.get('source') == 'init']
    return bool(inits) and all(
        c.get('decode_target') == 'gpu' and c.get('cpu_layer_ids') == [] and c.get('locked_hybrid') == []
        and c.get('pool_sizes') == [slots] and c.get('cache_size') == slots and c.get('distinct_layouts') == 1
        for c in calls) and all(all(c.get(k) == v for k, v in IDENTITY[arm].items()) for c in inits)


def pre_gates(arm: str, facts: dict, geometry: dict, model_id: str, calls: list[dict], paths: LocalPaths,
              cfg: dict) -> dict:
    gates = dict(
        right_model=model_id.rstrip('/') == str(paths.model).rstrip('/') or Path(model_id).name == paths.model.name,
        free_after_init=(facts['free_after_init_gib'] or -1) >= cfg['min_free_gib'],
        free_after_capture=(facts['free_after_capture_gib'] or -1) >= cfg['min_free_gib'],
        graphs_on=not facts['graph_disabled'],
        page_size_kept=facts['page_adjusted'] is None and geometry.get('page_size') == cfg['page_size'],
        kv_pages=geometry.get('num_pages') == cfg['pages'] and facts['kv_tokens'] == str(cfg['pages'] * cfg['page_size']),
        slots=geometry.get('moe_cache_size') == cfg['slots'][arm],
        logged_slots=facts['configured_slots'] == [cfg['slots'][arm]],
        embed_host=facts['embed_host'] is not None,
        server_args=not args_differ(arm, facts['server_args'], paths, cfg),
        no_log_error=not facts['log_errors'],
        predictor_loads=facts['predictor_loads'] == (1 if arm in PREDICTOR_ARMS else 0),
        no_diagnostic_switch=facts['switch_lines'] == 0,
        no_rows_rule=not facts['rows_rule'],
        memory_phases=device_use(facts) is not None,
        resolved_cache=cache_ok(calls, cfg['slots'][arm], arm),
    )
    if arm in PREDICTOR_ARMS:
        rows = facts['shared_loads']
        gates['section_caps'] = facts['predictor_caps'] == [cfg['section_caps']]
        gates['shared_bundle'] = len(rows) == 1 and (rows[0][0], rows[0][1], rows[0][3], rows[0][4]) == (
            '40', '1', str(BUNDLE), 'blockn')
    else:
        gates['no_shared_rule'] = not facts['shared_loads']
    return gates


def is_count(v) -> bool:
    return type(v) is int and v >= 0   # bool is a subclass of int; type() excludes it


def predictor_health(arm: str, text: str, rows: list[PromptRow]) -> dict:
    """R584: one complete summary per request (the tail flushes the last), decode - 1 steps, no nonfinite eviction
    score, every counter a count. F0Q: no predictor summary and no miss-log line."""
    matches = SUMMARY.findall(text)
    if arm not in PREDICTOR_ARMS:
        problems = (['predictor summaries in F0Q'] if matches else []) + (['miss lines in F0Q'] if MISS_LINE.search(text) else [])
        return dict(passed=not problems, requests=[], problems=problems)
    problems, out, uids = [], [], set()
    if len(matches) != len(rows):
        problems.append(f'{len(matches)} predictor summaries, expected {len(rows)}')
    for r, (uid, steps, body) in zip(rows, matches):
        try:
            s = ast.literal_eval(body)
            if not isinstance(s, dict) or type(s.get('evict_nonfinite_tokens')) is not int:
                raise ValueError('not a summary with an integer evict_nonfinite_tokens')
        except (ValueError, SyntaxError) as e:
            problems.append(f'row {r.number}: malformed summary ({e!r})')
            continue
        if uid in uids or int(steps) != r.decode - 1 or s['evict_nonfinite_tokens'] != 0:
            problems.append(f'row {r.number}: repeated request id, wrong decode steps or nonfinite eviction scores')
        if not all(is_count(s.get(k)) for k in SUMMARY_COUNTERS):
            problems.append(f'row {r.number}: a counter is missing or not a count')
        uids.add(uid)
        out.append(dict(number=r.number, steps=int(steps), summary=s))
    return dict(passed=not problems, requests=out, problems=problems)


# ---- one serve -------------------------------------------------------------------------------------------------------

def wait_gpu_released(cfg: dict) -> int:
    """Wait until the GPU has the room a serve needs (the previous server's memory returns after it exits)."""
    end = time.monotonic() + cfg['vram_release_s']
    free = gpu_free_mib()
    while free < cfg['min_gpu_free_mib'] and time.monotonic() < end:
        time.sleep(cfg['poll_s'])
        free = gpu_free_mib()
    return free


def run_serve(name: str, arm: str, folder: Path, rows: list[PromptRow], paths: LocalPaths, cfg: dict,
              first_cmd: dict | None, pcie: bool, snapshot: dict) -> dict:
    rec: dict = dict(arm=arm, passed=False, start_wall=wall())
    folder.mkdir()
    try:
        check_evidence(snapshot, paths, cfg)
        apps = compute_apps()
        need(not apps, f'another process uses the GPU: {apps}. This script never stops it.')
        rec['gpu_free_mib_before'] = free = wait_gpu_released(cfg)
        need(free >= cfg['min_gpu_free_mib'], f"GPU 0 has {free} MiB free, needs {cfg['min_gpu_free_mib']}")
        rec['mem_available_before'] = avail = mem_available()
        need(avail >= cfg['host_need_bytes'], f"MemAvailable {avail / 1e9:.1f} GB < {cfg['host_need_bytes'] / 1e9:.1f} GB")
        port = free_port()
        cmd, env = serve_cmd(arm, port, paths, cfg), serve_env(arm, paths, cfg, folder / 'cache_state')
        command = dict(cmd=cmd, env={k: v for k, v in env.items() if k != CACHE_VAR})
        write_json(folder / 'command.json', command)
        if first_cmd is not None:   # a repeated arm keeps its command and environment, the port aside
            def no_port(c):
                return [x for i, x in enumerate(c) if not (i and c[i - 1] == '--port')]
            need(no_port(first_cmd['cmd']) == no_port(cmd) and first_cmd['env'] == command['env'],
                 f'{name}: command or environment differs from the first {arm} serve')
        (folder / 'cache_state').mkdir()
        log_path = folder / 'server.log'
        t0 = raw()
        server = Server(cmd, env, log_path, REPO)
        watch = GpuWatch(cfg['gpu_watch_s'], server.pgid, pcie).__enter__()
        reqs: list[dict] = []
        try:
            status, end = None, raw() + cfg['startup_timeout_s']
            while raw() < end and server.exited() is None:
                try:
                    status = get_json(port, '/health').get('status')
                except (OSError, ValueError, http.client.HTTPException):
                    status = None   # not listening yet, or 503 while loading
                if status in ('ok', 'error'):
                    break
                time.sleep(cfg['poll_s'])
            rec.update(health=status, startup_s=round(raw() - t0, 1))
            need(status == 'ok', f'{name}: server not ready (health {status}, exit {server.exited()}); see {log_path}')
            facts = facts_of(log_path.read_text(errors='replace'))
            geometry = get_json(port, '/v1/cache/status').get('geometry', {})
            model_id = get_json(port, '/v1/models')['data'][0]['id']
            calls = cache_records(folder / 'cache_state')
            pre = rec['pre_gates'] = pre_gates(arm, facts, geometry, model_id, calls, paths, cfg)
            rec.update(geometry=geometry, model_id=model_id, cache_state=calls,
                       server_args_differ=args_differ(arm, facts['server_args'], paths, cfg),
                       free_after_init_gib=facts['free_after_init_gib'],
                       free_after_capture_gib=facts['free_after_capture_gib'])
            need(all(pre.values()), f'{name}: gates at ready failed: {[k for k, v in pre.items() if not v]}')
            for number, prompt, decode in [(r.number, r.prompt, r.decode) for r in rows] + [
                    ('tail', rows[0].prompt, cfg['tail_decode'])]:   # the tail flushes the last predictor summary
                client: dict = {}
                try:
                    q = stream(port, model_id, prompt, decode, cfg, client)
                except BaseException as e:
                    # The failure record: the client-side record, the error, and whether the server still ran.
                    rec['stream_failure'] = dict(client, request=number, error=str(e), error_type=type(e).__name__,
                                                 cause=None if e.__cause__ is None else repr(e.__cause__),
                                                 server_alive=server.exited() is None, server_exit=server.exited())
                    write_json(folder / f'request_{number}_failure.json', rec['stream_failure'])
                    raise
                if number == 'tail':
                    rec['tail_passed'] = (q['completion_tokens'] == q['events'] == decode
                                          and q['finish_reason'] == 'length')
                    continue
                q['number'] = number
                write_json(folder / f'request_{number:02d}.json', q)
                reqs.append(q)
        finally:
            # The watch stops first: a sample while the server shuts down could call its own exiting process foreign.
            watch.__exit__()
            rec['gpu_watch'] = watch.summary()
            if pcie:
                rec['pcie'] = watch.pcie_summary()
            rec['stop'] = server.stop(cfg['stop_grace_s'])
            rec['serve_s'] = round(raw() - t0, 1)
        text = log_path.read_text(errors='replace')
        facts = facts_of(text)
        rec['device_use_bytes'] = device_use(facts)
        rec['predictor'] = predictor_health(arm, text, rows)
        post = rec['post_gates'] = dict(
            in_order=[q['number'] for q in reqs] == [r.number for r in rows],
            tokens=all(q['completion_tokens'] == q['decode'] and 2 <= q['events'] <= q['decode'] and q['text']
                       for q in reqs),
            finish_length=all(q['finish_reason'] == 'length' for q in reqs),
            stamps=all(stamps_ok(q) for q in reqs),
            no_log_error=not facts['log_errors'],
            predictor=rec['predictor']['passed'],
            tail=rec['tail_passed'],
            stopped_clean=rec['stop']['clean'],
            no_foreign_gpu_process=not rec['gpu_watch']['foreign'],
            gpu_watch_ok=rec['gpu_watch']['ok'],
        )
        need(all(post.values()), f'{name}: gates after the serve failed: {[k for k, v in post.items() if not v]}'
             + (f" ({rec['predictor']['problems'][:3]})" if not post['predictor'] else '')
             + (f" (log: {facts['log_errors']})" if not post['no_log_error'] else '')
             + (f" (other GPU processes: {rec['gpu_watch']['foreign'][:3]}; this script never stops them)"
                if not post['no_foreign_gpu_process'] else '')
             + (f" (GPU watch: {rec['gpu_watch']['samples']} samples, {rec['gpu_watch']['errors']} errors "
                f"{rec['gpu_watch']['error_types'][:3]}, thread joined {rec['gpu_watch']['joined']})"
                if not post['gpu_watch_ok'] else ''))
        scored = [q for q in reqs if q['number'] >= 1]
        values = [ms_per_token(q) for q in scored]
        rec['ms_per_token_by_prompt'] = dict(zip([str(q['number']) for q in scored], values))
        rec['events_by_prompt'] = {str(q['number']): q['events'] for q in scored}
        rec['ms_per_token_mean'] = sum(values) / len(values)
        rec['passed'] = True
        return rec
    finally:
        rec['end_wall'] = wall()
        write_json(folder / 'result.json', rec)


def outputs(folder: Path, numbers: list[int]) -> dict[int, tuple]:
    out = {}
    for n in numbers:
        q = json.loads((folder / f'request_{n:02d}.json').read_text())
        out[n] = (q['pieces'], q['piece_kinds'])
    return out


def run_pass(pass_name: str, out: Path, paths: LocalPaths, cfg: dict, first_pass: bool, snapshot: dict) -> dict:
    rows = prompt_rows(pass_name, cfg)
    numbers = [r.number for r in rows]
    record = dict(pass_name=pass_name, status='running', rows=numbers,
                  decode={str(r.number): r.decode for r in rows},
                  groups={g: [r.number for r in rows if r.group == g] for g in PASSES[pass_name]['groups']},
                  serves={}, cross_gates={})
    out.mkdir()
    first_cmd: dict[str, dict] = {}
    warm, warm_name = None, cfg['serves'][0][0]
    try:
        for i, (name, arm) in enumerate(cfg['serves']):
            print(f'[{wall()}] {pass_name}: serve {i + 1}/{len(cfg["serves"])} {name}', flush=True)
            folder = out / f'serve_{name}'
            rec = run_serve(name, arm, folder, rows, paths, cfg, first_cmd.get(arm), pcie=first_pass and i == 0,
                            snapshot=snapshot)
            record['serves'][name] = rec
            first_cmd.setdefault(arm, json.loads((folder / 'command.json').read_text()))
            # No timing here: the cross-serve gates below can still fail the pass, and the timings of a failed pass
            # are not read or shown (the study's rule). They print once the whole pass is admitted.
            print(f'[{wall()}]   serve gates passed', flush=True)
            if i == 0:
                warm = outputs(folder, numbers)
                continue
            same = outputs(folder, numbers) == warm
            record['cross_gates'][f'{name}_text_equals_{warm_name}'] = same
            need(same, f'{name}: the streamed text differs from {warm_name}: the output changed')
            if arm in PREDICTOR_ARMS:
                def misses(s):
                    return {r['number']: r['summary']['misses_from_stage'] + r['summary']['misses_from_host']
                            for r in s['predictor']['requests']}
                equal = misses(rec) == misses(record['serves'][warm_name])
                record['cross_gates'][f'{name}_misses_equal_{warm_name}'] = equal
                need(equal, f'{name}: misses per row differ from {warm_name}')
        timed = [(n, a) for n, a in cfg['serves'][1:]]
        mean = lambda a: sum(record['serves'][n]['device_use_bytes'] for n, x in timed if x == a) / sum(  # noqa: E731
            1 for _, x in timed if x == a)
        diff = mean('F0Q') - mean('R584')
        lo, hi = cfg['device_band_bytes']
        record['equal_memory'] = dict(f0q_mean_bytes=mean('F0Q'), r584_mean_bytes=mean('R584'), difference_bytes=diff,
                                      difference_slots=diff / SLOT_BYTES, band_bytes=[lo, hi], inside=lo <= diff <= hi)
        need(lo <= diff <= hi, f'device use difference {diff / SLOT_BYTES:.2f} slots outside [-5, +2]: '
                               f'the arms did not use equal GPU memory')
        check_evidence(snapshot, paths, cfg)   # no input changed during the last serve
        record['status'] = 'complete'
        for name, _ in timed:
            m = record['serves'][name]['ms_per_token_mean']
            print(f'[{wall()}]   {name}: mean {m:.3f} ms/token ({1000 / m:.2f} tokens/s)', flush=True)
        return record
    except BaseException as e:
        record.update(status='failed', error=str(e))
        raise
    finally:
        write_json(out / 'record.json', record)


# ---- preflight and run -----------------------------------------------------------------------------------------------

def hardware() -> dict:
    """What the report states about the machine. Read-only queries."""
    g = nvsmi('name,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max,vbios_version')
    cpu = re.search(r'^model name\s*:\s*(.*)$', Path('/proc/cpuinfo').read_text(), re.M)
    total = re.search(r'^MemTotal:\s+(\d+) kB$', Path('/proc/meminfo').read_text(), re.M)
    osr = Path('/etc/os-release')
    pretty = re.search(r'^PRETTY_NAME="?(.*?)"?$', osr.read_text(), re.M) if osr.exists() else None
    return dict(gpus=[dict(zip(('name', 'driver', 'memory_total_mib', 'pcie_gen_max', 'pcie_width_max', 'vbios'), row))
                      for row in g],
                cpu=cpu.group(1) if cpu else None, mem_total_bytes=int(total.group(1)) * 1024 if total else None,
                kernel=os.uname().release, os=pretty.group(1) if pretty else None,
                wsl='microsoft' in os.uname().release.lower())


ORIGIN_PROBE = ("import sys, importlib.util as u, freetoken, commitmoe_prefetch.freetoken as f; "
                "s = u.find_spec('sitecustomize'); print(freetoken.__file__); print(f.__file__); "
                "print(s.origin if s else None); print('usercustomize' in sys.modules)")


def evidence(paths: LocalPaths, cfg: dict) -> dict:
    """Every fixed input of a serve, as the study's bench recorded it (exp609_port_bench.evidence): the engine code
    against the tested commit, the import origins of the serve environment (the engine and the cache hook must load
    from this checkout, and no user customization), every predictor release file, the R584 bundle, the prompts and
    this run's code. preflight() compares it with the pins; every serve and every pass end compares it with the
    preflight snapshot, so an input that changes during the run stops it.

    Not hashed: the model shards (35 GB; hashing them before every serve would add minutes per serve). The model is
    pinned by its revision folder (setup.sh), and the 'right_model' gate checks the path the server loaded."""
    diff = run(['git', '-C', str(REPO), 'diff', '--quiet', ENGINE_COMMIT, '--', *ENGINE_PATHS])
    dirty = run(['git', '-C', str(REPO), 'status', '--porcelain', '--', *ENGINE_PATHS])
    probe = run([str(paths.python), '-s', '-c', ORIGIN_PROBE], timeout=180, env=serve_env('F0Q', paths, cfg, None),
                cwd=REPO)
    lines = probe.stdout.splitlines() if probe.returncode == 0 else []
    return dict(
        engine_equals_commit=diff.returncode == 0 and dirty.returncode == 0 and not dirty.stdout.strip(),
        import_origins=lines[-4:] if len(lines) >= 4 else [f'probe failed: {probe.stderr.strip()[-300:]}'],
        predictor={f: sha256(paths.predictor / f) if (paths.predictor / f).is_file() else None
                   for f in PREDICTOR_RELEASE},
        bundle=sha256(BUNDLE),
        prompts={p: sha256(spec['prompts']) for p, spec in PASSES.items()},
        run_code={k: sha256(v) for k, v in RUN_CODE.items()},
    )


def pin_problems(ev: dict) -> list[str]:
    """What in the evidence differs from the pins. Empty when every input is the tested one."""
    want_origins = [str((REPO / 'python' / 'freetoken' / '__init__.py').resolve()),
                    str((REPO / 'commitmoe_prefetch' / 'commitmoe_prefetch' / 'freetoken.py').resolve()),
                    str((HERE / 'cache_hook' / 'sitecustomize.py').resolve()), 'False']
    got_origins = [str(Path(x).resolve()) if x not in ('False', 'True', 'None') else x for x in ev['import_origins']]
    problems = [] if ev['engine_equals_commit'] else [f'engine code differs from the tested commit {ENGINE_COMMIT}']
    if got_origins != want_origins:
        problems.append(f"import origins differ: {ev['import_origins']} (want {want_origins})")
    problems += [f'predictor release file {f} differs from the pinned release'
                 for f, h in PREDICTOR_RELEASE.items() if ev['predictor'].get(f) != h]
    if ev['bundle'] != BUNDLE_SHA256:
        problems.append('the R584 bundle differs from its pinned sha256')
    problems += [f'{PASSES[p]["prompts"].name} differs from its pinned sha256'
                 for p, spec in PASSES.items() if ev['prompts'][p] != spec['sha256']]
    return problems


def check_evidence(snapshot: dict, paths: LocalPaths, cfg: dict) -> None:
    current = evidence(paths, cfg)
    changed = sorted(k for k in snapshot if snapshot[k] != current.get(k))
    need(not changed, f'an input changed during the run: {changed}')


def preflight(paths_file: Path, cfg: dict) -> tuple[LocalPaths, dict]:
    """End state C on any failure: nothing was timed."""
    need(paths_file.is_file(), f'{paths_file} missing: run bash replication/setup.sh first', 'C')
    try:
        paths = LocalPaths.model_validate_json(paths_file.read_text())
    except ValueError as e:
        raise Stop('C', f'{paths_file}: {e}') from e
    ev = evidence(paths, cfg)
    problems = pin_problems(ev)
    need(not problems, '; '.join(problems), 'C')
    try:
        hw = hardware()
    except Stop as e:
        raise Stop('C', str(e)) from e
    need(len(hw['gpus']) >= 1 and cfg['expect_gpu'] in hw['gpus'][0]['name'],
         f"GPU 0 is {hw['gpus'][0]['name'] if hw['gpus'] else 'missing'}, expected '{cfg['expect_gpu']}' "
         f"(CONFIG['expect_gpu'])", 'C')
    apps = compute_apps()
    need(not apps, f'another process uses the GPU: {apps}. Close it; this script never stops it.', 'C')
    free = gpu_free_mib()
    need(free >= cfg['min_gpu_free_mib'], f"GPU 0 has {free} MiB free, a serve needs {cfg['min_gpu_free_mib']}", 'C')
    avail = mem_available()
    need(avail >= cfg['host_need_bytes'], f"MemAvailable is {avail / 1e9:.1f} GB, a serve needs "
                                          f"{cfg['host_need_bytes'] / 1e9:.1f} GB. Close programs or add RAM.", 'C')
    cfg['results'].mkdir(parents=True, exist_ok=True)
    disk = os.statvfs(cfg['results'])
    need(disk.f_bavail * disk.f_frsize >= cfg['min_disk_free_bytes'], 'less than 2 GB free for the results', 'C')
    sw = run([str(paths.python), '-c', 'import json, torch, triton, freetoken; print(json.dumps(dict(torch=torch.__version__, '
              'cuda=torch.version.cuda, triton=triton.__version__, capability=list(torch.cuda.get_device_capability(0)))))'],
             timeout=180, env=dict(serve_env('F0Q', paths, cfg, None), CUDA_VISIBLE_DEVICES='0'))
    need(sw.returncode == 0, f'the engine environment does not import: {sw.stderr.strip()[-300:]}', 'C')
    return paths, dict(hardware=hw, software=json.loads(sw.stdout.strip().splitlines()[-1]), gpu_free_mib=free,
                       mem_available_bytes=avail, evidence=ev)


def main(cfg: dict = CONFIG) -> int:
    cfg = dict(cfg)
    name = cfg['run_name'] or time.strftime('%Y%m%d-%H%M')
    run_dir = Path(cfg['results']) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    info = dict(run=name, start_wall=wall(), engine_commit=ENGINE_COMMIT, state=None, passes={},
                config={k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()},
                smoke=cfg['max_scored_rows'] is not None)
    print(f'[{wall()}] replication run {name}: results in {run_dir}', flush=True)
    code = 0
    try:
        paths, env = preflight(HERE / 'local_paths.json', cfg)
        info.update(env, paths={k: str(v) for k, v in paths.model_dump().items()})
        write_json(run_dir / 'run.json', info)
        for i, p in enumerate(cfg['passes']):
            info['passes'][p] = 'running'
            run_pass(p, run_dir / p, paths, cfg, first_pass=i == 0, snapshot=env['evidence'])
            info['passes'][p] = 'complete'
        info['state'] = 'A'
    except Stop as e:
        info.update(state=e.state, failure=str(e))
        for p, s in info['passes'].items():
            if s == 'running':
                info['passes'][p] = 'failed'
    except KeyboardInterrupt:
        info.update(state='interrupted', failure='interrupted with Ctrl-C')
        code = 130
    except Exception as e:   # noqa: BLE001 - a bug or an unexpected engine answer still ends in a recorded state B
        info.update(state='B', failure=f'unexpected error: {e!r}', traceback=traceback.format_exc())
    finally:
        info['end_wall'] = wall()
        write_json(run_dir / 'run.json', info)
    code = code or EXIT[info['state']]
    try:
        import report   # replication/report.py, beside this file
        page = report.build(run_dir)
        print(f'[{wall()}] report: {page}', flush=True)
    except Exception as e:   # noqa: BLE001 - the records stay; the report can be built again with report.py
        print(f'[{wall()}] report not built: {e!r}. Run: python replication/report.py {run_dir}', flush=True)
    print(f"[{wall()}] end state {info['state']}" + (f": {info.get('failure')}" if info.get('failure') else ''),
          flush=True)
    return code


if __name__ == '__main__':
    sys.path.insert(0, str(HERE))
    sys.exit(main())
