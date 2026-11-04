"""FreeToken decode hooks that prefetch the experts the predictor expects for the next token.

Entry point: ``create_decode_hooks(source, config)``, which FreeToken's --expert-predictor
calls. One decode token runs this schedule (the one of the measured arm GV1q16t):

  end of token t (main stream, after the decode graph):
      copy token t's captured features into the score inputs; store token t's id, the
      sampled id of token t+1 and the position.
  token t+1, every MoE layer L (inside the decode graph):
      observe   capture L's hidden state, router logits and routed ids (main stream).
      L == 0    the probe stream forks from main and scores all heads from the end-of-token-t
                inputs, overlapping L0's demand copy.
      demand    on the demand stream: mark the copy busy, copy L's misses (from a ready stage
                slot when one holds the expert, else from the host bank), clear busy. Main
                waits for it before the expert GEMM.
      L < 35    on the probe stream: at L = 5s, retire stage group s % 2 and select the top
                nonresident experts of target layers 5(s+1) .. 5(s+1)+4; at every layer, run
                the pump, which copies the selected experts while no demand copy runs.
      last L    main waits for the probe stream, so all prefetch work of the token ends with it.

Every input of the scores exists at the end of token t, and the selection reads the live
cache residency, so a stage slot only ever holds an expert that was not resident.

Learned eviction (EXP526, off by default; on when FREETOKEN_LEARNED_EVICT names a bundle):
      L == 0    observe() forks the probe stream before L0's ensure, so the keep scores are
                ready as early as possible. The probe runs the bank trunk once, writes the keep
                scores of the eviction layers, marks them valid (step > 0) and records the
                scores-ready event; then it finishes the prefetch scores from the same trunk
                output. copy_missing(0) does not score again, so the EMA prior moves once.
      L == J    observe() makes main wait for the scores-ready event before L's ensure.
      L >= J    ensure_experts chooses victims with learned_evict.learned_ensure; layers below
                J use the unchanged LRU rule. J is FREETOKEN_LEARNED_EVICT_START (required).

Shared learned eviction (EXP585, off by default; on when FREETOKEN_SHARED_EVICT names a bundle of
experiments/exp585_export.py; refused together with FREETOKEN_LEARNED_EVICT):
      L == 0    observe() snapshots the slot map on main (resident = slot >= 0): main has run every
                cache mutation of token t (its graph joined the demand and probe streams at the
                last layer), and none of token t+1 yet (L0's ensure follows observe). Then the
                probe stream forks and runs shared_evict.SharedEvict: advance the 40-layer eviction
                EMA and age with token t's routes (step > 0 only), score all 40 layers from R584's
                own trunk, publish valid = (step > 0) & finite, record scores-ready, and only then
                run the prefetch bank (trunk and head) and its EMA prior, unchanged.
      L == J    as above; every layer >= J reads its keep row (J from FREETOKEN_LEARNED_EVICT_START).

Fast shared learned eviction (EXP586; FREETOKEN_SHARED_EVICT_FAST = blockn or splitk, the F8 variant of
shared_evict_fast; needs FREETOKEN_SHARED_EVICT): the same bundle, state and function, scored by
shared_evict_fast.FastSharedEvict. In this mode only, the probe scores and publishes before it copies the
bank's raw features (R584 reads score_inputs and raw tid/upcoming_tid/pos, none of the copied fields). The
copies stay safe after scores-ready: main joins the whole probe stream at the last layer
(_after_demand), before after_decode writes score_inputs, tid, upcoming_tid and pos of the next token.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import triton

from freetoken.moe.decode_hooks import DecodeHooks
from freetoken.moe.learned_evict import COUNTERS as EVICT_COUNTERS
from freetoken.moe.learned_evict import LearnedEvict
from freetoken.moe.striped_demand import StripedDemand
from freetoken.utils import init_logger

from . import bank_loader, hub, kernels
from .evict import EvictRows
from .qbank import Int8Bank, qlinear
from .shared_evict import SharedEvict
from .shared_evict import load_bundle as load_shared_bundle
from .shared_evict_fast import F8, FastSharedEvict
from .spec import PredictorConfig

if TYPE_CHECKING:
    from freetoken.engine.config import EngineConfig
    from freetoken.moe.offload_cache import OffloadMoeCache

# FreeToken's loggers do not propagate to the root logger and the server configures no root
# handler, so a plain logging.getLogger logger would drop every message of this module.
logger = init_logger(__name__)

# Private device buffer for the stage slots (GROUPS * SLOTS experts per bank), the size of
# the measured run. The cache budget is computed after the hooks reserve memory, so the
# size must be fixed before the cache exists; bind() checks that the slots fit in it.
STAGE_BYTES = 32 * 1024**2
# Minimum head probability for a copy. The measured run used no gate (EXP497 tested gates).
MIN_SCORE = 0.0
LIVE = ('h', 'rl', 'routes', 'prev')  # per-layer features of the current and previous token
# Learned eviction switch: a bundle path (exp526_export_evict.py) and the first layer J whose
# ensure uses the keep scores. J has no default: it must come from the measured scores-ready wait.
EVICT_ENV, EVICT_START_ENV = 'FREETOKEN_LEARNED_EVICT', 'FREETOKEN_LEARNED_EVICT_START'
# Diagnostic for choosing J (EXP526): a .pt path. Per token, the GPU timer (ns) is stamped in
# columns: num_layers = the probe, right before it records the scores-ready event; num_layers + 1 =
# main at layer J right before its wait; num_layers + 2 = main right after the wait (the exposed
# wait is column +2 minus column +1). With EVICT_TIMING_ALL_ENV=1, columns 0..num_layers-1 also
# hold main's arrival at every observe(L), to screen candidate J values in one run (those 40 extra
# kernels slow main down: confirm the chosen J with the plain timing). The rows of all tokens so
# far are written when the next request begins, so the last request of a run is not written.
# The same file holds 'join' (EXP586), two columns per token: main right before (0) and right
# after (1) its wait for the probe stream at the last layer. A separate tensor, so that the
# column layout above stays the one that the EXP526/EXP585 checks read.
EVICT_TIMING_ENV, EVICT_TIMING_ALL_ENV = 'FREETOKEN_LEARNED_EVICT_TIMING', 'FREETOKEN_LEARNED_EVICT_TIMING_ALL'
# Shared learned eviction switch (EXP585): an exp585_export.py bundle path. It uses EVICT_START_ENV for J.
SHARED_EVICT_ENV = 'FREETOKEN_SHARED_EVICT'
# Fast shared learned eviction (EXP586): the F8 variant name; only with SHARED_EVICT_ENV.
SHARED_EVICT_FAST_ENV = 'FREETOKEN_SHARED_EVICT_FAST'
# Baseline picks for the EXP600 comparison, off by default. The bank still runs, so that its cost stays
# in the arm; only the scores that select_scored reads change. random: a hash of (expert, step, request,
# seed). ema: the EMA prior after token t, with the hash as a tie-break among never-used experts.
PICK_BASELINE_ENV, PICK_SEED_ENV = 'FREETOKEN_PICK_BASELINE', 'FREETOKEN_PICK_SEED'
PICK_BASELINES = ('random', 'ema')
# With value 1: per-layer published counts and one more log line per request (EXP600, untimed serves).
LAYER_COUNTERS_ENV = 'FREETOKEN_LAYER_COUNTERS'
# Below every prior of an expert used in the last 240 tokens (0.1 * 0.9 ** 240 is about 1e-12), and fp32
# holds it beside a prior of zero. So the tie-break cannot reorder experts with recent use.
EMA_TIE_BREAK = 1e-12
# Odd constants under 2**31 (murmur2 and murmur3). A masked value under 2**32 times one of them stays under 2**63, so
# no product overflows int64 and the low 32 bits are exact on every device.
_MUL = (0x5BD1E995, 0x1B873593)
_MASK32 = 0xFFFFFFFF


def hash_unit(ids: torch.Tensor, step: torch.Tensor, request: torch.Tensor, seed: int) -> torch.Tensor:
    """A value in [0, 1) per id, fixed by (id, step, request, seed): two xor-shift and multiply rounds.

    Trap: a torch random call inside a captured CUDA graph keeps its generator state outside the graph,
    so it cannot be trusted to give new values per replay. This hash reads the step and request buffers,
    which the main stream updates before each replay.
    """
    x = (ids + step.to(torch.int64) * 0x9E3779B1 + (request + seed) * 0x61C88647) & _MASK32
    x = x ^ (x >> 16)
    x = (x * _MUL[0]) & _MASK32
    x = x ^ (x >> 13)
    x = (x * _MUL[1]) & _MASK32
    x = x ^ (x >> 16)
    return (x >> 8).to(torch.float32) * 2.0 ** -24   # 24 bits: exact in fp32, so the value stays under 1


def baseline_scores(kind: str, prior: torch.Tensor, ids: torch.Tensor, step: torch.Tensor,
                    request: torch.Tensor, seed: int) -> torch.Tensor:
    """Scores (heads, E) of a baseline pick. Both are causal at the end of token t: random reads no
    model input, and the prior holds routes of tokens up to t."""
    u = hash_unit(ids, step, request, seed)
    return u if kind == 'random' else prior + EMA_TIE_BREAK * u


def pick_baseline_from_env(env) -> tuple[str, int] | None:
    """(kind, seed) when a baseline pick is switched on, else None (the published scores)."""
    kind = env.get(PICK_BASELINE_ENV)
    if not kind:
        return None
    if kind not in PICK_BASELINES:
        raise ValueError(f'{PICK_BASELINE_ENV}={kind!r}: need one of {PICK_BASELINES}')
    seed = env.get(PICK_SEED_ENV)
    if seed is None or not seed.isdigit():
        raise ValueError(f'{PICK_BASELINE_ENV} is set; {PICK_SEED_ENV} must give a seed of digits, got {seed!r}')
    return kind, int(seed)


class PrefetchHooks(DecodeHooks):
    """Scores, selections and copies of one predictor on one FreeToken engine."""

    single_request = True

    def __init__(self, cfg: PredictorConfig, state: dict[str, torch.Tensor], demand_stripes: int,
                 evict: tuple[Path, int] | None = None, timing: Path | None = None, timing_all: bool = False,
                 release_sha256: str | None = None, shared: tuple[Path, int] | None = None,
                 shared_fast: str | None = None, pick: tuple[str, int] | None = None,
                 layer_counters: bool = False):
        base, feats = cfg.base_model, cfg.features
        layers, sl = base.num_layers, kernels.SECTION_LAYERS
        targets = cfg.target_layers
        problems = []
        if layers % sl:
            problems.append(f'{layers} layers is not a multiple of the section length {sl}')
        if list(targets) != list(range(targets[0], layers)):
            problems.append(f'target layers {targets[0]}..{targets[-1]} must run without gaps to the last layer')
        self.sections = layers // sl - 1
        caps = cfg.serving_reference.section_caps
        if len(caps) != self.sections or not all(0 <= c <= kernels.SLOTS for c in caps):
            problems.append(f'section caps {caps}: need {self.sections} values in 0..{kernels.SLOTS}')
        if demand_stripes < 1:
            problems.append('--moe-demand-copy-stripes must be at least 1')
        if evict is not None and not 0 <= evict[1] < layers:
            problems.append(f'{EVICT_START_ENV}={evict[1]} is not a layer 0..{layers - 1}')
        if evict is not None and shared is not None:
            problems.append(f'{EVICT_ENV} and {SHARED_EVICT_ENV} are two eviction rules; set one')
        if shared is not None and not 0 <= shared[1] < layers:
            problems.append(f'{EVICT_START_ENV}={shared[1]} is not a layer 0..{layers - 1}')
        if shared_fast is not None and shared is None:
            problems.append(f'{SHARED_EVICT_FAST_ENV} needs {SHARED_EVICT_ENV}')
        if shared_fast is not None and shared_fast not in F8:
            problems.append(f'{SHARED_EVICT_FAST_ENV}={shared_fast!r} is not one of {F8}')
        if timing is not None and evict is None and shared is None:
            problems.append(f'{EVICT_TIMING_ENV} needs {EVICT_ENV} or {SHARED_EVICT_ENV}')
        if problems:
            raise ValueError('expert predictor: ' + '; '.join(problems))
        self.cfg, self._state, self.demand_stripes = cfg, state, demand_stripes
        self.caps = caps
        self.first = targets[0]
        self.heads = len(targets)
        self.offset = feats.hidden_source_offset
        # Router logits that some head reads (layers L-1 .. L-window of every target L).
        window = feats.router_logit_window
        self.rl_layers = frozenset(t - d for t in targets for d in range(1, window + 1)) \
            - set(feats.router_logit_layers_zeroed)
        self.uid: int | None = None
        self.steps = 0
        self.evict_cfg, self.timing, self.timing_all = evict, timing, timing_all
        self.release_sha256 = release_sha256
        self.stamp_rows: list[torch.Tensor] = []
        self.stamp_steps: list[int] = []
        self.join_rows: list[torch.Tensor] = []
        self.learned: LearnedEvict | None = None
        self.shared_cfg, self.shared_fast = shared, shared_fast
        self.shared: SharedEvict | None = None
        self.pick, self.layer_counters = pick, layer_counters

    # -- setup ---------------------------------------------------------------------

    def reserve(self, engine) -> None:
        cfg, dev = self.cfg, engine.device
        base, feats = cfg.base_model, cfg.features
        L, E, K, D, H = base.num_layers, base.num_experts, base.top_k, base.hidden_size, self.heads
        self.bank = Int8Bank(cfg, self._state, dev)
        del self._state  # the CPU copy is not needed after the upload
        self.live = {
            'h': torch.zeros(L, D, dtype=torch.float16, device=dev),
            'rl': torch.zeros(L, E, dtype=torch.float16, device=dev),
            'routes': torch.zeros(L, K, dtype=torch.int64, device=dev),
            'prev': torch.zeros(L, K, dtype=torch.int64, device=dev),
        }
        # Frozen copy of the live features at the end of token t; the decode graph of token
        # t+1 scores from it while its own layers overwrite the live buffers.
        self.score_inputs = {name: torch.zeros_like(t) for name, t in self.live.items()}
        self.raw = {
            'h': torch.zeros(H, D, dtype=torch.float16, device=dev),
            'h_prev': torch.zeros(H, D, dtype=torch.float16, device=dev),
            'win': torch.zeros(H, feats.route_window, K, dtype=torch.int64, device=dev),
            'prevwin': torch.zeros(H, feats.previous_token_window, K, dtype=torch.int64, device=dev),
            'curr': torch.zeros(H, K, dtype=torch.int64, device=dev),
            'prior_before': torch.zeros(H, E, dtype=torch.float32, device=dev),
            'pos': torch.zeros(H, 1, dtype=torch.float32, device=dev),
            'rl': torch.zeros(H, feats.router_logit_window * E, dtype=torch.float16, device=dev),
            'tid': torch.zeros(1, dtype=torch.int64, device=dev),
            'upcoming_tid': torch.zeros(1, dtype=torch.int64, device=dev),
        }
        self.probs = torch.zeros(H, E, dtype=torch.float32, device=dev)
        self.layer_published = torch.zeros(L, dtype=torch.int64, device=dev)   # FREETOKEN_LAYER_COUNTERS
        if self.pick is not None:
            self.pick_ids = torch.arange(self.first * E, (self.first + H) * E, device=dev).view(H, E)
            self.pick_requests = torch.zeros(1, dtype=torch.int64, device=dev)   # begin_request counts
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)  # decode step of the running graph
        slots, stripes = kernels.GROUPS * kernels.SLOTS, kernels.PUMP_STRIPES
        self.demand = StripedDemand(L, E, self.demand_stripes, stage_slots=slots, device=dev)
        self.records = torch.full((self.sections, kernels.SLOTS), -1, dtype=torch.int64, device=dev)
        self.cursors = torch.zeros((self.sections, stripes), dtype=torch.int32, device=dev)
        self.payload = torch.zeros((self.sections, kernels.SLOTS), dtype=torch.int32, device=dev)
        self.stats = torch.zeros(len(kernels.STATS_FIELDS), dtype=torch.int64, device=dev)
        self.unscored_ids = torch.zeros(1, dtype=torch.int64, device=dev)  # ids replaced by row 0
        self.busy = torch.zeros(1, dtype=torch.int32, device=dev)
        self.layer = torch.full((1,), -1, dtype=torch.int32, device=dev)
        self.storage = torch.empty(STAGE_BYTES, dtype=torch.uint8, device=dev)
        self.probe_stream = torch.cuda.Stream(device=dev, priority=-1)
        # begin_copy (EXP538): per layer, the point of main that the probe section depends on
        self.copy_events = [torch.cuda.Event() for _ in range(self.cfg.base_model.num_layers)]
        if self.evict_cfg is not None:
            path, start = self.evict_cfg
            self.evict = EvictRows(path, list(cfg.target_layers), cfg.architecture.up_dim, E,
                                   cfg.serving_reference.int8_group, dev, self.release_sha256, L)
            M = len(self.evict.layers)
            target_of_layer = torch.full((L,), -1, dtype=torch.int32)
            target_of_layer[self.evict.layers] = torch.arange(M, dtype=torch.int32)
            self.learned = LearnedEvict(keep=torch.zeros(M, E, dtype=torch.float32, device=dev),
                                        valid=torch.zeros(1, dtype=torch.int32, device=dev),
                                        target_of_layer=target_of_layer.to(dev), start_layer=start,
                                        counters=torch.zeros(len(EVICT_COUNTERS), dtype=torch.int64, device=dev))
            self.scores_ready = torch.cuda.Event()
            self.stamps = torch.zeros(L + 3, dtype=torch.int64, device=dev)
            self.join_stamps = torch.zeros(2, dtype=torch.int64, device=dev)
            # Tokens whose keep scores were not all finite (published with valid = 0, so LRU).
            self.nonfinite_tokens = torch.zeros(1, dtype=torch.int64, device=dev)
            logger.info('learned eviction: layers %s (source heads %s) from layer %d, rows %.1f MiB (%s)',
                        self.evict.layers, self.evict.source_layers, start, self.evict.nbytes / 2**20, path)
        if self.shared_cfg is not None:
            path, start = self.shared_cfg
            meta, tensors = load_shared_bundle(path, self.release_sha256, L, E, K, base.hidden_size,
                                               self.bank.buffers['table'],
                                               feats.position_scale, cfg.serving_reference.int8_group)
            if self.shared_fast is None:
                self.shared = SharedEvict(meta, tensors, dev, qlinear)
            else:  # the torch path on a CPU device (tests), the fused kernels on the GPU
                self.shared = FastSharedEvict(meta, tensors, dev, self.shared_fast, kernels=dev.type == 'cuda')
            del tensors
            # End-of-token-t residency (slot >= 0, the rule of the overlap path), written on main in observe(0).
            self.resident = torch.zeros(L, E, dtype=torch.float32, device=dev)
            self.learned = LearnedEvict(keep=torch.zeros(L, E, dtype=torch.float32, device=dev),
                                        valid=torch.zeros(1, dtype=torch.int32, device=dev),
                                        target_of_layer=torch.arange(L, dtype=torch.int32, device=dev),
                                        start_layer=start,
                                        counters=torch.zeros(len(EVICT_COUNTERS), dtype=torch.int64, device=dev))
            self.scores_ready = torch.cuda.Event()
            self.stamps = torch.zeros(L + 3, dtype=torch.int64, device=dev)
            self.join_stamps = torch.zeros(2, dtype=torch.int64, device=dev)
            self.nonfinite_tokens = torch.zeros(1, dtype=torch.int64, device=dev)
            logger.info('shared learned eviction: all %d layers from layer %d, R584 %.1f MiB (%s), fast %s',
                        L, start, self.shared.nbytes / 2**20, path, self.shared_fast)
        logger.info('expert predictor: %d heads L%02d-L%02d, int8 bank %.1f MiB, stage %d MiB, caps %s',
                    H, self.first, L - 1, self.bank.nbytes / 2**20, STAGE_BYTES // 2**20, list(self.caps))
        if self.pick is not None:
            logger.info('expert predictor: baseline picks %s, seed %d', *self.pick)
        if self.layer_counters:
            logger.info('expert predictor: layer counters on')

    def bind(self, engine, cache: OffloadMoeCache) -> None:
        base = self.cfg.base_model
        if (cache.num_layers, cache.num_experts) != (base.num_layers, base.num_experts):
            raise ValueError(f'expert predictor expects {base.num_layers} layers x {base.num_experts} experts, '
                             f'the offload cache has {cache.num_layers} x {cache.num_experts}')
        self.demand.check_cache(cache)
        # The engine refuses the hooks unless the cache has one slot pool, and a pool holds layers of
        # one bank layout, so one row-size tensor serves every layer. Read it again on every bind:
        # a cache rebuild makes new descriptors.
        if len(cache.layouts) != 1:
            raise ValueError(f'expert predictor needs one bank layout, the offload cache has {len(cache.layouts)}')
        self.feat_bytes = cache._copy_feat_bytes[cache.layer_layout[0]]
        slots = kernels.GROUPS * kernels.SLOTS
        offset, views = 0, []
        # Size each stage region from the copy descriptor, not from the bank tensor: HEAD stores
        # padded layouts as flat uint8 banks, where one bank element is one byte, not one expert row.
        for row_bytes in self.feat_bytes.tolist():
            size = slots * int(row_bytes)
            if offset + size > STAGE_BYTES:
                raise ValueError(f'{slots} stage slots of every bank need more than {STAGE_BYTES // 2**20} MiB')
            views.append(self.storage[offset:offset + size])
            offset += size
        self.stage_ptrs = torch.tensor([v.data_ptr() for v in views], dtype=torch.int64, device=self.storage.device)
        self.hosts = torch.stack(cache._copy_src_ptrs).contiguous()  # (layers, banks) host row pointers
        self.num_banks = len(views)
        words = sum(int(b) // 8 for b in self.feat_bytes.tolist())
        chunks = triton.cdiv(words, kernels.CHUNK_BYTES // 8)  # chunks per expert
        self.quota = kernels.SLOTS * chunks  # one pump call may try every chunk of a full section
        if self.shared is not None:
            if tuple(cache.slot_for_id.shape) != tuple(self.resident.shape):
                raise ValueError(f'shared eviction: slot map {tuple(cache.slot_for_id.shape)}, '
                                 f'expected {tuple(self.resident.shape)}')
            self.cache = cache  # observe(0) snapshots its slot map; bind runs again after a cache rebuild
        cache.learned_evict = self.learned
        self.begin_request(-1)

    # -- per request and per token (main stream, outside the graph) ----------------

    def begin_request(self, uid: int) -> None:
        if uid != self.uid and self.steps:
            logger.info('expert predictor: request %s, %d decode steps: %s', self.uid, self.steps, self.summary())
            # Decode misses (stage + host) of every layer, for per-layer comparisons (EXP526).
            per_layer = self.demand.counters[:, :2].sum(1).tolist()
            logger.info('expert predictor: request %s, misses per layer: %s', self.uid, per_layer)
            if self.layer_counters:
                logger.info('expert predictor: request %s, layer counters: %s', self.uid, self._layer_counts())
            if self.timing is not None and self.stamp_rows:
                torch.save(dict(stamps=torch.stack(self.stamp_rows).cpu(), step=torch.tensor(self.stamp_steps),
                                join=torch.stack(self.join_rows).cpu(), start_layer=self.learned.start_layer),
                           self.timing)
        self.uid, self.steps = uid, 0
        # The previous token's graph joined its probe and demand streams into main, so these
        # writes on main are ordered after all of its prefetch work.
        for buffers in (self.live, self.score_inputs, self.raw):
            for t in buffers.values():
                t.zero_()
        self.probs.zero_()
        self.layer_published.zero_()
        if self.pick is not None:
            self.pick_requests.add_(1)   # the same request order gives the same hash in each serve of an arm
        self.records.fill_(-1)
        self.cursors.zero_()
        self.payload.zero_()
        self.stats.zero_()
        self.unscored_ids.zero_()
        # The start state of a fresh engine. A graph warmup or recapture after serving runs
        # with this state, so the step-0 guard of the prior holds and no copy is attempted.
        self.step.zero_()
        self.demand.gen.fill_(1)
        self.busy.zero_()
        self.layer.fill_(-1)
        self.demand.ids.fill_(-1)
        self.demand.ready.zero_()
        self.demand.reset()
        if self.learned is not None:
            # No keep score survives a request boundary: until this request's first scored token
            # publishes, every ensure uses LRU (valid == 0).
            self.learned.keep.zero_()
            self.learned.valid.zero_()
            self.learned.counters.zero_()
            self.nonfinite_tokens.zero_()
            if self.shared is not None:
                self.shared.reset()
                self.resident.zero_()

    def before_decode(self, step: int) -> None:
        self.step.fill_(step)
        # GEN > 0 on every token; a stage published under another token's GEN is never read.
        self.demand.gen.fill_(step + 1)
        self.live['prev'].copy_(self.live['routes'])  # token t-1's routes for prevwin

    def after_decode(self, step: int, consumed: torch.Tensor, sampled: torch.Tensor) -> None:
        for name, t in self.live.items():
            self.score_inputs[name].copy_(t)
        rows = self.cfg.features.token_table_rows
        # Exception to the release contract (spec.py), which rejects ids without a table row
        # (audio/TTS ids): a serving runtime must not stop a request for a prefetch input. Row 0
        # stands in, the scores of that token have no meaning, and summary() counts these ids.
        # The model output is unaffected: every miss still gets its demand copy.
        for name, ids in (('tid', consumed), ('upcoming_tid', sampled)):
            ids = ids.reshape(1).to(torch.int64)
            valid = (ids >= 0) & (ids < rows)
            self.raw[name].copy_(torch.where(valid, ids, 0))
            self.unscored_ids.add_((~valid).to(torch.int64))
        # Training covered positions up to 256 tokens; later positions exceed 1.0.
        self.raw['pos'].fill_((step + 1) / self.cfg.features.position_scale)
        self.steps = step + 1
        if self.timing is not None:
            self.stamp_rows.append(self.stamps.clone())
            self.join_rows.append(self.join_stamps.clone())
            self.stamp_steps.append(step)

    # -- inside the decode graph ---------------------------------------------------

    def observe(self, layer_id: int, hidden: torch.Tensor, router_logits: torch.Tensor | None,
                topk_ids: torch.Tensor) -> None:
        base = self.cfg.base_model
        D, E, K = base.hidden_size, base.num_experts, base.top_k
        keep = layer_id in self.rl_layers
        if keep and router_logits is None:
            raise RuntimeError(f'expert predictor: layer {layer_id} gives no router logits')
        if hidden.shape[-1] != D or hidden.stride(-1) != 1 or topk_ids.stride(-1) != 1 or topk_ids.shape[-1] != K:
            raise RuntimeError(f'expert predictor: layer {layer_id} hidden {tuple(hidden.shape)}, '
                               f'routed ids {tuple(topk_ids.shape)}; expected (1, {D}) and (1, {K})')
        if keep and (router_logits.shape[-1] != E or router_logits.stride(-1) != 1):
            raise RuntimeError(f'expert predictor: layer {layer_id} router logits {tuple(router_logits.shape)}')
        kernels.capture_layer[(1,)](self.live['h'], self.live['rl'], self.live['routes'], hidden,
                                    router_logits if keep else hidden, topk_ids, LAYER=layer_id, KEEP_RL=keep,
                                    D=D, D_POW2=triton.next_power_of_2(D), E=E, E_POW2=triton.next_power_of_2(E),
                                    K=K, K_POW2=triton.next_power_of_2(K))
        if self.learned is None:
            return
        main = torch.cuda.current_stream()
        L = self.cfg.base_model.num_layers
        if self.timing_all:
            kernels.stamp[(1,)](self.stamps, IDX=layer_id)
        if layer_id == 0:
            if self.shared is not None:
                # Before the fork, so the probe reads token t's residency (see the module docstring).
                self.resident.copy_(self.cache.slot_for_id >= 0)
            # The scores read only the frozen end-of-token-t inputs, never this capture.
            self.probe_stream.wait_stream(main)
            with torch.cuda.stream(self.probe_stream):
                self._score()
        if layer_id == self.learned.start_layer:
            if self.timing is not None:
                kernels.stamp[(1,)](self.stamps, IDX=L + 1)
            main.wait_event(self.scores_ready)
            if self.timing is not None:
                kernels.stamp[(1,)](self.stamps, IDX=L + 2)

    def _score(self) -> None:
        """Score every head from the end-of-token-t inputs and advance the EMA prior.

        At step 0 the inputs are the reset zeros and no token was decoded yet, so the
        prior must not move; the mask keeps it on the device.
        """
        raw, feats = self.raw, self.cfg.features
        if self.shared_fast is None:
            self._copy_raw()
        if self.learned is None:
            self.probs.copy_(self.bank(raw))
        elif self.shared is not None:
            self._score_shared()
            if self.shared_fast is not None:
                self._copy_raw()  # EXP586 F1: after scores-ready (module docstring)
            self.probs.copy_(self.bank(raw))
        else:
            g = self.bank.trunk(raw)
            keep = self.learned.keep
            self.evict(g, keep)
            # Step 0 scores come from the reset zeros (no token decoded yet): never valid. A
            # non-finite score would break the victim kernel's selection (learned_evict
            # contracts): that token falls back to LRU and is counted.
            finite = torch.isfinite(keep).all()
            scored = self.step[0] > 0
            self.learned.valid.copy_((scored & finite).to(torch.int32).view(1))
            self.nonfinite_tokens.add_((scored & ~finite).to(torch.int64))
            if self.timing is not None:
                kernels.stamp[(1,)](self.stamps, IDX=self.cfg.base_model.num_layers)
            self.scores_ready.record()
            self.probs.copy_(self.bank.head(g))
        prior = raw['prior_before']
        before = prior.clone()
        beta = feats.ema_beta
        prior.mul_(beta)
        prior.scatter_(1, raw['curr'], prior.gather(1, raw['curr']) + (1 - beta))
        prior.copy_(torch.where(self.step > 0, prior, before))
        if self.pick is not None:
            kind, seed = self.pick
            self.probs.copy_(baseline_scores(kind, prior, self.pick_ids, self.step, self.pick_requests, seed))

    def _copy_raw(self) -> None:
        """The prefetch bank's raw features of the target heads from the end-of-token-t inputs."""
        raw, inputs, feats = self.raw, self.score_inputs, self.cfg.features
        h, routes, prev, rl = inputs['h'], inputs['routes'], inputs['prev'], inputs['rl']
        lo, hi = self.first, self.first + self.heads
        o = self.offset
        raw['h'].copy_(h[lo + o:hi + o])
        raw['h_prev'].copy_(h[lo + o - 1:hi + o - 1])
        for d in range(1, feats.route_window + 1):
            raw['win'][:, d - 1].copy_(routes[lo - d:hi - d])
        for d in range(1, feats.previous_token_window + 1):
            raw['prevwin'][:, d - 1].copy_(prev[lo - d:hi - d])
        raw['curr'].copy_(routes[lo:hi])
        raw['rl'].copy_(torch.cat([rl[lo - d:hi - d] for d in range(1, feats.router_logit_window + 1)], 1))

    def _score_shared(self) -> None:
        """R584 keep scores of all layers (EXP585), published before the prefetch bank runs."""
        raw, inputs, keep = self.raw, self.score_inputs, self.learned.keep
        if self.shared_fast is not None:
            self.shared.score(inputs['routes'], self.step, self.resident, inputs['h'], raw['tid'],
                              raw['upcoming_tid'], raw['pos'][:1], self.bank.buffers['table'], keep,
                              self.learned.valid, self.nonfinite_tokens)
            if self.timing is not None:
                kernels.stamp[(1,)](self.stamps, IDX=self.cfg.base_model.num_layers)
            self.scores_ready.record()
            return
        self.shared.advance(inputs['routes'], self.step)
        keep.copy_(self.shared.scores(self.shared.own(self.resident), inputs['h'], raw['tid'], raw['upcoming_tid'],
                                      raw['pos'][:1], self.bank.buffers['table']))
        # As in the EvictRows path: step 0 is never valid; a non-finite token falls back to LRU and is counted.
        finite = torch.isfinite(keep).all()
        scored = self.step[0] > 0
        self.learned.valid.copy_((scored & finite).to(torch.int32).view(1))
        self.nonfinite_tokens.add_((scored & ~finite).to(torch.int64))
        if self.timing is not None:
            kernels.stamp[(1,)](self.stamps, IDX=self.cfg.base_model.num_layers)
        self.scores_ready.record()

    def copy_missing(self, cache: OffloadMoeCache, layer_id: int) -> bool:
        main = torch.cuda.current_stream()
        self._launch_demand(cache, layer_id)
        main.wait_stream(self.demand.stream)
        self._after_demand(cache, layer_id, lambda: self.probe_stream.wait_stream(main))
        return True

    def begin_copy(self, cache: OffloadMoeCache, layer_id: int):
        """copy_missing without the join (DecodeHooks.begin_copy, EXP538).

        In copy_missing the probe section waits for main after main joined the demand copy, so it
        depends on main's work before the copy plus the copy itself. Here main runs the resident
        routes between the launch and finish(), so the probe section waits for an event recorded
        on main right after the launch and for the demand stream: the same two dependencies, and
        none on the resident-route GEMMs. The last layer's probe join moves into finish(), so it
        still precedes every GEMM that reads a missing expert, and the token's next step.
        """
        main = torch.cuda.current_stream()
        self._launch_demand(cache, layer_id)
        before = self.copy_events[layer_id]
        before.record(main)

        def wait_inputs() -> None:
            self.probe_stream.wait_event(before)
            self.probe_stream.wait_stream(self.demand.stream)

        def finish() -> None:
            main.wait_stream(self.demand.stream)
            self._after_demand(cache, layer_id, wait_inputs)
        return finish

    def _launch_demand(self, cache: OffloadMoeCache, layer_id: int) -> None:
        """The L0 score (without learned eviction) and the BUSY-bracketed demand copy of the layer."""
        main = torch.cuda.current_stream()
        probe, demand = self.probe_stream, self.demand
        if layer_id == 0 and self.learned is None:  # with learned eviction, observe(0) scored
            probe.wait_stream(main)
            with torch.cuda.stream(probe):
                self._score()
        demand.stream.wait_stream(main)
        with torch.cuda.stream(demand.stream):
            kernels.phase[(1,)](self.busy, self.layer, cache.num_indices, L=layer_id)
            demand.launch(cache, layer_id, stages=self.stage_ptrs)
            kernels.phase[(1,)](self.busy, self.layer, cache.num_indices, L=layer_id, CLEAR=True)

    def _after_demand(self, cache: OffloadMoeCache, layer_id: int, wait_inputs) -> None:
        """The probe section of the layer (stage retire and select, pump) after ``wait_inputs()``
        made the probe stream depend on the demand copy, then the last layer's probe join."""
        main = torch.cuda.current_stream()
        probe, demand = self.probe_stream, self.demand
        sl = kernels.SECTION_LAYERS
        if layer_id < self.sections * sl:
            section = layer_id // sl
            wait_inputs()
            with torch.cuda.stream(probe):
                if layer_id % sl == 0:
                    kernels.retire[(1,)](demand.ids, demand.ready, SECTION=section, SLOTS=kernels.SLOTS)
                    E = self.cfg.base_model.num_experts
                    kernels.select_scored[(1,)](cache.slot_for_id, self.probs, self.step, self.records,
                                                self.cursors, self.payload, SECTION=section, FIRST=self.first,
                                                FIRST_STEP=self.cfg.serving_reference.first_prefetch_step,
                                                MIN_SCORE=MIN_SCORE, E=E,
                                                SPAN_POW2=triton.next_power_of_2(sl * E), SLOTS=kernels.SLOTS,
                                                SECTION_LAYERS=sl, STRIPES=kernels.PUMP_STRIPES, num_warps=8)
                kernels.publishing_pump[(kernels.PUMP_STRIPES,)](
                    self.hosts, self.stage_ptrs, self.feat_bytes, self.busy, self.layer, self.step,
                    self.records, self.cursors, self.payload, self.stats, demand.ids, demand.ready, demand.gen,
                    self.layer_published,
                    SECTION=section, CAP=self.caps[section],
                    FIRST_STEP=self.cfg.serving_reference.first_prefetch_step, NUM_BANKS=self.num_banks,
                    E=self.cfg.base_model.num_experts, QUOTA=self.quota, SLOTS=kernels.SLOTS,
                    SECTION_LAYERS=sl, STRIPES=kernels.PUMP_STRIPES, CHUNK_BYTES=kernels.CHUNK_BYTES,
                    COUNT_LAYERS=self.layer_counters, num_warps=8)
        if layer_id == self.cfg.base_model.num_layers - 1:
            if self.timing is not None:
                kernels.stamp[(1,)](self.join_stamps, IDX=0)
            main.wait_stream(probe)
            if self.timing is not None:
                kernels.stamp[(1,)](self.join_stamps, IDX=1)

    # -- reporting -----------------------------------------------------------------

    def _layer_counts(self) -> dict[str, list[int]]:
        """Per-layer counters of the current request (synchronizes the device). Published counts by target layer."""
        counters = self.demand.counters.tolist()
        return dict(published=self.layer_published.tolist(), from_stage=[c[0] for c in counters],
                    from_host=[c[1] for c in counters], host_while_pending=[c[2] for c in counters])

    def summary(self) -> dict[str, int]:
        """Counters of the current request (synchronizes the device)."""
        demand = self.demand.counters.sum(0).tolist()
        out = dict(zip(kernels.STATS_FIELDS, self.stats.tolist()))
        out.update(misses_from_stage=demand[0], misses_from_host=demand[1], host_misses_while_pending=demand[2],
                   unscored_token_ids=int(self.unscored_ids))
        if self.learned is not None:
            out.update(zip(('evict_' + c for c in EVICT_COUNTERS), self.learned.counters.tolist()))
            out['evict_nonfinite_tokens'] = int(self.nonfinite_tokens)
        return out


def create_decode_hooks(source: str, config: EngineConfig) -> PrefetchHooks:
    """Load, check and wrap the predictor named by --expert-predictor for this engine."""
    from freetoken.utils import cached_load_hf_config

    folder = hub.resolve(source)
    hub.verify(folder)
    cfg, state = bank_loader.load(folder)
    cfg.check_base(cached_load_hf_config(config.model_path, config.hf_overrides).to_dict())
    logger.info('expert predictor %s: trained on %s@%s; the served checkpoint is checked by shape only',
                source, cfg.base_model.repo_id, cfg.base_model.revision[:12])
    timing = os.environ.get(EVICT_TIMING_ENV)
    release = json.loads((folder / 'manifest.json').read_text())['files']['model.safetensors']  # verified above
    return PrefetchHooks(cfg, state, config.moe_demand_copy_stripes, evict_from_env(os.environ),
                         Path(timing) if timing else None, os.environ.get(EVICT_TIMING_ALL_ENV) == '1', release,
                         shared_evict_from_env(os.environ), shared_fast_from_env(os.environ),
                         pick_baseline_from_env(os.environ), os.environ.get(LAYER_COUNTERS_ENV) == '1')


def evict_from_env(env) -> tuple[Path, int] | None:
    """(bundle, J) when learned eviction is switched on, else None (the default LRU path)."""
    bundle = env.get(EVICT_ENV)
    if not bundle:
        return None
    start = env.get(EVICT_START_ENV)
    if start is None:
        raise ValueError(f'{EVICT_ENV} is set; {EVICT_START_ENV} must give the first learned layer')
    return Path(bundle), int(start)


def shared_evict_from_env(env) -> tuple[Path, int] | None:
    """(bundle, J) when shared learned eviction (EXP585) is switched on, else None."""
    bundle = env.get(SHARED_EVICT_ENV)
    if not bundle:
        return None
    if env.get(EVICT_ENV):
        raise ValueError(f'{EVICT_ENV} and {SHARED_EVICT_ENV} are two eviction rules; set one')
    start = env.get(EVICT_START_ENV)
    if start is None:
        raise ValueError(f'{SHARED_EVICT_ENV} is set; {EVICT_START_ENV} must give the first learned layer')
    return Path(bundle), int(start)


def shared_fast_from_env(env) -> str | None:
    """The F8 variant when fast shared learned eviction (EXP586) is switched on, else None."""
    variant = env.get(SHARED_EVICT_FAST_ENV)
    if not variant:
        return None
    if not env.get(SHARED_EVICT_ENV):
        raise ValueError(f'{SHARED_EVICT_FAST_ENV} is set; it needs {SHARED_EVICT_ENV}')
    if variant not in F8:
        raise ValueError(f'{SHARED_EVICT_FAST_ENV}={variant!r} is not one of {F8}')
    return variant
