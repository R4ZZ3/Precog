"""Decode routing recorder for cache-replay experiments (research diagnostic, EXP519).

``RoutingRecorder`` wraps the engine's decode hooks and forwards every callback unchanged. In
addition it records, per decode step and MoE layer, the raw top-k expert ids before
``ensure_experts`` maps them to slots, and which of them were resident at that moment. At the
first decode step of a request it saves the expert-cache state after prefill (slot map, slot
reverse map, LRU usage stamps, the cache step clock). With a predictor it also saves the
predictor's scores of every step, and with learned eviction (EXP526) the keep scores and their
valid flag of every step. With shared learned eviction (EXP585/586, R584) it also saves the P4 audit
rows of every step (see _allocate_p4). With the prefetch hooks it also saves, after every step, the selected
experts of each section, the pump's per-slot payload and the cumulative pump and demand counters. When the next request begins, it writes one file per finished
request together with the live per-layer miss counters, which a replay must reproduce exactly.

The writes inside the decode graph use fixed buffers and a device row index, so they are
graph-safe. During graph warmup and capture, and for steps beyond the capacity, the row index
points to a scratch row that is never exported. The buffers are allocated in ``bind`` (after
the cache is sized), not in ``reserve``: memory charged in ``reserve`` would change the
automatic cache budget, and the point of the recorder is to observe the unchanged geometry.

Enabled by the environment variable FREETOKEN_ROUTING_RECORD_DIR; see engine._load_decode_hooks.
"""
from __future__ import annotations

from pathlib import Path

import torch

from freetoken.moe.decode_hooks import DecodeHooks
from freetoken.utils import init_logger

logger = init_logger(__name__)

SCHEMA = 1
P4_SCHEMA = 1
# scores[r] are the predictor scores the forward of decode step r used: computed at its layer 0
# from the inputs frozen after step r - 1. Row 0 scores reset (zero) inputs and forecasts nothing.
SCORE_CLOCK = "scores[r]: used by decode step r, computed from inputs frozen after step r-1; row 0 has reset inputs"


class RoutingRecorder(DecodeHooks):
    """Forward every callback to ``inner`` and record the decode routing of each request."""

    single_request = True  # rows are indexed by the decode step of one request

    def __init__(self, inner: DecodeHooks, out_dir: str | Path, top_k: int, capacity: int = 256):
        if capacity < 1 or top_k < 1:
            raise ValueError(f"capacity {capacity} and top_k {top_k} must be positive")
        self.inner = inner
        self.top_k = top_k
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.cache = None
        self.uid: int | None = None
        self.steps = 0
        self.overflow = False
        self.sequence = 0  # number of request files written
        self.p4: dict[str, torch.Tensor] | None = None  # set in bind (_allocate_p4)

    # -- lifecycle (main stream, outside the graph) --------------------------------

    def reserve(self, engine) -> None:
        self.inner.reserve(engine)

    def bind(self, engine, cache) -> None:
        # A rebind means a graph recapture, maybe after a cache rebuild, which keeps the cache
        # object but replaces its tensors and can change its size. Once a request was recorded,
        # the recorded geometry would no longer describe the cache, and the inner bind would reset
        # the live counters before the export: refuse. Before that, reallocate for the new geometry.
        # The check runs before inner.bind, so that the inner hooks see no partial bind.
        if self.uid is not None:
            raise RuntimeError("routing recorder: the engine rebound the hooks (graph recapture or cache rebuild) "
                               "after recording started; the recording is no longer valid")
        self.inner.bind(engine, cache)
        self._allocate(cache)
        self.row.fill_(self.capacity)  # graph warmup and capture write to the scratch row

    def _allocate(self, cache) -> None:
        dev = cache.slot_for_id.device
        before = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0
        L, E, C, S = cache.num_layers, cache.num_experts, self.capacity, cache.cache_size
        self.cache = cache
        self.row = torch.full((1,), C, dtype=torch.int64, device=dev)
        # one extra row: the scratch row for warmup, capture and overflow
        self.ids = torch.full((C + 1, L, self.top_k), -1, dtype=torch.int32, device=dev)
        self.hits = torch.zeros((C + 1, L, self.top_k), dtype=torch.bool, device=dev)
        self.snap_slot_for_id = torch.empty((L, E), dtype=torch.int32, device=dev)
        self.snap_id_of_slot = torch.empty((S,), dtype=torch.int32, device=dev)
        self.snap_usage = torch.empty((S,), dtype=torch.int64, device=dev)
        self.snap_step = torch.empty((), dtype=torch.int64, device=dev)
        probs = getattr(self.inner, "probs", None)
        self.scores = None if probs is None else torch.zeros((C,) + tuple(probs.shape), dtype=probs.dtype, device=dev)
        learned = getattr(self.inner, "learned", None)
        self.keep = None if learned is None else torch.zeros((C,) + tuple(learned.keep.shape), dtype=learned.keep.dtype,
                                                             device=dev)
        self.keep_valid = None if learned is None else torch.zeros((C,), dtype=torch.int32, device=dev)
        # Prefetch state after each step (EXP526 step 3): the selected gids of every section, the
        # pump's per-slot payload counters, and the cumulative pump stats and demand counters. With
        # these a replay compares its selections and stage/host misses step by step, not only per
        # request. Copied in after_decode, outside the graph, as the scores.
        self.prefetch = {}
        if getattr(self.inner, "records", None) is not None:
            for name, src in (("selections", self.inner.records), ("payload", self.inner.payload),
                              ("pump_stats", self.inner.stats), ("demand_counters", self.inner.demand.counters)):
                self.prefetch[name] = (src, torch.zeros((C,) + tuple(src.shape), dtype=src.dtype, device=dev))
        self.p4 = self._allocate_p4(cache, dev)
        after =torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0
        logger.info("routing recorder: %.2f MiB of buffers allocated after cache sizing (%d slots), output %s",
                    (after - before) / 2**20, S, self.out_dir)

    def _allocate_p4(self, cache, dev) -> dict[str, torch.Tensor] | None:
        """P4 audit rows of shared learned eviction (EXP586 step D), or None in every other mode.

        Two rows are written inside the decode graph, at the step's row index like ``ids`` (the
        extra row C is the scratch row):
          direct    (C+1, L, E) uint8: slot_for_id >= 0 read in observe(0) before the inner hook, so at
                    the same main-stream point where the inner hook snapshots its resident mask.
          producer  (C+1, n_h, D) FP32: the hidden argument of observe() at R584's hidden-source
                    layers. FP32 holds bf16/fp16 values exactly; p4_producer_dtype names the source.
        The rest is copied in after_decode before the inner after_decode overwrites the frozen
        inputs, so each row holds what this step's scorer read and the state it left:
          resident (uint8, plus resident_nonbinary: entries not 0 or 1), ema (FP32), age (int16),
          routes (score_inputs routes), h (score_inputs h at the hidden-source layers, fp16), tid,
          upcoming_tid, pos, and the step's consumed and sampled ids.
        """
        shared = getattr(self.inner, "shared", None)
        if shared is None:
            return None
        L, E, C, K = cache.num_layers, cache.num_experts, self.capacity, self.top_k
        far = int(shared.far)
        if not 0 <= far <= torch.iinfo(torch.int16).max:
            raise ValueError(f"routing recorder: R584 age index range 0..{far} does not fit int16")
        self.p4_layers = [int(v) for v in shared.hidden_layers.tolist()]
        self.p4_producer_dtype = None
        D = self.inner.score_inputs["h"].shape[1]
        n_h = len(self.p4_layers)
        z = lambda *shape, dtype: torch.zeros(shape, dtype=dtype, device=dev)  # noqa: E731
        return dict(direct=z(C + 1, L, E, dtype=torch.uint8), producer=z(C + 1, n_h, D, dtype=torch.float32),
                    resident=z(C, L, E, dtype=torch.uint8), resident_nonbinary=z(C, dtype=torch.int32),
                    ema=z(C, L, E, dtype=torch.float32), age=z(C, L, E, dtype=torch.int16),
                    routes=z(C, L, K, dtype=torch.int64), h=z(C, n_h, D, dtype=torch.float16),
                    tid=z(C, dtype=torch.int64), upcoming_tid=z(C, dtype=torch.int64), pos=z(C, dtype=torch.float32),
                    consumed=z(C, dtype=torch.int64), sampled=z(C, dtype=torch.int64))

    def begin_request(self, uid: int) -> None:
        if uid != self.uid:
            if self.steps:
                self._export()  # before inner.begin_request resets the live counters
            self.uid, self.steps, self.overflow = uid, 0, False
            self.ids.fill_(-1)
            self.hits.zero_()
            if self.scores is not None:
                self.scores.zero_()
            if self.keep is not None:
                self.keep.zero_()
                self.keep_valid.zero_()
            for _, rows in self.prefetch.values():
                rows.zero_()
            for rows in (self.p4 or {}).values():
                rows.zero_()
        self.row.fill_(self.capacity)
        self.inner.begin_request(uid)

    def before_decode(self, step: int) -> None:
        if step == 0:
            cache = self.cache
            self.snap_slot_for_id.copy_(cache.slot_for_id)
            self.snap_id_of_slot.copy_(cache.id_of_slot)
            self.snap_usage.copy_(cache.usage)
            self.snap_step.copy_(cache.step)
        if step >= self.capacity:
            self.overflow = True
        self.row.fill_(min(step, self.capacity))
        self.inner.before_decode(step)

    def after_decode(self, step: int, consumed: torch.Tensor, sampled: torch.Tensor) -> None:
        # The decode graph joined the predictor's probe stream into main before its end, so
        # these are the scores this step's forward used (computed at its layer 0).
        if self.scores is not None and step < self.capacity:
            self.scores[step].copy_(self.inner.probs)
        if self.keep is not None and step < self.capacity:
            # Published at this step's layer 0 and read by its layers >= start_layer.
            self.keep[step].copy_(self.inner.learned.keep)
            self.keep_valid[step].copy_(self.inner.learned.valid[0])
        if step < self.capacity:
            # The graph ended with main waiting for the probe and demand streams, so these hold
            # this step's final selections and counts. The counters are cumulative over the
            # request: a replay takes differences of consecutive rows.
            for src, rows in self.prefetch.values():
                rows[step].copy_(src)
        if self.p4 is not None and step < self.capacity:
            self._record_p4(step, consumed, sampled)
        self.inner.after_decode(step, consumed, sampled)
        self.steps = step + 1
        # Only observes between before_decode and after_decode write a real row; any other
        # observe (a stray capture or warmup forward) goes to the scratch row.
        self.row.fill_(self.capacity)

    def _record_p4(self, step: int, consumed: torch.Tensor, sampled: torch.Tensor) -> None:
        """The scorer's inputs and state of this step, before inner.after_decode freezes the next ones."""
        inner, p4, shared = self.inner, self.p4, self.inner.shared
        resident = inner.resident
        p4["resident"][step].copy_(resident.to(torch.uint8))
        p4["resident_nonbinary"][step].copy_(((resident != 0) & (resident != 1)).sum().to(torch.int32))
        p4["ema"][step].copy_(shared.ema)
        p4["age"][step].copy_(shared.age.to(torch.int16))
        p4["routes"][step].copy_(inner.score_inputs["routes"])
        p4["h"][step].copy_(inner.score_inputs["h"][self.p4_layers])
        for name in ("tid", "upcoming_tid"):
            p4[name][step].copy_(inner.raw[name][0])
        p4["pos"][step].copy_(inner.raw["pos"][0, 0])
        p4["consumed"][step].copy_(consumed.reshape(-1)[0].to(torch.int64))
        p4["sampled"][step].copy_(sampled.reshape(-1)[0].to(torch.int64))

    # -- inside the decode graph ---------------------------------------------------

    def observe(self, layer_id: int, hidden: torch.Tensor, router_logits: torch.Tensor | None,
                topk_ids: torch.Tensor) -> None:
        self.record(layer_id, topk_ids)
        if self.p4 is not None:
            self._observe_p4(layer_id, hidden)
        self.inner.observe(layer_id, hidden, router_logits, topk_ids)

    def _observe_p4(self, layer_id: int, hidden: torch.Tensor) -> None:
        """In-graph P4 rows; runs before the inner observe (see _allocate_p4)."""
        p4 = self.p4
        if layer_id == 0:
            direct = (self.cache.slot_for_id >= 0).to(torch.uint8)
            p4["direct"].index_copy_(0, self.row, direct.unsqueeze(0))
        if layer_id in self.p4_layers:
            D = p4["producer"].shape[-1]
            if hidden.shape != (1, D):
                raise RuntimeError(f"routing recorder: hidden {tuple(hidden.shape)} at layer {layer_id}, expected (1, {D})")
            if self.p4_producer_dtype not in (None, str(hidden.dtype)):
                raise RuntimeError(f"routing recorder: hidden dtype {hidden.dtype}, earlier {self.p4_producer_dtype}")
            self.p4_producer_dtype = str(hidden.dtype)
            j = self.p4_layers.index(layer_id)
            p4["producer"][:, j].index_copy_(0, self.row, hidden.float())

    def record(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Store this layer's raw ids and their residency in the current row (device ops only)."""
        k = self.ids.shape[-1]
        if topk_ids.shape != (1, k):
            raise RuntimeError(f"routing recorder: routed ids {tuple(topk_ids.shape)}, expected (1, {k})")
        ids = topk_ids.reshape(k).to(torch.int64)
        resident = self.cache.slot_for_id[layer_id].index_select(0, ids) >= 0
        self.ids[:, layer_id].index_copy_(0, self.row, ids.to(torch.int32).view(1, k))
        self.hits[:, layer_id].index_copy_(0, self.row, resident.view(1, k))

    def copy_missing(self, cache, layer_id: int) -> bool:
        return self.inner.copy_missing(cache, layer_id)

    def begin_copy(self, cache, layer_id: int):
        return self.inner.begin_copy(cache, layer_id)

    # -- export --------------------------------------------------------------------

    def _export(self) -> None:
        n = min(self.steps, self.capacity)
        demand = getattr(self.inner, "demand", None)
        cache = self.cache
        record = {
            "schema": SCHEMA,
            "geometry": dict(layers=cache.num_layers, experts=cache.num_experts, top_k=self.top_k,
                             slots=cache.cache_size, capacity=self.capacity),
            "score_clock": SCORE_CLOCK,
            "uid": self.uid,
            "sequence": self.sequence,
            "steps": self.steps,
            "overflow": self.overflow,
            "ids": self.ids[:n].cpu(),
            "hits": self.hits[:n].cpu(),
            "slot_for_id": self.snap_slot_for_id.cpu(),
            "id_of_slot": self.snap_id_of_slot.cpu(),
            "usage": self.snap_usage.cpu(),
            "cache_step": int(self.snap_step.item()),
            # (layers, 3): misses served from the stage, from the host, from the host while pending
            "live_counters": None if demand is None else demand.counters.cpu(),
        }
        if self.scores is not None:
            record["scores"] = self.scores[:n].cpu()
        if self.keep is not None:
            learned = self.inner.learned
            record.update(keep=self.keep[:n].cpu(), keep_valid=self.keep_valid[:n].cpu(),
                          evict_start_layer=learned.start_layer, evict_counters=learned.counters.cpu())
            if self.p4 is None:  # EvictRows (EXP526): its rows and their source layers
                record.update(evict_mode="rows", evict_layers=list(self.inner.evict.layers),
                              evict_source_layers=list(self.inner.evict.source_layers))
            else:  # shared R584 scores every layer; its hidden features come from p4_layers
                record.update(evict_mode="shared", evict_layers=list(range(cache.num_layers)),
                              shared_hidden_layers=list(self.p4_layers), p4_schema=P4_SCHEMA,
                              p4_producer_dtype=self.p4_producer_dtype,
                              **{"p4_" + name: rows[:n].cpu() for name, rows in self.p4.items()})
        if self.prefetch:
            # pump_stats columns: commitmoe_prefetch.kernels.STATS_FIELDS; demand_counters columns:
            # stage misses, host misses, host misses while pending (as live_counters).
            record.update({name: rows[:n].cpu() for name, (_, rows) in self.prefetch.items()})
        path =self.out_dir / f"request_{self.sequence:03d}_uid{self.uid}.pt"
        torch.save(record, path)
        self.sequence += 1



class ProfileWindow(DecodeHooks):
    """Forward every callback to ``inner``; profile only the chosen requests (EXP519 part A).

    Requests are counted when they reach decode step 0 (0 = the first request that decodes), so
    a prefill-only warmup can never shift the count. The profiler range (cudaProfilerStart/Stop,
    for ``nsys --capture-range=cudaProfilerApi``) covers the decode of a chosen request: it opens
    before its step 0 and closes when the next request begins. The wrapper adds no device work:
    graph replays do not call Python.
    """

    def __init__(self, inner: DecodeHooks, requests: frozenset[int]):
        self.inner = inner
        self.single_request = inner.single_request
        self.requests = requests
        self.uid: int | None = None
        self.index = -1
        self.open = False

    def reserve(self, engine) -> None:
        self.inner.reserve(engine)

    def bind(self, engine, cache) -> None:
        self.inner.bind(engine, cache)

    def begin_request(self, uid: int) -> None:
        if uid != self.uid:
            self.uid = uid
            if self.open:
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
                self.open = False
        self.inner.begin_request(uid)

    def before_decode(self, step: int) -> None:
        if step == 0:
            self.index += 1
            if self.index in self.requests:
                torch.cuda.synchronize()
                torch.cuda.profiler.start()
                self.open = True
                logger.info("profile window: request index %d (uid %s) opened", self.index, self.uid)
        self.inner.before_decode(step)

    def after_decode(self, step: int, consumed: torch.Tensor, sampled: torch.Tensor) -> None:
        self.inner.after_decode(step, consumed, sampled)

    def observe(self, layer_id: int, hidden: torch.Tensor, router_logits: torch.Tensor | None,
                topk_ids: torch.Tensor) -> None:
        self.inner.observe(layer_id, hidden, router_logits, topk_ids)

    def copy_missing(self, cache, layer_id: int) -> bool:
        return self.inner.copy_missing(cache, layer_id)

    def begin_copy(self, cache, layer_id: int):
        return self.inner.begin_copy(cache, layer_id)
