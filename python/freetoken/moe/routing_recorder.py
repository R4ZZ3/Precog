"""Decode routing recorder for cache-replay experiments (research diagnostic, EXP519).

``RoutingRecorder`` wraps the engine's decode hooks and forwards every callback unchanged. In
addition it records, per decode step and MoE layer, the raw top-k expert ids before
``ensure_experts`` maps them to slots, and which of them were resident at that moment. At the
first decode step of a request it saves the expert-cache state after prefill (slot map, slot
reverse map, LRU usage stamps, the cache step clock). With a predictor it also saves the
predictor's scores of every step. When the next request begins, it writes one file per finished
request together with the live per-layer miss counters, which a replay must reproduce exactly.

The writes inside the decode graph use fixed buffers and a device row index, so they are
graph-safe. During graph warmup and capture, and for steps beyond the capacity, the row index
points to a scratch row that is never exported. The buffers are allocated in ``bind`` (after
the cache is sized), not in ``reserve``: memory charged in ``reserve`` would change the
automatic cache budget, and the point of the recorder is to observe the unchanged geometry.

Enabled by the environment variable FREETOKEN_ROUTING_RECORD_DIR; see engine._load_decode_hooks.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from freetoken.moe.decode_hooks import DecodeHooks

logger = logging.getLogger(__name__)


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

    # -- lifecycle (main stream, outside the graph) --------------------------------

    def reserve(self, engine) -> None:
        self.inner.reserve(engine)

    def bind(self, engine, cache) -> None:
        self.inner.bind(engine, cache)
        if self.cache is None:
            self._allocate(cache)
        elif self.cache is not cache:
            raise RuntimeError("routing recorder: the offload cache was rebuilt; the recorded geometry is no longer valid")
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
        after = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0
        logger.info("routing recorder: %.2f MiB of buffers allocated after cache sizing (%d slots), output %s",
                    (after - before) / 2**20, S, self.out_dir)

    def begin_request(self, uid: int) -> None:
        if uid != self.uid:
            if self.steps:
                self._export()  # before inner.begin_request resets the live counters
            self.uid, self.steps, self.overflow = uid, 0, False
            self.ids.fill_(-1)
            self.hits.zero_()
            if self.scores is not None:
                self.scores.zero_()
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
        self.inner.after_decode(step, consumed, sampled)
        self.steps = step + 1

    # -- inside the decode graph ---------------------------------------------------

    def observe(self, layer_id: int, hidden: torch.Tensor, router_logits: torch.Tensor | None,
                topk_ids: torch.Tensor) -> None:
        self.record(layer_id, topk_ids)
        self.inner.observe(layer_id, hidden, router_logits, topk_ids)

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

    # -- export --------------------------------------------------------------------

    def _export(self) -> None:
        n = min(self.steps, self.capacity)
        demand = getattr(self.inner, "demand", None)
        record = {
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
        path = self.out_dir / f"request_{self.sequence:03d}_uid{self.uid}.pt"
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
