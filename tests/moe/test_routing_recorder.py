"""CPU tests of the routing recorder (freetoken.moe.routing_recorder).

A fake cache and fake inner hooks stand in for the engine. The tests drive the callbacks in the
engine's order: bind, graph warmup observes, begin_request (prefill), before_decode / observe /
after_decode per step, then the next request's begin_request, which exports the finished one.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.decode_hooks import DecodeHooks
from freetoken.moe.routing_recorder import RoutingRecorder

L, E, K, S = 3, 16, 2, 10


def make_cache():
    slot_for_id = torch.full((L, E), -1, dtype=torch.int32)
    slot_for_id[0, 1] = 4
    slot_for_id[1, 5] = 7
    return SimpleNamespace(num_layers=L, num_experts=E, cache_size=S, slot_for_id=slot_for_id,
                           id_of_slot=torch.arange(S, dtype=torch.int32), usage=torch.arange(S, dtype=torch.int64),
                           step=torch.tensor(42, dtype=torch.int64))


class FakeInner(DecodeHooks):
    """Logs every call; resets its counters in begin_request as the real hooks do."""

    def __init__(self, with_probs: bool):
        self.calls = []
        self.demand = SimpleNamespace(counters=torch.zeros((L, 3), dtype=torch.int64))
        self.probs = torch.zeros((2, E)) if with_probs else None
        if not with_probs:
            del self.probs

    def reserve(self, engine):
        self.calls.append(("reserve",))

    def bind(self, engine, cache):
        self.calls.append(("bind",))

    def begin_request(self, uid):
        self.calls.append(("begin_request", uid))
        self.demand.counters.zero_()

    def before_decode(self, step):
        self.calls.append(("before_decode", step))

    def after_decode(self, step, consumed, sampled):
        self.calls.append(("after_decode", step))

    def observe(self, layer_id, hidden, router_logits, topk_ids):
        self.calls.append(("observe", layer_id))

    def copy_missing(self, cache, layer_id):
        self.calls.append(("copy_missing", layer_id))
        return True


def ids(*values):
    return torch.tensor([values], dtype=torch.int64)


def decode_step(rec, step, routes, inner=None, probs_value=None):
    rec.before_decode(step)
    if probs_value is not None:
        inner.probs.fill_(probs_value)
    for layer, route in enumerate(routes):
        rec.observe(layer, torch.zeros(1, 4), None, ids(*route))
        assert rec.copy_missing(rec.cache, layer) is True
    rec.after_decode(step, torch.zeros(1), torch.zeros(1))


def files(path):
    return sorted(path.glob("request_*.pt"))


def test_records_raw_ids_residency_snapshot_and_counters(tmp_path):
    inner, cache = FakeInner(with_probs=False), make_cache()
    rec = RoutingRecorder(inner, tmp_path, top_k=K, capacity=4)
    rec.reserve(None)
    rec.bind(None, cache)
    # graph warmup and capture: written to the scratch row only
    rec.observe(0, torch.zeros(1, 4), None, ids(9, 9))
    rec.begin_request(7)
    rec.begin_request(7)  # a second prefill chunk of the same request: no export, no reset
    decode_step(rec, 0, [(1, 2), (5, 6), (3, 4)])
    # the cache changes after step 0: the snapshot must keep the post-prefill state
    cache.slot_for_id[0, 2] = 3
    cache.usage.add_(100)
    decode_step(rec, 1, [(2, 1), (0, 5), (3, 4)])
    inner.demand.counters[1, 1] = 3  # live counters of request 7, reset by the next begin_request
    rec.begin_request(8)

    [path] = files(tmp_path)
    record = torch.load(path)
    assert (record["uid"], record["sequence"], record["steps"], record["overflow"]) == (7, 0, 2, False)
    assert record["schema"] == 1 and record["geometry"] == dict(layers=L, experts=E, top_k=K, slots=S, capacity=4)
    assert record["ids"].tolist() == [[[1, 2], [5, 6], [3, 4]], [[2, 1], [0, 5], [3, 4]]]
    assert record["hits"].tolist() == [[[True, False], [True, False], [False, False]],
                                       [[True, True], [False, True], [False, False]]]
    assert record["usage"].tolist() == list(range(S))
    assert record["slot_for_id"][0, 2].item() == -1 and record["cache_step"] == 42
    assert record["live_counters"][1, 1].item() == 3
    assert "scores" not in record
    # every callback was forwarded, in order, and the inner reset ran after the export
    assert inner.calls[:4] == [("reserve",), ("bind",), ("observe", 0), ("begin_request", 7)]
    assert inner.calls[-1] == ("begin_request", 8) and inner.demand.counters.sum().item() == 0


def test_warmup_rows_never_exported_and_new_request_starts_clean(tmp_path):
    inner, cache = FakeInner(with_probs=False), make_cache()
    rec = RoutingRecorder(inner, tmp_path, top_k=K, capacity=4)
    rec.bind(None, cache)
    rec.begin_request(1)
    decode_step(rec, 0, [(1, 1), (1, 1), (1, 1)])
    rec.observe(2, torch.zeros(1, 4), None, ids(15, 15))  # a stray capture-time write
    assert rec.ids[-1, 2].tolist() == [15, 15]  # went to the scratch row
    rec.begin_request(2)
    decode_step(rec, 0, [(3, 3), (3, 3), (3, 3)])
    rec.begin_request(3)
    first, second = (torch.load(p) for p in files(tmp_path))
    assert first["ids"].tolist() == [[[1, 1]] * 3]
    assert second["ids"].tolist() == [[[3, 3]] * 3]


def test_overflow_steps_go_to_scratch_and_are_flagged(tmp_path):
    inner, cache = FakeInner(with_probs=True), make_cache()
    rec = RoutingRecorder(inner, tmp_path, top_k=K, capacity=2)
    rec.bind(None, cache)
    rec.begin_request(1)
    for step in range(3):
        decode_step(rec, step, [(step, step)] * L, inner, probs_value=float(step + 1))
    rec.begin_request(2)
    record = torch.load(files(tmp_path)[0])
    assert record["overflow"] is True and record["steps"] == 3
    assert record["ids"].shape == (2, L, K) and record["ids"][:, 0, 0].tolist() == [0, 1]
    assert record["scores"][:, 0, 0].tolist() == [1.0, 2.0]


def test_rebind_before_recording_reallocates_and_after_recording_is_refused(tmp_path):
    inner = FakeInner(with_probs=False)
    rec = RoutingRecorder(inner, tmp_path, top_k=K)
    rec.bind(None, make_cache())
    bigger = make_cache()
    bigger.cache_size, bigger.id_of_slot, bigger.usage = 12, torch.arange(12, dtype=torch.int32), torch.zeros(12, dtype=torch.int64)
    rec.bind(None, bigger)  # startup recapture: no request recorded yet
    assert rec.snap_usage.shape == (12,)
    rec.begin_request(1)
    binds = inner.calls.count(("bind",))
    with pytest.raises(RuntimeError, match="rebound"):
        rec.bind(None, bigger)
    assert inner.calls.count(("bind",)) == binds  # refused before the inner bind ran


def test_wrong_routed_shape_is_refused(tmp_path):
    rec = RoutingRecorder(FakeInner(with_probs=False), tmp_path, top_k=K)
    rec.bind(None, make_cache())
    with pytest.raises(RuntimeError, match="routed ids"):
        rec.observe(0, torch.zeros(2, 4), None, torch.zeros((2, K), dtype=torch.int64))


def test_profile_window_opens_and_closes_around_chosen_requests(monkeypatch):
    from freetoken.moe.routing_recorder import ProfileWindow

    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda.profiler, "start", lambda: events.append("start"))
    monkeypatch.setattr(torch.cuda.profiler, "stop", lambda: events.append("stop"))
    inner = FakeInner(with_probs=False)
    window = ProfileWindow(inner, frozenset({1, 2}))
    window.begin_request(9)  # a prefill-only warmup: never decodes, never counted
    for uid in (10, 11, 12, 13):
        window.begin_request(uid)
        window.begin_request(uid)  # a second prefill chunk of the same request
        window.before_decode(0)
        window.before_decode(1)
        events.append(f"uid{uid}")
    assert events == ["uid10", "start", "uid11", "stop", "start", "uid12", "stop", "uid13"]
    assert window.copy_missing(None, 0) is True and inner.calls[-1] == ("copy_missing", 0)
