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


def test_prefetch_state_saved_per_step_and_reset_per_request(tmp_path):
    inner, cache = FakeInner(with_probs=True), make_cache()
    # the prefetch hooks' buffers: selections (sections, slots), payload, pump stats, demand counters
    inner.records = torch.full((2, 3), -1, dtype=torch.int64)
    inner.payload = torch.zeros((2, 3), dtype=torch.int32)
    inner.stats = torch.zeros(4, dtype=torch.int64)
    rec = RoutingRecorder(inner, tmp_path, top_k=K, capacity=2)
    rec.bind(None, cache)
    rec.begin_request(1)
    for step in range(3):  # step 2 overflows the capacity: never exported
        rec.before_decode(step)
        inner.records[0, 0] = 100 + step
        inner.payload[1, 2] = step
        inner.stats[3] += 1
        inner.demand.counters[2, 0] += step
        for layer in range(L):
            rec.observe(layer, torch.zeros(1, 4), None, ids(0, 1))
        rec.after_decode(step, torch.zeros(1), torch.zeros(1))
    rec.begin_request(2)
    inner.records.fill_(-1)  # the real hooks reset their buffers in begin_request
    decode_step(rec, 0, [(0, 1)] * L)
    rec.begin_request(3)
    first, second = (torch.load(p) for p in files(tmp_path))
    assert first["selections"][:, 0, 0].tolist() == [100, 101] and first["selections"].shape == (2, 2, 3)
    assert first["payload"][:, 1, 2].tolist() == [0, 1]
    assert first["pump_stats"][:, 3].tolist() == [1, 2]  # cumulative, as the live stats
    assert first["demand_counters"][:, 2, 0].tolist() == [0, 1] and first["demand_counters"].shape == (2, L, 3)
    # the second request starts from zeroed rows, not from the first request's last values
    assert second["selections"].tolist() == [[[-1, -1, -1], [-1, -1, -1]]]
    assert second["pump_stats"].tolist() == [[0, 0, 0, 3]]  # FakeInner does not reset its stats


def test_no_prefetch_fields_without_prefetch_hooks(tmp_path):
    rec = RoutingRecorder(FakeInner(with_probs=False), tmp_path, top_k=K)
    rec.bind(None, make_cache())
    rec.begin_request(1)
    decode_step(rec, 0, [(0, 1)] * L)
    rec.begin_request(2)
    record = torch.load(files(tmp_path)[0])
    assert not {"selections", "payload", "pump_stats", "demand_counters"} & record.keys()


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


@pytest.mark.parametrize("record,profile", [(False, False), (True, False), (False, True), (True, True)])
def test_engine_wraps_hooks_for_recording_and_profiling(tmp_path, monkeypatch, record, profile):
    from freetoken.engine import engine
    from freetoken.moe.routing_recorder import ProfileWindow

    base = FakeInner(with_probs=False)
    monkeypatch.setattr(engine, "_build_decode_hooks", lambda config: base)
    monkeypatch.delenv("FREETOKEN_ROUTING_RECORD_DIR", raising=False)
    monkeypatch.delenv("FREETOKEN_PROFILE_REQUESTS", raising=False)
    if record:
        monkeypatch.setenv("FREETOKEN_ROUTING_RECORD_DIR", str(tmp_path))
    if profile:
        monkeypatch.setenv("FREETOKEN_PROFILE_REQUESTS", "2,11")
    config = SimpleNamespace(model_config=SimpleNamespace(num_experts_per_tok=K))
    hooks = engine._load_decode_hooks(config)
    # with both, the window is outermost and the recorder sits directly on the real hooks
    chain = []
    while hooks is not base:
        chain.append(type(hooks))
        hooks = hooks.inner
    want = ([ProfileWindow] if profile else []) + ([RoutingRecorder] if record else [])
    assert chain == want
    if profile:
        outer = engine._load_decode_hooks(config)
        assert outer.requests == frozenset({2, 11})


# -- shared learned eviction (EXP586 step D, P4 audit rows) ------------------------------------------

D = 4


class FakeShared(FakeInner):
    """Shared-mode inner hooks: observe(0) snapshots residency like the real hook and then mutates the
    cache (as a later ensure would); after_decode overwrites the frozen inputs like the real hook."""

    def __init__(self, cache, far=7, hidden_layers=(1, 2)):
        super().__init__(with_probs=False)
        self.cache = cache
        self.shared = SimpleNamespace(far=far, hidden_layers=torch.tensor(hidden_layers),
                                      ema=torch.zeros(L, E), age=torch.full((L, E), far, dtype=torch.int64))
        self.learned = SimpleNamespace(keep=torch.zeros(L, E), valid=torch.zeros(1, dtype=torch.int32),
                                       start_layer=1, counters=torch.zeros(3, dtype=torch.int64))
        self.resident = torch.zeros(L, E)
        self.score_inputs = dict(h=torch.zeros(L, D, dtype=torch.float16), routes=torch.zeros(L, K, dtype=torch.int64))
        self.raw = dict(tid=torch.zeros(1, dtype=torch.int64), upcoming_tid=torch.zeros(1, dtype=torch.int64),
                        pos=torch.zeros(1, 1))
        self.live_h = torch.zeros(L, D, dtype=torch.float16)
        self.mutations = 0

    def observe(self, layer_id, hidden, router_logits, topk_ids):
        super().observe(layer_id, hidden, router_logits, topk_ids)
        if layer_id == 0:
            self.resident.copy_(self.cache.slot_for_id >= 0)
            self.cache.slot_for_id[2, 9 + self.mutations] = 1  # a mutation after the snapshot point
            self.mutations += 1
        self.live_h[layer_id] = hidden.float().half()[0]

    def begin_request(self, uid):
        super().begin_request(uid)
        for t in (*self.score_inputs.values(), *self.raw.values()):  # the real hook zeroes these too
            t.zero_()

    def after_decode(self, step, consumed, sampled):
        super().after_decode(step, consumed, sampled)
        self.score_inputs["h"].copy_(self.live_h)
        self.score_inputs["routes"].fill_(step + 100)
        self.raw["tid"].fill_(int(consumed))
        self.raw["upcoming_tid"].fill_(int(sampled))
        self.raw["pos"].fill_((step + 1) / 256)


def shared_step(rec, inner, step, token):
    rec.before_decode(step)
    inner.shared.ema.fill_(0.5 + step)
    inner.shared.age.fill_(step)
    for layer in range(L):
        rec.observe(layer, torch.full((1, D), layer + step + 0.3, dtype=torch.bfloat16), None, ids(layer, layer + 1))
    rec.after_decode(step, torch.tensor([token]), torch.tensor([token + 1]))


def test_shared_mode_exports_p4_rows_with_their_clock(tmp_path):
    cache = make_cache()
    inner = FakeShared(cache)
    rec = RoutingRecorder(inner, tmp_path, top_k=K, capacity=4)
    rec.bind(None, cache)
    rec.observe(0, torch.zeros(1, D, dtype=torch.bfloat16), None, ids(9, 9))  # capture: scratch row only
    rec.begin_request(5)
    shared_step(rec, inner, 0, 40)
    shared_step(rec, inner, 1, 41)
    rec.begin_request(6)
    record = torch.load(files(tmp_path)[0])
    assert record["evict_mode"] == "shared" and record["evict_layers"] == list(range(L))
    assert record["shared_hidden_layers"] == [1, 2] and "evict_source_layers" not in record
    assert record["p4_schema"] == 1 and record["p4_producer_dtype"] == "torch.bfloat16"
    # direct: read before the inner observe(0), so step 0's mutation (at [2, 10]; the capture call made
    # [2, 9]) is missing at step 0 and present at step 1 ...
    assert record["p4_direct"][:, 2, 9:12].tolist() == [[1, 0, 0], [1, 1, 0]]
    assert record["p4_direct"][0, 0, 1].item() == 1
    # ... and it equals the inner hook's own snapshot at both steps
    assert torch.equal(record["p4_direct"], record["p4_resident"])
    assert record["p4_resident_nonbinary"].tolist() == [0, 0]
    # producer: layers 1 and 2 only, exact bf16 values in FP32
    want = torch.tensor([[1.3, 2.3], [2.3, 3.3]], dtype=torch.bfloat16).float()
    assert torch.equal(record["p4_producer"][:, :, 0], want)
    # the frozen inputs this step's scorer read: reset zeros at step 0, step 0's values at step 1
    assert record["p4_h"][0].abs().sum().item() == 0
    assert torch.equal(record["p4_h"][1], record["p4_producer"][0].half())
    assert record["p4_routes"][:, 0, 0].tolist() == [0, 100]
    assert record["p4_tid"].tolist() == [0, 40] and record["p4_upcoming_tid"].tolist() == [0, 41]
    assert record["p4_pos"].tolist() == [0.0, 1 / 256]
    assert record["p4_consumed"].tolist() == [40, 41] and record["p4_sampled"].tolist() == [41, 42]
    assert record["p4_ema"][:, 0, 0].tolist() == [0.5, 1.5] and record["p4_age"][:, 0, 0].tolist() == [0, 1]
    assert record["p4_age"].dtype == torch.int16
    # the next request starts from zero rows
    assert rec.p4["consumed"].abs().sum().item() == 0 and rec.p4["direct"][:2].sum().item() == 0


def test_shared_mode_refuses_bad_hidden_and_age_range(tmp_path):
    cache = make_cache()
    with pytest.raises(ValueError, match="int16"):
        RoutingRecorder(FakeShared(cache, far=40000), tmp_path, top_k=K, capacity=4).bind(None, cache)
    rec = RoutingRecorder(FakeShared(cache), tmp_path, top_k=K, capacity=4)
    rec.bind(None, cache)
    with pytest.raises(RuntimeError, match="hidden"):
        rec.observe(1, torch.zeros(2, D, dtype=torch.bfloat16), None, ids(1, 2))
    rec.observe(1, torch.zeros(1, D, dtype=torch.bfloat16), None, ids(1, 2))
    with pytest.raises(RuntimeError, match="dtype"):
        rec.observe(2, torch.zeros(1, D, dtype=torch.float16), None, ids(1, 2))


def test_shared_mode_refuses_a_producer_dtype_that_fp32_does_not_hold(tmp_path):
    cache = make_cache()
    rec = RoutingRecorder(FakeShared(cache), tmp_path, top_k=K, capacity=4)
    rec.bind(None, cache)
    with pytest.raises(RuntimeError, match="not exact in FP32"):
        rec.observe(1, torch.zeros(1, D, dtype=torch.float64), None, ids(1, 2))
    assert rec.p4_producer_dtype is None  # the refused call names no source dtype
    rec.observe(1, torch.zeros(1, D, dtype=torch.float32), None, ids(1, 2))
    assert rec.p4_producer_dtype == "torch.float32"


def test_shared_mode_counts_ages_outside_the_index_range_before_the_int16_cast(tmp_path):
    cache = make_cache()
    inner = FakeShared(cache, far=7)
    rec = RoutingRecorder(inner, tmp_path, top_k=K, capacity=4)
    rec.bind(None, cache)
    rec.begin_request(1)
    shared_step(rec, inner, 0, 40)  # ages 0: inside 0..far
    rec.before_decode(1)
    inner.shared.age.fill_(7)  # far itself is inside the range
    inner.shared.age[0, 0] = -1
    inner.shared.age[1, 3] = 65536  # wraps to 0 in int16: a plausible age without the count
    inner.shared.age[2, 4] = 8
    for layer in range(L):
        rec.observe(layer, torch.zeros(1, D, dtype=torch.bfloat16), None, ids(layer, layer))
    rec.after_decode(1, torch.tensor([41]), torch.tensor([42]))
    rec.begin_request(2)
    record = torch.load(files(tmp_path)[0])
    assert record["p4_age_out_of_range"].tolist() == [0, 3]
    assert record["p4_age_out_of_range"].dtype == torch.int32
    assert record["p4_age"][1, 1, 3].item() == 0 and record["p4_age"][1, 0, 0].item() == -1


def test_shared_mode_lifecycle_over_capacity_and_two_requests(tmp_path):
    cache = make_cache()
    inner = FakeShared(cache)
    rec = RoutingRecorder(inner, tmp_path, top_k=K, capacity=2)
    rec.bind(None, cache)
    rec.observe(1, torch.full((1, D), 9.0, dtype=torch.bfloat16), None, ids(9, 9))  # warmup: scratch row
    rec.begin_request(5)
    for step in range(3):  # step 2 overflows the capacity of 2 rows
        shared_step(rec, inner, step, 40 + step)
    rec.observe(2, torch.full((1, D), 8.0, dtype=torch.bfloat16), None, ids(8, 8))  # stray: scratch row
    rec.begin_request(6)
    for step in range(2):
        shared_step(rec, inner, step, 60 + step)
    rec.begin_request(7)
    first, second = (torch.load(p) for p in files(tmp_path))
    assert first["overflow"] is True and first["steps"] == 3 and second["overflow"] is False
    for record, base in ((first, 0), (second, 0)):
        n = record["ids"].shape[0]
        assert n == 2 and all(v.shape[0] == n for k, v in record.items() if k.startswith("p4_")
                              and isinstance(v, torch.Tensor))
        # rows and ids stay aligned: ids at layer 0 route (0, 1) every step; producer per step
        assert record["ids"][:, 0].tolist() == [[0, 1], [0, 1]]
        want = torch.tensor([[1.3 + s, 2.3 + s] for s in range(2)], dtype=torch.bfloat16).float()
        assert torch.equal(record["p4_producer"][:, :, 0], want)
    # no warmup, overflow or stray value reaches an exported row
    for record in (first, second):
        assert not (record["p4_producer"] == 9.0).any() and not (record["p4_producer"] == 8.0).any()
        # step 2's layer-1 value (step 1's layer-2 value is the same number, so test layer 1 only)
        assert not (record["p4_producer"][:, 0] == torch.tensor(1.3 + 2, dtype=torch.bfloat16).float()).any()
    assert first["p4_consumed"].tolist() == [40, 41] and second["p4_consumed"].tolist() == [60, 61]
    # the second request's step 0 read reset frozen inputs, not the first request's last values
    assert second["p4_h"][0].abs().sum().item() == 0 and second["p4_routes"][0].abs().sum().item() == 0
    assert second["p4_tid"][0].item() == 0 and second["p4_pos"][0].item() == 0
    assert torch.equal(second["p4_h"][1], second["p4_producer"][0].half())
    assert second["p4_tid"][1].item() == 60


def test_rows_mode_export_names_its_rows_and_has_no_p4(tmp_path):
    cache = make_cache()
    inner = FakeInner(with_probs=False)
    inner.learned = SimpleNamespace(keep=torch.zeros(2, E), valid=torch.zeros(1, dtype=torch.int32),
                                    start_layer=1, counters=torch.zeros(3, dtype=torch.int64))
    inner.evict = SimpleNamespace(layers=[0, 2], source_layers=[0, 1])
    rec = RoutingRecorder(inner, tmp_path, top_k=K, capacity=4)
    rec.bind(None, cache)
    rec.begin_request(1)
    decode_step(rec, 0, [(1, 1), (1, 1), (1, 1)])
    rec.begin_request(2)
    record = torch.load(files(tmp_path)[0])
    assert rec.p4 is None and record["evict_mode"] == "rows"
    assert record["evict_layers"] == [0, 2] and record["evict_source_layers"] == [0, 1]
    assert not [k for k in record if k.startswith("p4_") or k == "shared_hidden_layers"]
