"""The live learned victim rule (learned_evict.learned_ensure) against the replay rule.

Every learned-eviction number of EXP522-EXP525 comes from the replay in
experiments/exp522_replay/native_multi.cpp. The live kernel must make the same decision on the
same state, slot for slot, and with ``valid == 0`` (or below ``start_layer``) it must equal
flashlib's ``lru_ensure``. Three references:

* ``ReplayRef``: a line-by-line NumPy port of native_multi.cpp's event loop, plus the live
  ``start_layer`` gate (the replay's ``score_start_layer``; layers below it use LRU). It is
  compared event by event: maps, usage, clock, plan and the rescued-slot count.
* the compiled native_multi.cpp itself (when its source is present): final owners, ages,
  clock, per-event hits and miss counts over a whole sequence.
* flashlib ``lru_ensure`` for the identity cases.

Fixtures are small (tens of slots) so that every branch is hit often: full and partial target
quotas, empty slots, keep-score ties, hits inside a batch, and slots that the swap rescues.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from freetoken.moe.learned_evict import COUNTERS, LearnedEvict, learned_ensure

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs a GPU")

REPLAY_SRC = Path(os.environ.get(
    "EXP522_REPLAY_DIR",
    "/mnt/e/CommoditizeTeraModel/commitmoe_modeling_bundle/commitmoe_export/.worktrees/codex-exp520/"
    "commitmoe_modeling_bundle/commitmoe_export/experiments/exp522_replay",
)) / "native_multi.cpp"


class ReplayRef:
    """native_multi.cpp, one layer event at a time (clock_swap on, as in every EXP523+ replay)."""

    def __init__(self, S: int, L: int, E: int, targets: list[int]):
        self.S, self.L, self.E, self.targets = S, L, E, targets
        self.owners = np.full(S, -1, np.int64)
        self.ages = np.zeros(S, np.int64)
        self.slot_of = np.full(L * E, -1, np.int64)
        self.clock = 0

    def event(self, layer: int, experts: list[int], keep: np.ndarray | None, start_layer: int):
        """Returns (victims in install order, incoming gids, rescued count)."""
        owners, ages, E = self.owners, self.ages, self.E
        self.clock += 1
        clock = self.clock
        missing = []
        for e in experts:
            gid = layer * E + e
            s = self.slot_of[gid]
            if s >= 0:
                ages[s] = clock
            else:
                missing.append(gid)
        if not missing:
            return [], [], 0
        missing.sort()
        older = lambda s: (ages[s], s)  # noqa: E731
        cands = sorted((s for s in range(self.S) if ages[s] != clock), key=older)
        ordinary = cands[:len(missing)]
        victims = list(ordinary)
        rescued_total = 0
        scored = keep is not None and layer >= start_layer
        for m, target in enumerate(self.targets):
            ord_t = [s for s in ordinary if owners[s] >= 0 and owners[s] // E == target]
            if not scored or not ord_t:
                continue
            tc = [s for s in range(self.S) if owners[s] >= 0 and owners[s] // E == target and ages[s] != clock]
            tc.sort(key=lambda s: (keep[m][owners[s] % E], ages[s], s))
            sel = tc[:len(ord_t)]
            rescued = sorted((s for s in ord_t if s not in sel), key=older)
            donors = sorted((s for s in sel if s not in ord_t), key=older)
            assert len(rescued) == len(donors)
            for r, d in zip(rescued, donors):
                ages[r] = ages[d]
            rescued_total += len(rescued)
            nxt = 0
            for i, s in enumerate(ordinary):
                if owners[s] >= 0 and owners[s] // E == target:
                    victims[i] = tc[nxt]
                    nxt += 1
        for victim, incoming in zip(victims, missing):
            old = owners[victim]
            if old >= 0:
                self.slot_of[old] = -1
            owners[victim] = incoming
            self.slot_of[incoming] = victim
            ages[victim] = clock
        return victims, missing, rescued_total


def make_cache(S: int, L: int, E: int, dev) -> SimpleNamespace:
    """The fields of OffloadMoeCache that ensure_experts reads, with the same dtypes and shapes."""
    plan = max(E, S)
    return SimpleNamespace(
        slot_for_id=torch.full((L, E), -1, dtype=torch.int32, device=dev),
        id_of_slot=torch.full((S,), -1, dtype=torch.int32, device=dev),
        usage=torch.zeros((S,), dtype=torch.int64, device=dev),
        step=torch.zeros((), dtype=torch.int64, device=dev),
        src_indices=torch.empty((plan,), dtype=torch.int32, device=dev),
        evict_slots=torch.empty((plan,), dtype=torch.int32, device=dev),
        num_indices=torch.zeros((1,), dtype=torch.int64, device=dev),
        collect_stats=False, lru_stats=None, num_experts=E,
    )


def make_rule(targets: list[int], L: int, E: int, start_layer: int, dev) -> LearnedEvict:
    tol = torch.full((L,), -1, dtype=torch.int32)
    tol[targets] = torch.arange(len(targets), dtype=torch.int32)
    return LearnedEvict(keep=torch.zeros(len(targets), E, dtype=torch.float32, device=dev),
                        valid=torch.zeros(1, dtype=torch.int32, device=dev),
                        target_of_layer=tol.to(dev), start_layer=start_layer,
                        counters=torch.zeros(len(COUNTERS), dtype=torch.int64, device=dev))


def make_routes(rng, T: int, L: int, E: int, K: int, hot: int) -> np.ndarray:
    """(T, L, K) distinct experts per event; a hot subset per layer makes hits and reuse common."""
    routes = np.empty((T, L, K), np.int32)
    for t in range(T):
        for layer in range(L):
            pool = E if rng.random() < 0.3 else hot
            routes[t, layer] = rng.choice(pool, K, replace=False)
    return routes


def make_keep(rng, T: int, M: int, E: int, ties: bool) -> np.ndarray:
    """(T, M, E) float32 keep scores; with ties, values come from 4 levels so (usage, slot) decides."""
    if ties:
        return rng.integers(0, 4, (T, M, E)).astype(np.float32) / 4
    return rng.random((T, M, E), dtype=np.float32)


CASES = [
    # S, L, E, K, targets, start_layer, hot, ties, available fraction
    (24, 4, 16, 4, [1, 3], 0, 6, False, 1.0),
    (24, 4, 16, 4, [1, 3], 0, 6, True, 1.0),
    (40, 6, 16, 4, [0, 2, 5], 3, 8, False, 0.7),
    (64, 6, 32, 8, [2, 3], 0, 12, True, 0.5),
    (17, 5, 8, 3, [4], 2, 4, False, 1.0),  # S not a power of two, one target
]


def run_live(case, seed: int, T: int = 40):
    S, L, E, K, targets, start, hot, ties, avail = case
    rng = np.random.default_rng(seed)
    routes = make_routes(rng, T, L, E, K, hot)
    keep = make_keep(rng, T, len(targets), E, ties)
    available = rng.random(T) < avail
    dev = torch.device("cuda")
    cache, rule = make_cache(S, L, E, dev), make_rule(targets, L, E, start, dev)
    ref = ReplayRef(S, L, E, targets)
    rescued_total = 0
    for t in range(T):
        # Scores of row t-1 serve token t (the replay's convention); token 0 has none.
        scored = t > 0 and bool(available[t - 1])
        rule.keep.copy_(torch.from_numpy(keep[t - 1] if t > 0 else keep[0]))
        rule.valid.fill_(int(scored))
        for layer in range(L):
            ids = torch.from_numpy(routes[t, layer].copy()).to(dev)
            learned_ensure(cache, layer, ids, rule)
            victims, incoming, rescued = ref.event(layer, list(routes[t, layer]), keep[t - 1] if scored else None,
                                                   start)
            rescued_total += rescued
            n = int(cache.num_indices.item())
            assert n == len(incoming), (t, layer)
            assert cache.evict_slots[:n].tolist() == victims, (t, layer)
            assert (cache.src_indices[:n] + layer * E).tolist() == incoming, (t, layer)
            assert cache.id_of_slot.tolist() == ref.owners.tolist(), (t, layer)
            assert cache.usage.tolist() == ref.ages.tolist(), (t, layer)
            assert int(cache.step) == ref.clock
            # ids were rewritten in place to the slot of each routed expert
            assert cache.id_of_slot[ids.long()].tolist() == [layer * E + e for e in routes[t, layer]]
    assert int(rule.counters[2]) == rescued_total
    return routes, keep, available, cache, rule, rescued_total


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_replay_port(case, seed):
    _, _, _, _, rule, rescued = run_live(case, seed)
    if case[7] is False and case[8] == 1.0:
        # These fixtures are meant to exercise the swap; a zero would mean the test is vacuous.
        assert rescued > 0 and int(rule.counters[1]) > 0


def test_no_scores_equals_lru():
    """valid == 0 for every token: the learned kernel must equal flashlib lru_ensure exactly."""
    from flashlib.kernels.slot_cache.triton.lru_ensure import lru_ensure

    S, L, E, K, T = 40, 6, 16, 4, 40
    rng = np.random.default_rng(7)
    routes = make_routes(rng, T, L, E, K, 8)
    dev = torch.device("cuda")
    a, b = make_cache(S, L, E, dev), make_cache(S, L, E, dev)
    rule = make_rule([1, 4], L, E, 0, dev)
    rule.keep.copy_(torch.rand(2, E))
    for t in range(T):
        for layer in range(L):
            ida = torch.from_numpy(routes[t, layer].copy()).to(dev)
            idb = ida.clone()
            learned_ensure(a, layer, ida, rule)
            lru_ensure(idb, b.slot_for_id.view(-1), b.id_of_slot, b.usage, b.step, idb, b.src_indices,
                       b.evict_slots, b.num_indices, stats=None, id_base=layer * E)
            n = int(b.num_indices.item())
            assert int(a.num_indices.item()) == n
            assert torch.equal(ida, idb)
            assert torch.equal(a.evict_slots[:n], b.evict_slots[:n])
            assert torch.equal(a.src_indices[:n], b.src_indices[:n])
            for name in ("slot_for_id", "id_of_slot", "usage", "step"):
                assert torch.equal(getattr(a, name), getattr(b, name)), (t, layer, name)
    assert rule.counters.tolist() == [0, 0, 0]


def test_duplicate_query_ids_match_lru():
    """Prefill can route one expert twice in a batch; with no scores the result equals lru_ensure."""
    from flashlib.kernels.slot_cache.triton.lru_ensure import lru_ensure

    S, L, E = 24, 3, 16
    dev = torch.device("cuda")
    a, b = make_cache(S, L, E, dev), make_cache(S, L, E, dev)
    rule = make_rule([2], L, E, 0, dev)
    rng = np.random.default_rng(3)
    for layer in (0, 1, 2, 0, 2):
        ids = torch.from_numpy(rng.integers(0, E, 20).astype(np.int32)).to(dev)
        idb = ids.clone()
        learned_ensure(a, layer, ids, rule)
        lru_ensure(idb, b.slot_for_id.view(-1), b.id_of_slot, b.usage, b.step, idb, b.src_indices,
                   b.evict_slots, b.num_indices, stats=None, id_base=layer * E)
        assert torch.equal(ids, idb)
        for name in ("slot_for_id", "id_of_slot", "usage", "step"):
            assert torch.equal(getattr(a, name), getattr(b, name))


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    if not REPLAY_SRC.is_file():
        pytest.skip(f"replay source not found: {REPLAY_SRC}")
    lib = tmp_path_factory.mktemp("native") / "native_multi.so"
    subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", str(REPLAY_SRC), "-o", str(lib)], check=True)
    return ctypes.CDLL(str(lib))


@pytest.mark.parametrize("case", [c for c in CASES if c[5] == 0])  # the compiled replay has no start layer
@pytest.mark.parametrize("seed", [0, 1])
def test_matches_compiled_replay(native, case, seed):
    S, L, E, K, targets, _, _, _, _ = case
    T = 40
    routes, keep, available, cache, _, _ = run_live(case, seed, T)
    M = len(targets)
    owners = np.full(S, -1, np.int32)
    ages = np.zeros(S, np.int64)
    clock = np.zeros(1, np.int64)
    scores = np.ascontiguousarray(keep.transpose(1, 0, 2))  # (M, T, E): row t-1 serves token t
    avail = np.ascontiguousarray(np.broadcast_to(available, (M, T)).astype(np.uint8))
    tg = np.asarray(targets, np.int32)
    ev = T * L
    counts = np.zeros(ev, np.int16)
    hits = np.zeros(ev * K, np.uint8)
    tv, ob, oa = (np.zeros(M * ev, np.int16) for _ in range(3))
    p = lambda a: a.ctypes.data_as(ctypes.c_void_p)  # noqa: E731
    rc = native.replay_multi(p(np.ascontiguousarray(routes)), T, L, E, K, S, p(owners), p(ages), p(clock),
                             p(scores), p(avail), p(tg), M, 1, 1, p(counts), p(hits), p(tv), p(ob), p(oa))
    assert rc == 0
    assert cache.id_of_slot.cpu().numpy().tolist() == owners.tolist()
    assert cache.usage.cpu().numpy().tolist() == ages.tolist()
    assert int(cache.step) == int(clock[0])
