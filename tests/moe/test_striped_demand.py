"""--moe-demand-copy-stripes: the striped demand copy must move the same bytes as the fused
copy, read a ready stage instead of the host, and count each miss by its source.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.striped_demand import StripedDemand, StripedDemandHooks

from .test_fused_copy import _build_cache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("stripes,stage_slots", [(0, 1), (16, 0)])
def test_refuses_non_positive_sizes(stripes, stage_slots):
    with pytest.raises(ValueError, match="must be positive"):
        StripedDemand(2, 4, stripes, stage_slots, torch.device("cpu"))


def _demand_stub():
    # check_cache and reset need no stream; skip __init__ so the test runs without CUDA
    demand = StripedDemand.__new__(StripedDemand)
    demand.choices = torch.full((8,), 5, dtype=torch.int64)
    demand.counters = torch.ones((2, 3), dtype=torch.int64)
    return demand


def test_refuses_cache_without_fused_copy():
    cache = SimpleNamespace(_copy_fused_ok=False, _copy_feat_bytes=torch.tensor([64]))
    with pytest.raises(RuntimeError, match="fused copy path"):
        _demand_stub().check_cache(cache)


def test_refuses_rows_not_a_multiple_of_8_bytes():
    cache = SimpleNamespace(_copy_fused_ok=True, _copy_feat_bytes=torch.tensor([64, 260]))
    with pytest.raises(RuntimeError, match="multiple of 8"):
        _demand_stub().check_cache(cache)


def test_hooks_reset_per_request_and_replace_the_engine_copy():
    demand = _demand_stub()
    calls = []
    demand.copy = lambda cache, layer: calls.append(layer)
    hooks = StripedDemandHooks(demand)
    hooks.begin_request(3)
    assert not demand.choices.any() and not demand.counters.any()
    assert hooks.copy_missing(object(), 7) is True and calls == [7]
    assert hooks.single_request is False


def _set_misses(cache, layer, slots, experts):
    dev = torch.device("cuda")
    cache._pending_src_layer = layer
    cache.num_indices.fill_(len(slots))
    if slots:
        cache.evict_slots[: len(slots)] = torch.tensor(slots, dtype=torch.int32, device=dev)
        cache.src_indices[: len(slots)] = torch.tensor(experts, dtype=torch.int32, device=dev)


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("stripes", [1, 16])
@pytest.mark.parametrize("misses", [0, 1, 4])
def test_host_copy_matches_fused_copy(stripes, misses):
    num_layers, num_experts, layer = 8, 8, 3
    cache = _build_cache(num_layers, num_experts, cache_size=32)
    _set_misses(cache, layer, list(range(0, 2 * misses, 2)), list(range(misses)))
    for _, bank in cache.banks:
        bank.zero_()
    cache.copy_missing()
    torch.cuda.synchronize()
    reference = [bank.clone() for _, bank in cache.banks]

    demand = StripedDemand(num_layers, num_experts, stripes, stage_slots=1, device=torch.device("cuda"))
    demand.check_cache(cache)
    for _, bank in cache.banks:
        bank.zero_()
    demand.copy(cache, layer)
    torch.cuda.synchronize()
    for index, (want, (_, got)) in enumerate(zip(reference, cache.banks)):
        assert torch.equal(want, got), f"bank {index} differs with {stripes} stripes and {misses} misses"
    assert demand.counters[layer].tolist() == [0, misses, 0]


@CUDA
@pytest.mark.slow
def test_ready_stage_is_read_and_pending_stage_is_not():
    num_layers, num_experts, layer, stage_slots = 8, 8, 3, 4
    dev = torch.device("cuda")
    cache = _build_cache(num_layers, num_experts, cache_size=32)
    demand = StripedDemand(num_layers, num_experts, stripes=16, stage_slots=stage_slots, device=dev)
    # one stage row per slot and bank, filled with values the host rows cannot match (host is 0..255)
    stages = [torch.full((stage_slots, bank.shape[1]), 0, dtype=torch.uint8, device=dev) for _, bank in cache.banks]
    for slot in range(stage_slots):
        for stage in stages:
            stage[slot].fill_(slot + 1)
    stage_ptrs = torch.tensor([stage.data_ptr() for stage in stages], dtype=torch.int64, device=dev)
    # slot 1 holds expert 5 and is ready; slot 2 holds expert 6 but is still pending; expert 2 is not staged
    ids = torch.full((stage_slots,), -1, dtype=torch.int32, device=dev)
    ids[1], ids[2] = layer * num_experts + 5, layer * num_experts + 6
    ready = torch.zeros(stage_slots, dtype=torch.int64, device=dev)
    ready[1] = 1
    _set_misses(cache, layer, [0, 1, 2], [5, 6, 2])
    cache.copy_missing()  # the host rows, as the fused copy moves them
    torch.cuda.synchronize()
    host_rows = [bank[:3].clone() for _, bank in cache.banks]
    for _, bank in cache.banks:
        bank.zero_()
    demand.launch(cache, layer, stages=stage_ptrs, ids=ids, ready=ready, gen=demand.gen)
    torch.cuda.synchronize()

    for (_, bank), stage, host in zip(cache.banks, stages, host_rows):
        assert torch.equal(bank[0], stage[1]), "a ready stage must be copied from the stage"
        assert torch.equal(bank[1:3], host[1:3]), "a pending or unstaged expert must be copied from the host"
    # (staged, host, host while pending)
    assert demand.counters[layer].tolist() == [1, 2, 1]
