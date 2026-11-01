"""FREETOKEN_OVERLAP_RESIDENT=1 (EXP538) must hand the split FP8 kernel the slot views of the
pool that serves its layer. HEAD a2ce636 removed OffloadMoeCache.bank_views(); the per-pool
pool_bank_views(layer_id) replaces it, and its rows are indexed by the pool's local slot ids.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import freetoken.kernel.triton.fp8_blockscale_moe as fp8_moe
from freetoken.layers.moe import OffloadMoELayer


class Fp8BlockMoEMethod:  # the overlap path checks these class names, not the classes
    def __init__(self):
        self.kernel = TritonFp8BlockMoEKernel()


class TritonFp8BlockMoEKernel:
    pass


def test_overlap_reads_the_pool_views_of_its_layer(monkeypatch):
    layer_id = 2
    views = {lid: tuple(torch.full((1,), float(10 * lid + k)) for k in range(4)) for lid in range(4)}
    asked = []

    def pool_bank_views(lid):
        asked.append(lid)
        return views[lid]

    cache = SimpleNamespace(
        slot_for_id=torch.tensor([[0, -1, 1, -1]] * 4, dtype=torch.int32),
        bank_schema=["gate_up_blocks", "gate_up_scales", "down_blocks", "down_scales"],
        pool_bank_views=pool_bank_views,
        ensure_experts=lambda lid, ids: None,
    )
    hooks = SimpleNamespace(begin_copy=lambda c, lid: "finish")
    seen = {}

    def split(hidden, gate_up, gate_up_scale, down, down_scale, weights, ids, resident, join, *rest):
        seen.update(banks=(gate_up, gate_up_scale, down, down_scale), resident=resident.tolist(), join=join)
        return hidden

    monkeypatch.setattr(fp8_moe, "fused_experts_decode_fp8_blockscale_split", split)
    layer = SimpleNamespace(quant_method=Fp8BlockMoEMethod(), apply_router_weight_on_input=False,
                            layer_id=layer_id, activation="silu", alpha=1.0, limit=None)
    ids = torch.tensor([[0, 1]])
    OffloadMoELayer._decode_overlap(layer, cache, hooks, torch.zeros(1, 4), torch.ones(1, 2), ids)

    assert asked == [layer_id]
    assert all(a is b for a, b in zip(seen["banks"], views[layer_id]))
    assert seen["resident"] == [1, 0] and seen["join"] == "finish"
