"""Striped demand copy for decode misses (--moe-demand-copy-stripes N).

FreeToken's fused copy moves a layer's missing experts with one CTA group per bank. This
copy splits every expert row over N CTAs per bank (grid = banks x N), each copying a
disjoint stripe of BLOCK-sized steps, and runs on its own high-priority stream that the
main stream joins before the expert GEMM. On an RTX 4080 over PCIe 4.0 x16, 16 stripes moved
2-4 misses at 23.1-23.3 GB/s against 21.1-21.6 GB/s for 8.

A prefetcher can hand the copy a private stage: a staged expert whose ``READY`` generation
equals the current ``GEN`` is copied from the stage (device memory) instead of the host bank.
All CTAs of one launch agree on one source per expert through ``CHOICES``, also when a stage
publishes mid-launch. Without a prefetcher every ``IDS`` entry is -1 and every miss reads the host.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.moe.decode_hooks import DecodeHooks


@triton.jit
def demand_copy(DSTS, HOSTS, STAGES, SIZES, DI, SI, N, IDS, READY, GEN, CHOICES, COUNTERS,
                LAYER: tl.constexpr, NUM_EXPERTS: tl.constexpr, STAGE_SLOTS: tl.constexpr,
                STAGE_POW2: tl.constexpr, BLOCK: tl.constexpr = 256, STRIPES: tl.constexpr = 8):
    bank = tl.program_id(0)
    stripe = tl.program_id(1)
    n = tl.load(N)
    generation = tl.load(GEN)
    size = tl.load(SIZES + bank) // 8
    dest = tl.load(DSTS + bank).to(tl.pointer_type(tl.uint64))
    host = tl.load(HOSTS + bank).to(tl.pointer_type(tl.uint64))
    stage = tl.load(STAGES + bank).to(tl.pointer_type(tl.uint64))
    positions = tl.arange(0, STAGE_POW2)
    ids = tl.load(IDS + positions, positions < STAGE_SLOTS, other=-1)
    for row in range(n):
        expert = tl.load(SI + row)
        gid = LAYER * NUM_EXPERTS + expert
        dst = tl.load(DI + row)
        slot = tl.min(tl.where((positions < STAGE_SLOTS) & (ids == gid), positions, STAGE_POW2), 0)
        matched = slot < STAGE_SLOTS
        ready_gen = tl.atomic_add(READY + tl.minimum(slot, STAGE_SLOTS - 1), 0, sem='acquire')
        ready = matched & (ready_gen == generation) & (generation > 0)
        # CHOICES = generation << 2 | source: 1 staged, 2 host without a selection, 3 host while selected but pending
        proposed = (generation << 2) | tl.where(ready, 1, tl.where(matched, 3, 2)).to(tl.int64)
        previous = tl.atomic_add(CHOICES + gid, 0, sem='acquire')
        choice = previous
        if (previous >> 2) != generation:
            winner = tl.atomic_cas(CHOICES + gid, previous, proposed, sem='acq_rel')
            choice = tl.where(winner == previous, proposed, winner)
        staged = (choice & 3) == 1
        source = tl.where(staged, stage + slot * size, host + expert * size)
        if (bank == 0) & (stripe == 0):
            tl.atomic_add(COUNTERS, staged.to(tl.int64), sem='relaxed')
            tl.atomic_add(COUNTERS + 1, (~staged).to(tl.int64), sem='relaxed')
            tl.atomic_add(COUNTERS + 2, ((choice & 3) == 3).to(tl.int64), sem='relaxed')
        for start in range(stripe * BLOCK, size, STRIPES * BLOCK):
            ix = start + tl.arange(0, BLOCK)
            value = tl.load(source + ix, ix < size, other=0)
            tl.store(dest + dst * size + ix, value, ix < size)


class StripedDemand:
    """State of the striped demand copy: source choices, per-layer counters and the copy stream.

    ``COUNTERS[layer]`` accumulates (staged, host, host while pending) miss counts per request.
    """

    def __init__(self, num_layers: int, num_experts: int, stripes: int, stage_slots: int,
                 device: torch.device):
        if stripes < 1 or stage_slots < 1:
            raise ValueError(f"stripes {stripes} and stage_slots {stage_slots} must be positive")
        self.num_experts, self.stripes, self.stage_slots = num_experts, stripes, stage_slots
        self.ids = torch.full((stage_slots,), -1, dtype=torch.int32, device=device)
        self.ready = torch.zeros(stage_slots, dtype=torch.int64, device=device)
        self.gen = torch.ones(1, dtype=torch.int64, device=device)
        self.choices = torch.zeros(num_layers * num_experts, dtype=torch.int64, device=device)
        self.counters = torch.zeros((num_layers, 3), dtype=torch.int64, device=device)
        self.stream = torch.cuda.Stream(device=device, priority=-1)

    def check_cache(self, cache) -> None:
        if not cache._copy_fused_ok:
            raise RuntimeError("--moe-demand-copy-stripes needs the fused copy path (every layer pinned on the host)")
        if bool((cache._copy_feat_bytes % 8 != 0).any()):
            raise RuntimeError("--moe-demand-copy-stripes copies 8-byte words; an expert row is not a multiple of 8 bytes")

    def reset(self) -> None:
        self.choices.zero_()
        self.counters.zero_()

    def launch(self, cache, layer: int, stages: torch.Tensor | None = None, ids: torch.Tensor | None = None,
               ready: torch.Tensor | None = None, gen: torch.Tensor | None = None) -> None:
        """Enqueue the copy of ``layer``'s misses on the current stream.

        Without a stage the host pointers stand in for the stage pointers; they are never
        read, because every ``ids`` entry is -1.
        """
        hosts = cache._copy_src_ptrs[layer]
        demand_copy[(len(cache._copy_dst_ptrs), self.stripes)](
            cache._copy_dst_ptrs, hosts, hosts if stages is None else stages, cache._copy_feat_bytes,
            cache.evict_slots, cache.src_indices, cache.num_indices,
            self.ids if ids is None else ids, self.ready if ready is None else ready,
            self.gen if gen is None else gen, self.choices, self.counters[layer],
            LAYER=layer, NUM_EXPERTS=self.num_experts, STAGE_SLOTS=self.stage_slots,
            STAGE_POW2=triton.next_power_of_2(self.stage_slots), STRIPES=self.stripes, num_warps=8,
        )

    def copy(self, cache, layer: int) -> None:
        """Host-only demand copy of one layer on the copy stream, joined back into the main stream."""
        main = torch.cuda.current_stream()
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            self.launch(cache, layer)
        main.wait_stream(self.stream)


class StripedDemandHooks(DecodeHooks):
    """The striped demand copy without a predictor."""

    single_request = False  # no per-request state: every miss reads the host

    def __init__(self, demand: StripedDemand):
        self.demand = demand

    def bind(self, engine, cache) -> None:
        self.demand.check_cache(cache)

    def begin_request(self, uid: int) -> None:
        self.demand.reset()

    def copy_missing(self, cache, layer_id: int) -> bool:
        self.demand.copy(cache, layer_id)
        return True
