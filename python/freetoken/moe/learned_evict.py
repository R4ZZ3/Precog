"""Learned victim choice for the decode slot cache (EXP526).

The default decode path evicts the least-recently-used slots (flashlib ``lru_ensure``). With a
``LearnedEvict`` attached to the cache, ``ensure_experts`` runs ``learned_ensure`` instead. It is
the victim rule of the EXP522-EXP525 replay (exp522_replay/native_multi.cpp), which produced
every learned-eviction number in the log:

1. Stamp every hit of this layer event with the new clock. A slot touched in this event is
   never evicted.
2. Pair the missing ids, sorted ascending, with the oldest evictable slots, ordered by
   (usage, slot). This global LRU batch is fixed before any target decision.
3. For each target layer m that owns q > 0 slots of the batch, and only when its keep scores are
   published: choose q slots among all evictable slots of layer m, lowest keep score first, then
   (usage, slot). They take the batch positions of m's slots in the order of those positions.
   Empty slots are LRU candidates, never target candidates.
4. Clock swap: a slot of m that LRU would evict but the rule keeps ("rescued") takes the usage of
   a slot that the rule evicts but LRU would keep ("donor"); both lists are sorted by pre-swap
   (usage, slot) and paired in order. The surviving slots of m then hold the same usage values as
   under LRU, so the rule does not make layer m look younger to later LRU batches.
5. Install the incoming ids into the batch slots, as lru_ensure does.

Keep scores are P(next use of the expert within 8 tokens) for the next token. The hook writes them
once per token and sets ``valid``; ``gate`` is 0 for the layers that run before the scores are
guaranteed to exist, and those events use plain LRU (the replay's missing-score fallback).
With no target owning a batch slot, or with ``valid == 0``, the result equals ``lru_ensure`` slot
for slot (tested).

Contracts the caller keeps (the kernel does not check them on the device):
- ``keep`` is finite whenever ``valid == 1``. A NaN never compares equal to the row minimum, so
  the selection would read the sentinel key and return a slot outside the cache. The decode
  hooks publish ``valid = 0`` for a token whose scores are not all finite.
- The usage clock stays below 2**(63 - SLOT_BITS): (usage, slot) is packed into one int64 key
  (2**51 at 4096 lanes, about 10**13 decode tokens at 40 layer events per token).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

# Columns of LearnedEvict.counters (cumulative; zeroed by the hook per request).
COUNTERS = ('events_scored', 'target_batches_reranked', 'slots_rescued')


@dataclass
class LearnedEvict:
    """Device state of the learned victim rule, owned by the decode hooks.

    keep: (M, E) float32, keep score of every expert of target m for the next token.
    valid: (1,) int32, 1 when ``keep`` holds this token's published scores, else 0.
    target_of_layer: (L,) int32, the row m of ``keep`` for a target layer, -1 otherwise.
    start_layer: events of layers below it use LRU (the scores may not exist yet there).
    counters: (len(COUNTERS),) int64.
    """

    keep: torch.Tensor
    valid: torch.Tensor
    target_of_layer: torch.Tensor
    start_layer: int
    counters: torch.Tensor

    def __post_init__(self) -> None:
        M, E = self.keep.shape
        problems = []
        if self.keep.dtype != torch.float32 or not self.keep.is_contiguous():
            problems.append(f'keep must be contiguous float32, got {self.keep.dtype}')
        if self.valid.dtype != torch.int32 or self.valid.numel() != 1:
            problems.append('valid must be one int32')
        if self.target_of_layer.dtype != torch.int32:
            problems.append('target_of_layer must be int32')
        rows = sorted(r for r in self.target_of_layer.tolist() if r >= 0)
        if rows != list(range(M)):
            problems.append(f'target_of_layer must map each keep row 0..{M - 1} exactly once, got {rows}')
        if self.start_layer < 0:
            problems.append(f'start_layer {self.start_layer} < 0')
        if self.counters.dtype != torch.int64 or self.counters.numel() != len(COUNTERS):
            problems.append(f'counters must be {len(COUNTERS)} int64')
        if problems:
            raise ValueError('learned eviction: ' + '; '.join(problems))


@triton.jit
def _gather(vec, idx, lanes):
    """vec[idx] for a small register vector: one lane matches, the others add 0."""
    return tl.sum(tl.where(lanes == idx, vec, 0))


@triton.jit(do_not_specialize=['K', 'num_cached', 'id_base', 'gate', 'E', 'M'])
def _learned_ensure_kernel(
    query_ptr, slot_of_id_ptr, id_of_slot_ptr, usage_ptr, step_ptr,
    out_ptr, src_ptr, dst_ptr, num_copy_ptr, stats_ptr,
    keep_ptr, valid_ptr, target_of_layer_ptr, counters_ptr,
    K, num_cached, id_base, gate, E, M,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr, SLOT_BITS: tl.constexpr,
    USAGE_MAX: tl.constexpr, COLLECT_STATS: tl.constexpr,
):
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)

    # Phase 1, as flashlib lru_ensure._phase1: dedup the query, stamp hits, rank the misses by id.
    k = tl.arange(0, BLOCK_K)
    kmask = k < K
    q = tl.load(query_ptr + k, mask=kmask, other=-1) + id_base
    s = tl.load(slot_of_id_ptr + q, mask=kmask, other=-1)
    hit = kmask & (s >= 0)
    miss = kmask & (s == -1)
    same = (q[:, None] == q[None, :]) & (k[:, None] > k[None, :]) & kmask[:, None] & kmask[None, :]
    first = kmask & (tl.sum(same.to(tl.int32), axis=1) == 0)
    first_miss = miss & first
    rank = tl.sum(((q[None, :] < q[:, None]) & first_miss[None, :]).to(tl.int32), axis=1)
    num_missing = tl.sum(first_miss.to(tl.int32))
    tl.store(num_copy_ptr, num_missing.to(tl.int64))
    tl.store(usage_ptr + s, step, mask=hit)
    out = tl.where(hit, s, -1)

    if num_missing > 0:
        # The hit stamp above is a scatter store; the bulk load below must see it (see the same
        # fence in flashlib _lru_ensure_kernel: without it a just-hit slot can be evicted).
        tl.debug_barrier()
        c = tl.arange(0, BLOCK_C)
        cmask = c < num_cached
        u0 = tl.load(usage_ptr + c, mask=cmask, other=USAGE_MAX)
        evictable = cmask & (u0 != step)
        umax = tl.full([BLOCK_C], USAGE_MAX, u0.dtype)
        u = tl.where(evictable, u0, umax)

        # Step 2: the global LRU batch. tl.argmin returns the lowest index among equal minima,
        # so the order is (usage, slot), as in lru_ensure and the replay.
        batch = tl.full([BLOCK_K], -1, tl.int32)
        for i in tl.range(num_missing):
            v = tl.argmin(u, axis=0).to(tl.int32)
            batch = tl.where(k == i, v, batch)
            u = tl.where(c == v, umax, u)

        use_scores = gate != 0
        if use_scores:
            use_scores = tl.load(valid_ptr) != 0
        if use_scores:
            in_batch = k < num_missing
            owner = tl.load(id_of_slot_ptr + c, mask=cmask, other=-1)
            # Integer division truncates toward zero, so an empty slot (-1) needs its own branch.
            owner_layer = tl.where(owner >= 0, owner // E, -1)
            owner_expert = tl.where(owner >= 0, owner % E, 0)
            b_owner = tl.load(id_of_slot_ptr + batch, mask=in_batch, other=-1)
            b_layer = tl.where(b_owner >= 0, b_owner // E, -1)
            b_target = tl.load(target_of_layer_ptr + b_layer, mask=in_batch & (b_layer >= 0), other=-1)
            packed = (u0.to(tl.int64) << SLOT_BITS) | c.to(tl.int64)
            KEY_MAX: tl.constexpr = 0x7FFFFFFFFFFFFFFF
            # Loop-carried values must be tensors in Triton, not Python ints.
            reranked = tl.full((), 0, tl.int32)
            rescued_total = tl.full((), 0, tl.int32)
            new_batch = batch
            for m in tl.range(M):
                is_t = in_batch & (b_target == m)
                quota = tl.sum(is_t.to(tl.int32))
                if quota > 0:
                    t_layer = tl.max(tl.where(is_t, b_layer, -1))
                    cand = evictable & (owner_layer == t_layer)
                    score = tl.load(keep_ptr + m * E + owner_expert, mask=cand, other=0.0)
                    # Step 3: quota slots by (keep score, usage, slot).
                    sel = tl.full([BLOCK_K], -1, tl.int32)
                    for j in tl.range(quota):
                        sc = tl.where(cand, score, float('inf'))
                        low = tl.min(sc)
                        best = tl.min(tl.where(cand & (sc == low), packed, KEY_MAX))
                        v = (best & ((1 << SLOT_BITS) - 1)).to(tl.int32)
                        sel = tl.where(k == j, v, sel)
                        cand = cand & (c != v)
                    # The j-th batch position owned by m (in position order) takes sel[j].
                    before = tl.sum((is_t[None, :] & (k[None, :] < k[:, None])).to(tl.int32), axis=1)
                    taken = tl.sum(tl.where(before[:, None] == k[None, :], sel[None, :], 0), axis=1)
                    new_batch = tl.where(is_t, taken, new_batch)
                    # Step 4: clock swap between rescued and donor slots.
                    sel_ok = k < quota
                    ord_slots = tl.where(is_t, batch, -1)
                    ord_in_sel = tl.sum(((ord_slots[:, None] == sel[None, :]) & sel_ok[None, :]).to(tl.int32), 1) > 0
                    sel_in_ord = tl.sum(((sel[:, None] == ord_slots[None, :]) & is_t[None, :]).to(tl.int32), 1) > 0
                    rescued = is_t & ~ord_in_sel
                    donor = sel_ok & ~sel_in_ord
                    r_slot = tl.where(rescued, batch, 0)
                    d_slot = tl.where(donor, sel, 0)
                    r_use = tl.load(usage_ptr + r_slot, mask=rescued, other=0)
                    d_use = tl.load(usage_ptr + d_slot, mask=donor, other=0)
                    r_key = tl.where(rescued, (r_use.to(tl.int64) << SLOT_BITS) | r_slot.to(tl.int64), KEY_MAX)
                    d_key = tl.where(donor, (d_use.to(tl.int64) << SLOT_BITS) | d_slot.to(tl.int64), KEY_MAX)
                    r_rank = tl.sum((r_key[None, :] < r_key[:, None]).to(tl.int32), axis=1)
                    d_rank = tl.sum((d_key[None, :] < d_key[:, None]).to(tl.int32), axis=1)
                    # Rescued i takes the usage of the donor with the same rank.
                    swap_use = tl.sum(tl.where((d_rank[None, :] == r_rank[:, None]) & donor[None, :],
                                               d_use[None, :], 0), axis=1)
                    # Every donor is read above before any rescued slot is written (disjoint slots).
                    tl.debug_barrier()
                    tl.store(usage_ptr + r_slot, swap_use.to(u0.dtype), mask=rescued)
                    reranked += 1
                    rescued_total += tl.sum(rescued.to(tl.int32))
            batch = new_batch
            ci = tl.arange(0, 4)
            cv = tl.where(ci == 0, 1, tl.where(ci == 1, reranked, rescued_total))
            tl.atomic_add(counters_ptr + ci, cv.to(tl.int64), mask=ci < 3)
            tl.debug_barrier()

        # Step 5: install, exactly as the lru_ensure sequential loop does.
        for i in tl.range(num_missing):
            victim = _gather(batch, i, k)
            old = tl.load(id_of_slot_ptr + victim)
            if old >= 0:
                tl.store(slot_of_id_ptr + old, -1)
            e = tl.sum(tl.where((rank == i) & first_miss, q, 0))
            tl.store(id_of_slot_ptr + victim, e)
            tl.store(slot_of_id_ptr + e, victim)
            tl.store(usage_ptr + victim, step)
            tl.store(dst_ptr + i, victim)
            tl.store(src_ptr + i, e - id_base)
            out = tl.where((rank == i) & miss, victim, out)

    tl.store(out_ptr + k, out, mask=kmask)
    if COLLECT_STATS:
        si = tl.arange(0, 4)
        sv = tl.where(si == 0, tl.sum(first.to(tl.int32)), tl.where(si == 1, num_missing, 1))
        tl.atomic_add(stats_ptr + si, sv.to(tl.int64), mask=si < 3)


def learned_ensure(cache, layer_id: int, expert_ids: torch.Tensor, rule: LearnedEvict) -> None:
    """Drop-in for ``lru_ensure`` in ``ensure_experts``: same maps, plan and in-place rewrite."""
    query = expert_ids.view(-1)
    k = query.numel()
    num_cached = cache.id_of_slot.numel()
    assert query.dtype == torch.int32 and query.is_contiguous()
    assert cache.usage.dtype == cache.step.dtype == torch.int64
    block_c = max(2, triton.next_power_of_2(num_cached))
    slot_bits = block_c.bit_length() - 1
    stats = cache.lru_stats[layer_id] if cache.collect_stats else None
    _learned_ensure_kernel[(1,)](
        query, cache.slot_for_id.view(-1), cache.id_of_slot, cache.usage, cache.step,
        query, cache.src_indices, cache.evict_slots, cache.num_indices,
        stats if stats is not None else cache.num_indices,
        rule.keep, rule.valid, rule.target_of_layer, rule.counters,
        k, num_cached, layer_id * cache.num_experts, int(layer_id >= rule.start_layer),
        cache.num_experts, rule.keep.shape[0],
        BLOCK_K=max(2, triton.next_power_of_2(k)), BLOCK_C=block_c, SLOT_BITS=slot_bits,
        USAGE_MAX=torch.iinfo(torch.int64).max, COLLECT_STATS=stats is not None,
        num_warps=8 if block_c >= 2048 else 4,
    )
