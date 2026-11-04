"""Device kernels of the prefetch schedule. All of them run inside the captured decode graph.

These are the kernels of the measured arm GV1q16t (experiments exp483_kernels,
exp483b_kernels, exp476_kernels), with two kinds of change and no change to the copy
or publication logic:
- Model sizes are parameters: experts per layer, top-k, hidden size, number of banks.
- The per-step bookkeeping is one row per section, not one row per (decode step, section).
  The research kernels kept a row for each of the first 256 steps for offline analysis,
  which limited a request to 256 decode steps. All state that a section uses is local to
  one token, so `select_scored` now clears the section's row before the pump reads it.
  The timestamps and the per-stripe logs of the research kernels are left out.

Terms used below:
- gid: global expert id, layer * E + expert.
- Section s: the selection made at source layer 5s for target layers 5(s+1) .. 5(s+1)+4.
- Stage slot: one expert-sized buffer per bank in device memory. Sections alternate between
  two groups of SLOTS stage slots (group s % 2), so section s fills one group while the
  demand copy of section s-1's targets can still read the other.
"""
import triton
import triton.language as tl

SECTION_LAYERS = 5  # source layers per section; also the target span of one selection
SLOTS = 4           # stage slots per group = most experts one section can stage
GROUPS = 2
PUMP_STRIPES = 8    # CTAs per pump launch (EXP477 measured 8 fastest for the pump)
CHUNK_BYTES = 262144
STATS_FIELDS = ('copied_bytes', 'yields_to_demand', 'expired_calls', 'published')


@triton.jit
def capture_layer(H, RL, ROUTES, HIDDEN, LOGITS, IDS, LAYER: tl.constexpr, KEEP_RL: tl.constexpr,
                  D: tl.constexpr, D_POW2: tl.constexpr, E: tl.constexpr, E_POW2: tl.constexpr,
                  K: tl.constexpr, K_POW2: tl.constexpr):
    """Store token t's MoE input hidden state, router logits and routed ids of one layer.

    H and RL are fp16, as in training. The float conversion rounds to nearest-even, which is
    what torch's copy_ does for bf16 -> fp16.
    """
    i = tl.arange(0, D_POW2)
    tl.store(H + LAYER * D + i, tl.load(HIDDEN + i, i < D).to(tl.float32).to(tl.float16), i < D)
    if KEEP_RL:
        j = tl.arange(0, E_POW2)
        tl.store(RL + LAYER * E + j, tl.load(LOGITS + j, j < E).to(tl.float32).to(tl.float16), j < E)
    k = tl.arange(0, K_POW2)
    tl.store(ROUTES + LAYER * K + k, tl.load(IDS + k, k < K).to(tl.int64), k < K)


@triton.jit
def stamp(OUT, IDX: tl.constexpr):
    """OUT[IDX] = the GPU global timer (ns) when this kernel runs on its stream. Diagnostic only.

    The asm needs one input operand to give the result its shape; the zero is not read."""
    z = tl.zeros([1], tl.int32)
    t = tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;', '=l,r', [z], dtype=tl.int64, is_pure=False, pack=1)
    tl.store(OUT + IDX + tl.arange(0, 1), t)


@triton.jit
def phase(BUSY, LAYER, N, L: tl.constexpr, CLEAR: tl.constexpr = False):
    """Tell the pump which layer the demand copy serves and whether it copies now.

    The pump yields while BUSY is set, so a demand miss never shares PCIe with a
    speculative copy, and it stops a section once LAYER reaches the section's targets.
    """
    if CLEAR:
        tl.atomic_xchg(BUSY, 0, sem='release')
    else:
        tl.atomic_xchg(LAYER, L, sem='release')
        tl.atomic_xchg(BUSY, (tl.load(N) > 0).to(tl.int32), sem='release')


@triton.jit
def retire(IDS, READY, SECTION: tl.constexpr, SLOTS: tl.constexpr = 4):
    """Clear the stage group that section SECTION reuses.

    The group last served section SECTION - 2 (or section 6 or 5 of the previous token),
    whose target layers all finished their demand copy before this section's first source
    layer, where retire runs.
    """
    i = (SECTION % 2) * SLOTS + tl.arange(0, SLOTS)
    tl.atomic_xchg(READY + i, tl.zeros((SLOTS,), dtype=tl.int64), sem='relaxed')
    tl.atomic_xchg(IDS + i, tl.full((SLOTS,), -1, dtype=tl.int32), sem='release')


@triton.jit
def select_scored(MAP, PROBS, STEP, RECORDS, CURSORS, PAYLOAD, SECTION: tl.constexpr, FIRST: tl.constexpr,
                  FIRST_STEP: tl.constexpr, MIN_SCORE: tl.constexpr, E: tl.constexpr, SPAN_POW2: tl.constexpr,
                  SLOTS: tl.constexpr = 4, SECTION_LAYERS: tl.constexpr = 5, STRIPES: tl.constexpr = 8):
    """Pick the SLOTS most probable target-section experts that are not resident now.

    PROBS (heads, E) fp32 scores target layers FIRST .. FIRST + heads - 1. It was written
    from the end of token t; this runs during token t+1. MAP is the cache's gid -> slot
    map, so residency is the live cache state at selection time. Ties go to the lower gid.
    A slot with no eligible expert gets gid -1 and the pump skips it. MIN_SCORE: an expert
    is eligible only with a probability of at least this value; the slot is not refilled
    with a lower-scored expert. No selection before decode step FIRST_STEP.
    Also clears the section's pump cursors and payload counters for this token.
    """
    step = tl.load(STEP)
    i = tl.arange(0, SPAN_POW2)
    gid = (SECTION + 1) * SECTION_LAYERS * E + i
    valid = (i < SECTION_LAYERS * E) & (step >= FIRST_STEP) & (gid // E >= FIRST)
    resident = tl.load(MAP + gid, valid, other=0) >= 0
    prob = tl.load(PROBS + gid - FIRST * E, valid, other=0.0)
    score = tl.where(valid & ~resident & (prob >= MIN_SCORE), prob, -1.0)
    for slot in tl.static_range(SLOTS):
        best = tl.max(score, 0)
        chosen = tl.min(tl.where(score == best, gid, 2147483647), 0)
        chosen = tl.where(best >= 0, chosen, -1)
        tl.store(RECORDS + SECTION * SLOTS + slot, chosen.to(tl.int64))
        score = tl.where(gid == chosen, -1.0, score)
    tl.store(CURSORS + SECTION * STRIPES + tl.arange(0, STRIPES), tl.zeros((STRIPES,), dtype=tl.int32))
    tl.store(PAYLOAD + SECTION * SLOTS + tl.arange(0, SLOTS), tl.zeros((SLOTS,), dtype=tl.int32))


@triton.jit
def publishing_pump(HOSTS, STAGES, SIZES, BUSY, LAYER, STEP, RECORDS, CURSORS, PAYLOAD, STATS, IDS, READY, GEN,
                    LAYER_PUBLISHED,
                    SECTION: tl.constexpr, CAP: tl.constexpr, FIRST_STEP: tl.constexpr, NUM_BANKS: tl.constexpr,
                    E: tl.constexpr, QUOTA: tl.constexpr, SLOTS: tl.constexpr = 4,
                    SECTION_LAYERS: tl.constexpr = 5, STRIPES: tl.constexpr = 8,
                    CHUNK_BYTES: tl.constexpr = 262144, BLOCK: tl.constexpr = 1024,
                    COUNT_LAYERS: tl.constexpr = False):
    """Copy the section's first CAP selected experts from the host banks into its stage group.

    Each expert is split into CHUNK_BYTES chunks. Every CTA walks the same chunk sequence
    with its own cursor and copies a disjoint stripe of each chunk (BLOCK-sized steps
    pid, pid + STRIPES, ...). Before each chunk a CTA stops when the section expired (the
    demand copy reached the section's targets) or when a demand copy runs (BUSY). A later
    call of the same section continues from the cursor. A CTA never stops inside a stripe.
    QUOTA bounds the chunks one call may try. The CTA that lands an expert's last stripe
    publishes IDS[stage slot] = gid, then READY[stage slot] = GEN. demand_copy reads a stage
    only when both match.
    STATS: copied bytes, yields to a demand copy, expired calls, published experts.
    COUNT_LAYERS (FREETOKEN_LAYER_COUNTERS): LAYER_PUBLISHED[target layer] also counts each
    published expert. With False the kernel never reads that pointer, so any buffer may stand there.
    """
    pid = tl.program_id(0)
    step = tl.load(STEP)
    cursor_ptr = CURSORS + SECTION * STRIPES + pid
    cursor = tl.load(cursor_ptr).to(tl.int64)
    total = 0
    for bank in tl.static_range(NUM_BANKS): total += tl.load(SIZES + bank) // 8
    units = CHUNK_BYTES // 8
    count = tl.cdiv(total, units)
    attempts = 0
    running = True
    while (step >= FIRST_STEP) & (cursor < CAP * count) & (attempts < QUOTA) & running:
        current = tl.atomic_add(LAYER, 0, sem='acquire')
        if current >= (SECTION + 1) * SECTION_LAYERS:
            tl.atomic_add(STATS + 2, 1, sem='relaxed')
            running = False
        elif tl.atomic_add(BUSY, 0, sem='acquire') != 0:
            tl.atomic_add(STATS + 1, 1, sem='relaxed')
            running = False
        else:
            slot, chunk = cursor // count, cursor % count
            gid = tl.load(RECORDS + SECTION * SLOTS + slot)
            if gid < 0:
                cursor = (slot + 1) * count
            else:
                begin = chunk * units
                end = tl.minimum(begin + units, total)
                stage_slot = (SECTION % 2) * SLOTS + slot
                copied = begin * 0  # int64 like `begin`: a loop-carried scalar must keep one type
                offset = 0
                for bank in tl.static_range(NUM_BANKS):
                    size = tl.load(SIZES + bank) // 8
                    lo = tl.maximum(begin - offset, 0)
                    hi = tl.minimum(end - offset, size)
                    src = tl.load(HOSTS + (gid // E) * NUM_BANKS + bank).to(tl.pointer_type(tl.uint64))
                    dst = tl.load(STAGES + bank).to(tl.pointer_type(tl.uint64))
                    for start in range(lo + pid * BLOCK, hi, STRIPES * BLOCK):
                        ix = start + tl.arange(0, BLOCK)
                        value = tl.load(src + (gid % E) * size + ix, ix < hi, other=0)
                        tl.store(dst + stage_slot * size + ix, value, ix < hi)
                        copied += tl.minimum(hi - start, BLOCK)
                    offset += size
                # All threads of this CTA have issued their stores before the counters move.
                tl.debug_barrier()
                tl.atomic_add(STATS, copied * 8, sem='relaxed')
                previous = tl.atomic_add(PAYLOAD + SECTION * SLOTS + slot, 1, sem='acq_rel')
                if previous == count * STRIPES - 1:
                    # The acq_rel counter chain makes every CTA's stripes visible to this CTA.
                    # Publish the id first and the generation last, both with release:
                    # demand_copy acquires READY before it reads the staged bytes.
                    tl.atomic_xchg(IDS + stage_slot, gid.to(tl.int32), sem='release')
                    tl.atomic_xchg(READY + stage_slot, tl.load(GEN), sem='release')
                    tl.atomic_add(STATS + 3, 1, sem='relaxed')
                    if COUNT_LAYERS:
                        tl.atomic_add(LAYER_PUBLISHED + gid // E, 1, sem='relaxed')
                cursor += 1
            tl.store(cursor_ptr, cursor)
        attempts += 1
