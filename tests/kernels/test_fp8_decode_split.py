"""The two-pass FP8 decode (EXP519 direction 1) against the one-pass decode: bit-identical output.

Pass 1 computes the resident routes, ``join`` waits for the copy, pass 2 computes the missed
routes. The race test starts the missed experts' rows as NaN and copies the real rows in on a
side stream after a delay: if pass 2 ran before the join, NaN would reach the output. (Pass 1
reading a missed row would only waste work: pass 2 recomputes every missed route.)
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SLOTS, H, I, TOPK = 24, 256, 256, 8
BLOCK = 128


def _quant_block(w):
    n, k = w.shape
    blocks = w.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / 448.0
    q = (blocks / scale).to(torch.float8_e4m3fn).view(n, k)
    return q, scale.squeeze(1).squeeze(-1).to(torch.bfloat16)


def _banks(seed):
    torch.manual_seed(seed)
    gu = [_quant_block(torch.randn(2 * I, H, device="cuda") / H**0.5) for _ in range(SLOTS)]
    dn = [_quant_block(torch.randn(H, I, device="cuda") / I**0.5) for _ in range(SLOTS)]
    stack = lambda xs, i: torch.stack([x[i] for x in xs]).contiguous()
    return [stack(gu, 0), stack(gu, 1), stack(dn, 0), stack(dn, 1)]


def _routing(m, generator):
    ids = torch.stack([torch.randperm(SLOTS, generator=generator)[:TOPK] for _ in range(m)]).to(torch.int32).cuda()
    w = torch.softmax(torch.randn(m, TOPK, generator=generator), dim=-1).cuda()
    return w, ids


@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("kind", ["mixed", "all_resident", "all_missed"])
def test_split_equals_one_pass(m, kind):
    from freetoken.kernel.triton.fp8_blockscale_moe import (
        fused_experts_decode_fp8_blockscale,
        fused_experts_decode_fp8_blockscale_split,
    )

    banks = _banks(0)
    generator = torch.Generator().manual_seed(m * 7 + len(kind))
    for _ in range(5):
        w, ids = _routing(m, generator)
        x = torch.randn(m, H, generator=generator).to("cuda", torch.bfloat16)
        if kind == "mixed":
            resident = (torch.rand(m * TOPK, generator=generator) < 0.7).to(torch.int32).cuda()
        else:
            resident = torch.full((m * TOPK,), int(kind == "all_resident"), dtype=torch.int32, device="cuda")
        joins = []
        one = fused_experts_decode_fp8_blockscale(x, *banks, w, ids)
        two = fused_experts_decode_fp8_blockscale_split(x, *banks, w, ids, resident, lambda: joins.append(1))
        assert joins == [1]
        assert torch.equal(one, two)


def test_missed_routes_wait_for_the_join():
    from freetoken.kernel.triton.fp8_blockscale_moe import (
        fused_experts_decode_fp8_blockscale,
        fused_experts_decode_fp8_blockscale_split,
    )

    real = _banks(1)
    generator = torch.Generator().manual_seed(519)
    w, ids = _routing(1, generator)
    x = torch.randn(1, H, generator=generator).to("cuda", torch.bfloat16)
    expected = fused_experts_decode_fp8_blockscale(x, *real, w, ids)
    resident = torch.tensor([1, 0, 1, 1, 0, 1, 1, 0], dtype=torch.int32, device="cuda")
    missed_slots = ids.flatten()[resident == 0].long()
    work = [b.clone() for b in real]
    for b in work:  # the missed experts' rows hold NaN until the copy lands
        b[missed_slots] = torch.tensor(float("nan"), dtype=torch.float32).to(b.dtype)
    torch.cuda.synchronize()

    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        torch.cuda._sleep(50_000_000)  # long enough that pass 1 finishes before the copy lands
        for b, r in zip(work, real):
            b[missed_slots] = r[missed_slots]
    out = fused_experts_decode_fp8_blockscale_split(x, *work, w, ids, resident, lambda: main.wait_stream(side))
    torch.cuda.synchronize()
    assert torch.equal(out, expected)


def test_split_under_graph_replay_at_release_shapes():
    """Qwen3.5 expert shapes (hidden 2048, intermediate 512), scale banks laid out by the serve's
    padding rule (no padding is needed at these shapes), captured in a CUDA graph with a copy-stream fork and join placed as ``copy_unjoined`` places
    them. Replays change the routes, the resident mask and the hidden state in place."""
    from freetoken.kernel.triton.fp8_blockscale_moe import (
        fused_experts_decode_fp8_blockscale,
        fused_experts_decode_fp8_blockscale_split,
    )
    from freetoken.layers.quantization.moe.fp8_block import _pad_scale

    slots, h, i = 24, 2048, 512
    torch.manual_seed(2)

    def bank(n, k):
        q, s = zip(*[_quant_block(torch.randn(n, k, device="cuda") / k**0.5) for _ in range(slots)])
        padded = torch.zeros((slots, n // BLOCK, _pad_scale(n // BLOCK, k // BLOCK)), dtype=torch.bfloat16, device="cuda")
        padded[:, :, : k // BLOCK].copy_(torch.stack(s))
        return torch.stack(q).contiguous(), padded

    banks = [*bank(2 * i, h), *bank(h, i)]
    x = torch.zeros((1, h), dtype=torch.bfloat16, device="cuda")
    w = torch.zeros((1, TOPK), device="cuda")
    ids = torch.zeros((1, TOPK), dtype=torch.int32, device="cuda")
    resident = torch.zeros(TOPK, dtype=torch.int32, device="cuda")
    copy_stream = torch.cuda.Stream()
    flag = torch.zeros(1, device="cuda")

    def forward():
        main = torch.cuda.current_stream()
        copy_stream.wait_stream(main)
        with torch.cuda.stream(copy_stream):
            flag.add_(1)  # stands in for the demand copy kernel
        one = fused_experts_decode_fp8_blockscale(x, *banks, w, ids)
        two = fused_experts_decode_fp8_blockscale_split(x, *banks, w, ids, resident, lambda: main.wait_stream(copy_stream))
        return one, two

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            one, two = forward()
    torch.cuda.current_stream().wait_stream(stream)

    generator = torch.Generator().manual_seed(21)
    masks = [torch.ones(TOPK), torch.zeros(TOPK)] + [(torch.rand(TOPK, generator=generator) < 0.6).float() for _ in range(6)]
    for mask in masks:
        x.copy_(torch.randn((1, h), generator=generator))
        w.copy_(torch.softmax(torch.randn((1, TOPK), generator=generator), dim=-1))
        ids.copy_(torch.randperm(slots, generator=generator)[:TOPK].view(1, TOPK))
        resident.copy_(mask)
        graph.replay()
        torch.cuda.synchronize()
        assert one.abs().sum() > 0
        assert torch.equal(one, two)
        assert torch.equal(two, fused_experts_decode_fp8_blockscale(x, *banks, w, ids))


def test_missed_routes_wait_for_the_join_under_graph_replay():
    """EXP538: the NaN race test inside two CUDA graphs captured from the same stream, copy stream
    and function objects, replayed alternately. Each replay poisons the missed rows first; the
    captured copy stream sleeps, then writes the real rows. A missing join edge in either graph
    lets NaN reach the output."""
    from freetoken.kernel.triton.fp8_blockscale_moe import (
        fused_experts_decode_fp8_blockscale,
        fused_experts_decode_fp8_blockscale_split,
    )

    real = _banks(4)
    work = [b.clone() for b in real]
    missed_count = 3
    x = torch.zeros((1, H), dtype=torch.bfloat16, device="cuda")
    w = torch.zeros((1, TOPK), device="cuda")
    ids = torch.zeros((1, TOPK), dtype=torch.int32, device="cuda")
    resident = torch.ones(TOPK, dtype=torch.int32, device="cuda")
    missed = torch.zeros(missed_count, dtype=torch.int64, device="cuda")
    copy_stream = torch.cuda.Stream()

    def forward():
        main = torch.cuda.current_stream()
        copy_stream.wait_stream(main)
        with torch.cuda.stream(copy_stream):
            torch.cuda._sleep(20_000_000)  # long enough that pass 1 finishes before the copy lands
            for b, r in zip(work, real):  # index_copy_ has no FP8 CUDA kernel; index_put_ has
                b[missed] = r[missed]
        return fused_experts_decode_fp8_blockscale_split(x, *work, w, ids, resident,
                                                         lambda: main.wait_stream(copy_stream))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs, outs = [], []
    with torch.cuda.stream(stream):
        forward()  # eager warm-up, as the graph runner does
        for _ in range(2):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outs.append(forward())
            graphs.append(graph)
    torch.cuda.current_stream().wait_stream(stream)

    generator = torch.Generator().manual_seed(538)
    for replay in range(8):
        w_new, ids_new = _routing(1, generator)
        positions = torch.randperm(TOPK, generator=generator)[:missed_count].cuda()
        x.copy_(torch.randn((1, H), generator=generator))
        w.copy_(w_new)
        ids.copy_(ids_new)
        resident.fill_(1)
        resident[positions] = 0
        missed.copy_(ids_new.flatten()[positions].long())
        for b in work:  # the missed experts' rows hold NaN until the captured copy lands
            b[missed] = torch.tensor(float("nan"), dtype=torch.float32).to(b.dtype)
        torch.cuda.synchronize()
        graphs[replay % 2].replay()
        torch.cuda.synchronize()
        expected = fused_experts_decode_fp8_blockscale(x, *real, w, ids)
        assert torch.isfinite(outs[replay % 2]).all(), replay
        assert torch.equal(outs[replay % 2], expected), replay


# ---- Negative slots (-1): an inactive route contributes exactly zero, and its store obeys the route mask.
# The decode GEMM stores zero for slot < 0 and returns. In the split, MASK_MODE 1 runs the routes with
# resident != 0 and MASK_MODE 2 the others; a route that its pass skips must be left untouched, or one pass
# could overwrite the other pass's row. The reference for an inactive route is the same route with a valid
# slot and routing weight 0: its down-GEMM row is acc * 0 = +-0, so the reduced output is equal
# (torch.equal treats -0 as 0).


def _inactive(ids, w, off):
    """ids with -1 at the routes in off (bool, the shape of ids), and the reference weights with 0 there."""
    return torch.where(off, torch.full_like(ids, -1), ids), torch.where(off, torch.zeros_like(w), w)


def _half_resident(generator):
    mask = torch.zeros(TOPK, dtype=torch.int32)
    mask[torch.randperm(TOPK, generator=generator)[: TOPK // 2]] = 1
    return mask


@pytest.mark.parametrize("gemm", ["gate_up", "down"])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_negative_slot_store_obeys_the_route_mask(gemm, mode):
    """_decode_gemm on a sentinel-filled output: a route its pass runs gets 0 (slot -1) or exactly the
    unmasked value (valid slot); a route its pass skips keeps the sentinel, also when its slot is -1."""
    from freetoken.kernel.triton.fp8_blockscale_moe import _decode_gemm

    m, sentinel = 2, 7.0
    banks = _banks(5)
    generator = torch.Generator().manual_seed(100 + mode)
    w, ids = _routing(m, generator)
    resident = torch.cat([_half_resident(generator) for _ in range(m)]).cuda()
    off = torch.zeros(m * TOPK, dtype=torch.bool)
    for res in (0, 1):  # two -1 routes among the resident and two among the missed routes of each pass
        off[torch.nonzero(resident.cpu() == res).flatten()[torch.randperm(m * TOPK // 2, generator=generator)[:4]]] = True
    neg_ids = torch.where(off.view(m, TOPK).cuda(), torch.full_like(ids, -1), ids)
    if gemm == "gate_up":
        a = torch.randn(m, H, generator=generator).to("cuda", torch.bfloat16)
        wq, ws, n, mul, a_row_is_route = banks[0], banks[1], 2 * I, False, False
    else:
        a = torch.randn(m * TOPK, I, generator=generator).to("cuda", torch.bfloat16)
        wq, ws, n, mul, a_row_is_route = banks[2], banks[3], H, True, True
    full = torch.empty((m, TOPK, n), device="cuda", dtype=torch.bfloat16)
    _decode_gemm(a, wq, ws, full, w, ids, mul, a_row_is_route)  # every route, valid slots
    out = torch.full((m, TOPK, n), sentinel, device="cuda", dtype=torch.bfloat16)
    _decode_gemm(a, wq, ws, out, w, neg_ids, mul, a_row_is_route, resident if mode else None, mode)
    torch.cuda.synchronize()
    run = torch.ones(m * TOPK, dtype=torch.bool) if mode == 0 else (resident.cpu() != 0) == (mode == 1)
    out, full = out.view(m * TOPK, n).cpu(), full.view(m * TOPK, n).cpu()
    for r in range(m * TOPK):
        if not run[r]:
            assert torch.equal(out[r], torch.full_like(out[r], sentinel)), r
        elif off[r]:
            assert torch.equal(out[r], torch.zeros_like(out[r])), r
        else:
            assert torch.equal(out[r], full[r]), r
    assert bool((off & run).any()) and bool((off & ~run).any() or mode == 0)


@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("owner", ["pass1", "pass2", "both"])
def test_negative_slots_equal_zero_weight_routes(m, owner):
    """The one-pass and the split decode with three -1 routes per token (five active routes kept) equal the
    one-pass decode with valid slots and routing weight 0 at those routes."""
    from freetoken.kernel.triton.fp8_blockscale_moe import (
        fused_experts_decode_fp8_blockscale,
        fused_experts_decode_fp8_blockscale_split,
    )

    banks = _banks(0)
    generator = torch.Generator().manual_seed(17 * m + len(owner))
    for _ in range(5):
        w, ids = _routing(m, generator)
        x = torch.randn(m, H, generator=generator).to("cuda", torch.bfloat16)
        off = torch.zeros(m, TOPK, dtype=torch.bool)
        off[:, :3] = True
        off = off[:, torch.randperm(TOPK, generator=generator)].cuda()
        resident = (torch.rand(m, TOPK, generator=generator) < 0.5).to(torch.int32).cuda()
        if owner != "both":  # the pass that runs a route: resident 1 -> pass 1, resident 0 -> pass 2
            resident = torch.where(off, torch.full_like(resident, int(owner == "pass1")), resident)
        neg_ids, ref_w = _inactive(ids, w, off)
        ref = fused_experts_decode_fp8_blockscale(x, *banks, ref_w, ids)
        assert ref.abs().sum() > 0
        joins = []
        one = fused_experts_decode_fp8_blockscale(x, *banks, w, neg_ids)
        two = fused_experts_decode_fp8_blockscale_split(x, *banks, w, neg_ids, resident.flatten(),
                                                        lambda: joins.append(1))
        assert joins == [1]
        assert torch.equal(one, ref)
        assert torch.equal(two, ref)


def test_all_negative_slots_give_zero():
    from freetoken.kernel.triton.fp8_blockscale_moe import (
        fused_experts_decode_fp8_blockscale,
        fused_experts_decode_fp8_blockscale_split,
    )

    banks = _banks(1)
    generator = torch.Generator().manual_seed(3)
    w, _ = _routing(2, generator)
    x = torch.randn(2, H, generator=generator).to("cuda", torch.bfloat16)
    ids = torch.full((2, TOPK), -1, dtype=torch.int32, device="cuda")
    resident = torch.tensor([1, 0] * TOPK, dtype=torch.int32, device="cuda")
    assert torch.equal(fused_experts_decode_fp8_blockscale(x, *banks, w, ids), torch.zeros_like(x))
    assert torch.equal(fused_experts_decode_fp8_blockscale_split(x, *banks, w, ids, resident, lambda: None),
                       torch.zeros_like(x))


def test_negative_slots_under_graph_replay_at_release_shapes():
    """One graph holds the one-pass and the split decode at the release shapes (hidden 2048, intermediate
    512). Replays run valid -> -1 -> valid -> ... and draw a new resident mask (4 resident, 4 missed) each
    time, so the pass that owns a route changes. Each -1 replay follows a valid replay, so the graph's
    intermediate rows hold that replay's non-zero values: a -1 route whose zero store did not run in its pass
    leaves a stale row, and a valid replay after a -1 replay shows any row that stayed zero."""
    from freetoken.kernel.triton.fp8_blockscale_moe import (
        fused_experts_decode_fp8_blockscale,
        fused_experts_decode_fp8_blockscale_split,
    )
    from freetoken.layers.quantization.moe.fp8_block import _pad_scale

    slots, h, i = 24, 2048, 512
    torch.manual_seed(6)

    def bank(n, k):
        q, s = zip(*[_quant_block(torch.randn(n, k, device="cuda") / k**0.5) for _ in range(slots)])
        padded = torch.zeros((slots, n // BLOCK, _pad_scale(n // BLOCK, k // BLOCK)), dtype=torch.bfloat16, device="cuda")
        padded[:, :, : k // BLOCK].copy_(torch.stack(s))
        return torch.stack(q).contiguous(), padded

    banks = [*bank(2 * i, h), *bank(h, i)]
    x = torch.zeros((1, h), dtype=torch.bfloat16, device="cuda")
    w = torch.zeros((1, TOPK), device="cuda")
    ids = torch.zeros((1, TOPK), dtype=torch.int32, device="cuda")
    resident = torch.zeros(TOPK, dtype=torch.int32, device="cuda")

    def forward():
        one = fused_experts_decode_fp8_blockscale(x, *banks, w, ids)
        two = fused_experts_decode_fp8_blockscale_split(x, *banks, w, ids, resident, lambda: None)
        return one, two

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            one, two = forward()
    torch.cuda.current_stream().wait_stream(stream)

    generator = torch.Generator().manual_seed(23)
    negative_replays = 0
    for replay in range(9):
        w_new = torch.softmax(torch.randn((1, TOPK), generator=generator), dim=-1).cuda()
        ids_new = torch.randperm(slots, generator=generator)[:TOPK].view(1, TOPK).to(torch.int32).cuda()
        x_new = torch.randn((1, h), generator=generator).to("cuda", torch.bfloat16)
        mask = _half_resident(generator).cuda()
        off = torch.zeros(TOPK, dtype=torch.bool, device="cuda")
        if replay % 2 == 1:  # -1 at two resident routes (pass 1) and two missed routes (pass 2)
            off[torch.nonzero(mask == 1).flatten()[:2]] = True
            off[torch.nonzero(mask == 0).flatten()[:2]] = True
            negative_replays += 1
        neg_ids, ref_w = _inactive(ids_new, w_new, off.view(1, TOPK))
        x.copy_(x_new)
        w.copy_(w_new)
        ids.copy_(neg_ids)
        resident.copy_(mask)
        graph.replay()
        torch.cuda.synchronize()
        expected = fused_experts_decode_fp8_blockscale(x_new, *banks, ref_w, ids_new)
        assert expected.abs().sum() > 0, replay
        assert torch.equal(one, expected), replay
        assert torch.equal(two, expected), replay
    assert negative_replays == 4
