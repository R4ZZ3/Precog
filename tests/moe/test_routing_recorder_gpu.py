"""GPU preflight of the routing recorder under CUDA-graph capture and replay (EXP519).

One captured graph stands in for the decode forward: per layer it calls ``observe`` and then
overwrites the routed ids with slot ids (as ``ensure_experts`` does), and a side stream writes
the predictor scores and joins the main stream (as the predictor's probe stream does). Replays
change the routed ids, the step row and the residency between steps.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.decode_hooks import DecodeHooks
from freetoken.moe.routing_recorder import RoutingRecorder

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E, K, S, H = 4, 32, 3, 20, 2


class ScoringInner(DecodeHooks):
    """Writes probs on a side stream inside the graph, like the predictor's probe stream."""

    def __init__(self, dev):
        self.probs = torch.zeros((H, E), device=dev)
        self.demand = SimpleNamespace(counters=torch.zeros((L, 3), dtype=torch.int64, device=dev))
        self.side = torch.cuda.Stream(device=dev)
        self.source = torch.zeros((), device=dev)

    def observe(self, layer_id, hidden, router_logits, topk_ids):
        if layer_id == 0:
            main = torch.cuda.current_stream()
            self.side.wait_stream(main)
            with torch.cuda.stream(self.side):
                self.probs.copy_(self.source.expand(H, E))
            main.wait_stream(self.side)


def test_recorder_under_graph_replay(tmp_path):
    dev = torch.device("cuda")
    cache = SimpleNamespace(num_layers=L, num_experts=E, cache_size=S,
                            slot_for_id=torch.full((L, E), -1, dtype=torch.int32, device=dev),
                            id_of_slot=torch.full((S,), -1, dtype=torch.int32, device=dev),
                            usage=torch.zeros(S, dtype=torch.int64, device=dev),
                            step=torch.zeros((), dtype=torch.int64, device=dev))
    inner = ScoringInner(dev)
    rec = RoutingRecorder(inner, tmp_path, top_k=K, capacity=8)
    rec.bind(None, cache)
    static_ids = torch.zeros((L, 1, K), dtype=torch.int64, device=dev)
    routed = torch.zeros((L, 1, K), dtype=torch.int64, device=dev)  # the forward's per-layer topk buffers

    def forward():
        for layer in range(L):
            routed[layer].copy_(static_ids[layer])
            rec.observe(layer, None, None, routed[layer])
            routed[layer].fill_(S + 7)  # ensure_experts rewrites the ids into slot ids

    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()  # warmup: scratch row
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    assert int((rec.ids[:-1] >= 0).sum()) == 0  # warmup and capture wrote only the scratch row

    generator = torch.Generator().manual_seed(519)
    expected_ids, expected_hits = [], []
    rec.begin_request(5)
    cache.usage.fill_(3)
    for step in range(3):
        rec.before_decode(step)
        if step == 0:
            cache.usage.fill_(99)  # after the snapshot: must not appear in it
        new_ids = torch.stack([torch.randperm(E, generator=generator)[:K] for _ in range(L)]).view(L, 1, K)
        resident = torch.rand((L, E), generator=generator) < 0.5
        cache.slot_for_id.copy_(torch.where(resident, 1, -1).to(torch.int32))
        static_ids.copy_(new_ids)
        inner.source.fill_(float(step + 1))
        graph.replay()
        rec.after_decode(step, None, None)
        expected_ids.append(new_ids.view(L, K).tolist())
        expected_hits.append(torch.gather(resident, 1, new_ids.view(L, K)).tolist())
    rec.begin_request(6)
    torch.cuda.synchronize()

    [path] = sorted(tmp_path.glob("request_*.pt"))
    record = torch.load(path)
    assert record["ids"].tolist() == expected_ids
    assert record["hits"].tolist() == expected_hits
    assert record["usage"].tolist() == [3] * S
    assert record["scores"][:, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert record["steps"] == 3 and record["overflow"] is False
