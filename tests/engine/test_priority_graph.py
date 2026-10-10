"""--cuda-graph-node-priority: a PriorityGraph replays the captured work like a plain CUDAGraph."""

from __future__ import annotations

import pytest
import torch

from freetoken.engine.priority_graph import PriorityGraph

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_replay_matches_and_reruns():
    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    side = torch.cuda.Stream(priority=-1)
    graph = PriorityGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            y = x * 2
        torch.cuda.current_stream().wait_stream(side)
        out.copy_(y + 1)
    x.fill_(3)
    graph.replay()
    graph.replay()  # a second launch reuses the same executable
    torch.cuda.synchronize()
    assert torch.equal(out, torch.full_like(x, 7))
    assert graph._priority_exec is not None


def test_reinstantiate_replaces_executable():
    graph = PriorityGraph()
    x = torch.zeros(4, device="cuda")
    with torch.cuda.graph(graph):
        x.add_(1)
    graph.instantiate()
    graph.instantiate()  # must destroy the first executable, not leak or launch it twice
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(x, torch.ones(4, device="cuda"))
