"""memory_phase is a read-only, machine-readable memory reading (CPU test, CUDA queries faked)."""

from __future__ import annotations

import json

import pytest
import torch

from freetoken.engine import graph


@pytest.fixture
def fake_cuda(monkeypatch):
    calls: list[str] = []

    def forbidden(name):
        def call(*args, **kwargs):
            calls.append(name)
        return call

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (150_323_855, 17_171_480_576))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 14_565_000_000)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 14_654_000_000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 14_700_000_000)
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats", "reset_max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, forbidden(name))
    return calls


def test_reading_is_raw_bytes_in_one_parseable_line(fake_cuda):
    line = graph.memory_phase(torch.device("cuda", 0), "after_init")
    tag, payload = line.split(" ", 1)
    reading = json.loads(payload)
    assert tag == "MEMORY_PHASE"
    assert "\n" not in line
    assert reading["phase"] == "after_init"
    assert (reading["allocated"], reading["reserved"], reading["peak_reserved"]) == (14_565_000_000, 14_654_000_000, 14_700_000_000)
    assert (reading["device_free"], reading["device_total"]) == (150_323_855, 17_171_480_576)
    assert all(isinstance(reading[k], int) for k in ("allocated", "reserved", "peak_reserved", "device_free", "device_total"))
    assert isinstance(reading["wall"], float)


def test_reading_does_not_change_allocator_state(fake_cuda):
    # Engine._sync_get_memory synchronizes, empties the cache and resets the peaks; a reading that
    # did the same would move the very numbers it is meant to attribute.
    graph.memory_phase(torch.device("cuda", 0), "before_load")
    assert fake_cuda == []
