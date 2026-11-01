"""--expert-load bounded: same tensors as the parallel reader, a smaller pending set, safe fallback.

CPU-only. The real-checkpoint test reads safetensors headers only and skips without the files.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from freetoken.models import weight
from freetoken.models.weight import expert_shard_order, iter_expert_tensors_bounded, iter_expert_tensors_parallel

# A local snapshot of Qwen/Qwen3.5-35B-A3B-FP8 at revision 9d1823d2dee688a6b25e77009dc727688c44936e.
# The checkpoint test skips when FREETOKEN_TEST_QWEN35_FP8_DIR is unset.
QWEN_FP8 = os.environ.get("FREETOKEN_TEST_QWEN35_FP8_DIR", "")
# Logged by the EXP513 server (2026-10-10) for that checkpoint: the order the measured runs read in.
EXP513_ORDER = [f"model.safetensors-{i:05d}-of-00014.safetensors" for i in (13, 8, 9, 1, 2, 10, 3, 4, 11, 5, 6, 12, 7)]


def _expert(layer, e, proj):
    return f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"


def _pending_peak(shards, sizes, order):
    need, got, held, cur, peak = {}, {}, {}, 0, 0
    for names in shards.values():
        for n in names:
            need[n.rsplit(".", 2)[0]] = need.get(n.rsplit(".", 2)[0], 0) + 1
    for shard in order:
        for n in shards[shard]:
            key = n.rsplit(".", 2)[0]
            got[key] = got.get(key, 0) + 1
            held[key] = held.get(key, 0) + sizes[n]
            cur += sizes[n]
            if got[key] == need[key]:
                cur -= held.pop(key)
        peak = max(peak, cur)
    return peak


def test_greedy_order_completes_experts_early():
    # gate/up of experts 0-3 in shards a and b, all down weights in shard c (the Qwen FP8 pattern)
    shards = {
        "a": [_expert(0, e, "gate_proj") for e in range(4)] + [_expert(0, e, "up_proj") for e in range(2)],
        "b": [_expert(0, e, "up_proj") for e in range(2, 4)],
        "c": [_expert(0, e, "down_proj") for e in range(4)],
    }
    sizes = {n: 10 for names in shards.values() for n in names}
    order = expert_shard_order(shards, sizes)
    assert sorted(order) == ["a", "b", "c"]
    assert _pending_peak(shards, sizes, order) <= _pending_peak(shards, sizes, sorted(shards))


def test_mixed_tensor_counts_keep_file_order():
    shards = {"b": [_expert(0, 0, "gate_proj"), _expert(0, 0, "up_proj")], "a": [_expert(0, 1, "gate_proj")]}
    assert expert_shard_order(shards, {n: 1 for v in shards.values() for n in v}) == ["a", "b"]


def test_stacked_expert_tensors_keep_file_order():
    shards = {"b": ["model.layers.0.mlp.experts.gate_up_proj"], "a": ["model.layers.1.mlp.experts.gate_up_proj"]}
    assert expert_shard_order(shards, {n: 1 for v in shards.values() for n in v}) == ["a", "b"]


@pytest.mark.skipif(not os.path.isdir(QWEN_FP8), reason="needs the Qwen3.5-35B-A3B-FP8 checkpoint")
def test_real_checkpoint_order_matches_measured_runs():
    import struct

    from freetoken.models.loader import safetensors_weight_map

    shards = {}
    for name, shard in safetensors_weight_map(QWEN_FP8).items():
        # the language-model experts only: FreeToken does not load the MTP layer (mtp.layers.0)
        if name.startswith("model.language_model.layers.") and ".mlp.experts." in name:
            shards.setdefault(shard, []).append(name)
    sizes = {}
    for shard, names in shards.items():
        with open(os.path.join(QWEN_FP8, shard), "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        sizes.update({n: header[n]["data_offsets"][1] - header[n]["data_offsets"][0] for n in names})
    order = expert_shard_order(shards, sizes)
    assert order == EXP513_ORDER
    assert _pending_peak(shards, sizes, order) < _pending_peak(shards, sizes, sorted(shards)) / 4


def _write_checkpoint(folder):
    tensors = {}
    gen = torch.Generator().manual_seed(0)
    for e in range(3):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            tensors[_expert(0, e, proj)] = torch.randn(4, 8, generator=gen).to(torch.bfloat16)
    tensors["model.embed_tokens.weight"] = torch.randn(5, 8, generator=gen).to(torch.bfloat16)
    names = sorted(tensors)
    split = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
             "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    weight_map = {}
    for shard, part in split.items():
        save_file({n: tensors[n] for n in part}, str(folder / shard))
        weight_map.update({n: shard for n in part})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return tensors


def _is_expert(name):
    return ".experts." in name


def test_bounded_reader_yields_the_same_tensors(tmp_path):
    tensors = _write_checkpoint(tmp_path)
    plain = {n: t.clone() for n, t in iter_expert_tensors_parallel(str(tmp_path), _is_expert, workers=2, chunk=4096)}
    bounded = dict(iter_expert_tensors_bounded(str(tmp_path), _is_expert, workers=2, chunk=4096))
    assert set(plain) == set(bounded) == {n for n in tensors if _is_expert(n)}
    for name, value in bounded.items():
        assert value.dtype == tensors[name].dtype and torch.equal(value, tensors[name]) and torch.equal(value, plain[name])


def test_switch_routes_the_parallel_reader(tmp_path, monkeypatch):
    _write_checkpoint(tmp_path)
    calls = []
    real = weight.iter_expert_tensors_bounded

    def spy(*args, **kwargs):
        calls.append(args[0])
        yield from real(*args, **kwargs)

    monkeypatch.setattr(weight, "iter_expert_tensors_bounded", spy)
    weight.set_bounded_reader(True)
    try:
        assert len(list(iter_expert_tensors_parallel(str(tmp_path), _is_expert, workers=2, chunk=4096))) == 9
    finally:
        weight.set_bounded_reader(False)
    assert calls == [str(tmp_path)]
    list(iter_expert_tensors_parallel(str(tmp_path), _is_expert, workers=2, chunk=4096))
    assert len(calls) == 1


def test_engine_maps_bounded_to_the_parallel_path():
    from types import SimpleNamespace

    from freetoken.engine.engine import _expert_parallel

    try:
        assert _expert_parallel(SimpleNamespace(expert_load="bounded")) is True and weight._bounded_reader is True
        assert _expert_parallel(SimpleNamespace(expert_load="parallel")) is True and weight._bounded_reader is False
        assert _expert_parallel(SimpleNamespace(expert_load="auto")) is None and weight._bounded_reader is False
    finally:
        weight.set_bounded_reader(False)
