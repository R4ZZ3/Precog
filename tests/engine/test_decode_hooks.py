"""--expert-predictor plumbing: setting checks, memory charge, call order of the decode hooks.

CPU-only: the engine and the MoE layer are driven through stand-in objects.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.engine import Engine, _load_decode_hooks
from freetoken.layers.moe import OffloadMoELayer
from freetoken.moe.decode_hooks import DecodeHooks, charge_reservation


def _cfg(**change):
    cfg = dict(expert_predictor="some/predictor", moe_strategy="offload", moe_cpu_layers=None, max_running_req=1,
               moe_demand_copy_stripes=16, cuda_graph_bs=None, cuda_graph_max_bs=None,
               model_config=SimpleNamespace(num_moe_layers=40, num_experts=256))
    cfg.update(change)
    return SimpleNamespace(**cfg)


def test_no_flag_no_hooks():
    assert _load_decode_hooks(_cfg(expert_predictor=None, moe_strategy="cpu", max_running_req=4,
                                   moe_demand_copy_stripes=0)) is None


def test_stripes_without_predictor_build_striped_hooks():
    from freetoken.moe.striped_demand import StripedDemandHooks

    hooks = _load_decode_hooks(_cfg(expert_predictor=None, max_running_req=4))
    assert isinstance(hooks, StripedDemandHooks) and hooks.shape == (40, 256, 16)


@pytest.mark.parametrize("graphs", [dict(cuda_graph_bs=[1, 2]), dict(cuda_graph_max_bs=4)])
def test_predictor_refuses_graph_batches_above_one(graphs):
    with pytest.raises(ValueError, match="CUDA graph batch sizes up to"):
        _load_decode_hooks(_cfg(**graphs))


def test_refuses_unsupported_settings_and_names_each():
    with pytest.raises(ValueError) as err:
        _load_decode_hooks(_cfg(moe_strategy="hybrid", moe_cpu_layers="8", max_running_req=4, moe_demand_copy_stripes=0))
    message = str(err.value)
    assert "--moe-demand-copy-stripes 0" in message
    assert "--moe-strategy hybrid" in message
    assert "--moe-cpu-layers" in message
    assert "--max-running-requests 4" in message


def test_missing_package_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "commitmoe_prefetch", None)  # makes the import fail
    # an earlier test may have imported the submodule; a cached submodule is returned without its parent
    monkeypatch.setitem(sys.modules, "commitmoe_prefetch.freetoken", None)
    with pytest.raises(RuntimeError, match="commitmoe-prefetch"):
        _load_decode_hooks(_cfg())


class _Reserving(DecodeHooks):
    def __init__(self, engine, nbytes):
        self.engine, self.nbytes = engine, nbytes

    def reserve(self, engine):
        self.engine.free -= self.nbytes


def _memory_engine(free):
    engine = SimpleNamespace(free=free, _weights_bytes=1000)
    engine._sync_get_memory = lambda: (engine.free, engine.free)
    return engine


def test_reservation_is_charged_to_weights():
    engine = _memory_engine(10_000)
    assert charge_reservation(engine, _Reserving(engine, 300)) == 300
    assert engine._weights_bytes == 1300


def test_freed_memory_is_not_a_negative_charge():
    engine = _memory_engine(10_000)
    assert charge_reservation(engine, _Reserving(engine, -500)) == 0
    assert engine._weights_bytes == 1000


class _Recorder(DecodeHooks):
    def __init__(self, handled=False):
        self.calls, self.handled = [], handled

    def begin_request(self, uid):
        self.calls.append(("begin", uid))

    def before_decode(self, step):
        self.calls.append(("before", step))

    def observe(self, layer_id, hidden, router_logits, topk_ids):
        self.calls.append(("observe", layer_id, topk_ids.tolist()))

    def copy_missing(self, cache, layer_id):
        self.calls.append(("copy", layer_id))
        return self.handled


def _batch(phase, uid, size=1):
    return SimpleNamespace(
        size=size, reqs=[SimpleNamespace(uid=uid)] * size, is_prefill=phase == "prefill",
        is_decode=phase == "decode", input_ids=torch.tensor([7], dtype=torch.int32),
    )


def _engine():
    return SimpleNamespace(_hooks_uid=None, _hooks_step=0)


def test_request_and_step_tracking():
    engine, hooks = _engine(), _Recorder()
    step = Engine._hooks_before_forward
    assert step(engine, hooks, _batch("prefill", 1)) is None
    assert step(engine, hooks, _batch("prefill", 1)) is None  # a second prefill chunk resets again
    consumed = step(engine, hooks, _batch("decode", 1))
    assert consumed.tolist() == [7]
    engine._hooks_step += 1  # forward_batch advances the step after after_decode
    step(engine, hooks, _batch("decode", 1))
    step(engine, hooks, _batch("decode", 2))  # a switch without prefill still resets
    assert hooks.calls == [("begin", 1), ("begin", 1), ("before", 0), ("before", 1), ("begin", 2), ("before", 0)]


def test_consumed_ids_are_a_copy():
    batch = _batch("decode", 1)
    consumed = Engine._hooks_before_forward(_engine(), _Recorder(), batch)
    batch.input_ids.fill_(9)
    assert consumed.tolist() == [7]


def test_batch_of_two_is_refused():
    with pytest.raises(RuntimeError, match="batch of 2"):
        Engine._hooks_before_forward(_engine(), _Recorder(), _batch("decode", 1, size=2))


def test_stateless_hooks_serve_batches_without_tracking():
    hooks = _Recorder()
    hooks.single_request = False
    engine = _engine()
    assert Engine._hooks_before_forward(engine, hooks, _batch("decode", 1, size=2)) is None
    assert hooks.calls == [] and engine._hooks_uid is None


class _Cache:
    decode_target = "gpu"

    def __init__(self, hooks, log):
        self.decode_hooks, self.log = hooks, log

    def is_cpu_layer(self, layer_id):
        return False

    def ensure_experts(self, layer_id, topk_ids):
        self.log.append(("ensure", layer_id))
        topk_ids.fill_(-5)  # the real cache rewrites expert ids into slot ids in place

    def copy_missing(self):
        self.log.append(("engine copy",))

    def pool_bank_views(self, layer_id):
        return None

    def alphas_for_slots(self, layer_id):
        return None


def _decode(hooks):
    log = [] if hooks is None else hooks.calls
    cache = _Cache(hooks, log)
    layer = SimpleNamespace(offload_cache=cache, layer_id=3, _expert_gemm=lambda *a, **k: "out")
    ids = torch.tensor([[4, 9]])
    assert OffloadMoELayer._decode_routed(layer, torch.zeros(1, 8), torch.ones(1, 2), ids, torch.zeros(1, 16)) == "out"
    return log


def test_layer_order_with_engine_copy():
    assert _decode(_Recorder(handled=False)) == [
        ("observe", 3, [[4, 9]]), ("ensure", 3), ("copy", 3), ("engine copy",),
    ]


def test_layer_order_when_hook_copies():
    assert _decode(_Recorder(handled=True)) == [("observe", 3, [[4, 9]]), ("ensure", 3), ("copy", 3)]


def test_layer_without_hooks_is_unchanged():
    assert _decode(None) == [("ensure", 3), ("engine copy",)]


def _parse(*extra):
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3VLForConditionalGeneration"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        return parse_args(["--model", "/models/anon", *extra])[0]


def test_flags_default_off():
    args = _parse()
    assert args.expert_predictor is None
    assert args.pretouch_host_banks is False and args.moe_demand_copy_stripes == 0


def test_flags_parse():
    args = _parse("--expert-predictor", "RASMUS/predictor", "--pretouch-host-banks",
                  "--moe-demand-copy-stripes", "16")
    assert args.expert_predictor == "RASMUS/predictor"
    assert args.pretouch_host_banks is True and args.moe_demand_copy_stripes == 16


# _refuse_hooks_bypass: the hooks run only on the GPU offload path of OffloadMoELayer.decode. A CPU layer or a
# hybrid target returns before hooks.observe/copy_missing, and learned_ensure needs one global slot space. The
# fake caches below stand for the cache state after every engine resolution step: the explicit flags
# (--moe-strategy hybrid, --moe-cpu-layers 3) and the auto cases (--moe-cpu-layers auto or the pin probe resolving
# CPU layers while the decode target stays "gpu", locked fetch, several pools from --moe-cache-scope or mixed formats).
def _resolved_cache(**changes):
    state = dict(decode_target="gpu", cpu_layer_ids=frozenset(), locked_hybrid=frozenset(), pool_sizes=[3237])
    state.update(changes)
    return SimpleNamespace(**state)


def test_hooks_accept_a_gpu_offload_cache_with_one_pool():
    from freetoken.engine.engine import _refuse_hooks_bypass

    _refuse_hooks_bypass(_resolved_cache())


@pytest.mark.parametrize("changes, words", [
    (dict(decode_target="hybrid"), "decode target 'hybrid'"),                      # explicit --moe-strategy hybrid
    (dict(decode_target="cpu", cpu_layer_ids=frozenset(range(40))), "decode target 'cpu'"),
    (dict(cpu_layer_ids=frozenset({3})), "CPU or locked MoE layers [3]"),          # explicit --moe-cpu-layers
    (dict(cpu_layer_ids=frozenset({30, 31})), "CPU or locked MoE layers [30, 31]"),  # auto: probe resolved layers
    (dict(decode_target="hybrid", cpu_layer_ids=frozenset({5}), locked_hybrid=frozenset({5})),
     "locked-fetch layers [5]"),
    (dict(pool_sizes=[1600, 1637]), "2 MoE slot pools"),
])
def test_hooks_refuse_every_resolved_bypass(changes, words):
    from freetoken.engine.engine import _refuse_hooks_bypass

    with pytest.raises(ValueError, match="decode hooks") as info:
        _refuse_hooks_bypass(_resolved_cache(**changes))
    assert words in str(info.value)


def test_refusal_runs_before_every_hook_bind():
    # Both bind sites (cache construction and the rebind before each recapture) must check the resolved cache
    # first, so no graph is captured with hooks on a path that skips them.
    import inspect

    from freetoken.engine import engine as eng

    source = inspect.getsource(eng.Engine)
    binds = [i for i in range(len(source)) if source.startswith("self.decode_hooks.bind(", i)]
    assert len(binds) == 2
    for at in binds:
        before = source[:at]
        assert before.rfind("_refuse_hooks_bypass(") > before.rfind("\n    def "), "bind without a refusal check"
