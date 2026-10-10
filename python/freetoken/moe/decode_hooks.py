"""Extension point for decode-time expert prefetch on the offload MoE cache.

A ``DecodeHooks`` object sees each decode token and each MoE layer of it, and may replace the
demand copy of a layer. The engine owns the call order; a hook never patches engine code.
Callbacks that run inside the decode forward (``observe``, ``copy_missing``) are captured
into the decode CUDA graph, so they must enqueue device work only: no host syncs, no
host-side branches on device values, and fixed tensor shapes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.moe.offload_cache import OffloadMoeCache


class DecodeHooks:
    """Default implementation: every callback is a no-op and the engine's own copy runs."""

    # True: the hooks keep per-request state, so the engine refuses batches of more than one request
    single_request = True

    def reserve(self, engine) -> None:
        """Allocate device memory before the cache is sized; the engine charges the bytes."""

    def bind(self, engine, cache: OffloadMoeCache) -> None:
        """Called after the offload cache exists and again before every CUDA graph recapture
        (after any cache rebuild), each time before the capture. Graph warmup and capture call ``observe`` and ``copy_missing``
        without any request or step callback, so bind must leave all device state valid."""

    def begin_request(self, uid: int) -> None:
        """Called before every prefill forward (one per chunk) and when the decoding request changes."""

    def before_decode(self, step: int) -> None:
        """Called before decode step ``step`` (0 = the first decoded token) runs its forward."""

    def after_decode(self, step: int, consumed: torch.Tensor, sampled: torch.Tensor) -> None:
        """Called after decode step ``step``: ``consumed`` is its input token, ``sampled`` the next."""

    def observe(self, layer_id: int, hidden: torch.Tensor, router_logits: torch.Tensor | None,
                topk_ids: torch.Tensor) -> None:
        """Called per decode MoE layer before ``topk_ids`` is rewritten into slot ids."""

    def copy_missing(self, cache: OffloadMoeCache, layer_id: int) -> bool:
        """Return True after enqueueing this layer's whole copy schedule; False runs the engine's copy.

        Called on every decode MoE layer, also when nothing is missing. True means the hook
        also made the current stream wait for its copy before the expert GEMM.
        """
        return False


def charge_reservation(engine, hooks: DecodeHooks) -> int:
    """Run ``hooks.reserve`` and add the device bytes it took to the engine's weight bytes.

    The cache budget is computed from free memory after the weights, so memory a hook holds
    must be counted as weights; otherwise the auto-sized cache would overcommit.
    """
    before = engine._sync_get_memory()[0]
    hooks.reserve(engine)
    charged = max(before - engine._sync_get_memory()[0], 0)
    engine._weights_bytes += charged
    return charged
