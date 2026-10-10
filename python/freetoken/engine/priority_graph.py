"""CUDA graphs instantiated with per-node priorities (--cuda-graph-node-priority).

PyTorch instantiates captured graphs without cudaGraphInstantiateFlagUseNodePriority, so every
kernel node runs at default priority on replay and the stream priorities seen at capture are
lost. With this flag, side-stream work captured from a high-priority stream (for example an
expert copy that the next layer waits on) keeps its priority inside the replayed graph.
"""
from __future__ import annotations

import ctypes
import functools

import torch

# cudaGraphInstantiateFlagUseNodePriority in driver_types.h
_USE_NODE_PRIORITY = 8


@functools.cache
def _cudart() -> ctypes.CDLL:
    """The CUDA runtime that PyTorch loaded; a second copy would not own PyTorch's graphs."""
    torch.cuda.init()
    with open("/proc/self/maps") as maps:
        paths = sorted({line.split()[-1] for line in maps if "/libcudart.so" in line})
    if len(paths) != 1:
        raise RuntimeError(f"--cuda-graph-node-priority: expected one loaded libcudart, found {paths}")
    lib = ctypes.CDLL(paths[0])
    lib.cudaGraphInstantiateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_ulonglong]
    lib.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]
    return lib


class PriorityGraph(torch.cuda.CUDAGraph):
    """A CUDAGraph whose replay launches an executable instantiated with node priorities.

    PyTorch still owns the captured graph and its memory pool (keep_graph=True keeps the raw
    graph for our instantiation); only instantiation and launch go through the runtime here.
    """

    def __new__(cls, keep_graph: bool = False):
        obj = super().__new__(cls, keep_graph=True)
        obj._priority_exec = None
        return obj

    def __init__(self, keep_graph: bool = False):
        super().__init__(keep_graph=True)

    def instantiate(self) -> None:
        self._destroy_exec()
        handle = ctypes.c_void_p()
        rc = _cudart().cudaGraphInstantiateWithFlags(ctypes.byref(handle), self.raw_cuda_graph(), _USE_NODE_PRIORITY)
        if rc != 0:
            raise RuntimeError(f"cudaGraphInstantiateWithFlags failed with CUDA error {rc}")
        self._priority_exec = handle

    def replay(self) -> None:
        if self._priority_exec is None:
            self.instantiate()
        rc = _cudart().cudaGraphLaunch(self._priority_exec, torch.cuda.current_stream().cuda_stream)
        if rc != 0:
            raise RuntimeError(f"cudaGraphLaunch failed with CUDA error {rc}")

    def _destroy_exec(self) -> None:
        if getattr(self, "_priority_exec", None) is not None:
            _cudart().cudaGraphExecDestroy(self._priority_exec)
            self._priority_exec = None

    def __del__(self):
        self._destroy_exec()
