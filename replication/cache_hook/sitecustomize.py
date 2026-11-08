"""Resolved-cache record for the replication serves (replication/run_bench.py puts this folder on PYTHONPATH).
A copy of the EXP602 cache hook of the CommitMoE study, with only the environment variable renamed.

With REPLICATION_CACHE_DIR set, this module adds two record points to freetoken.engine.engine, and appends each record to
$REPLICATION_CACHE_DIR/cache_<pid>.json:

  - source 'check': _refuse_hooks_bypass, as exp593_cache_hook does. It records before the original check runs, so a
    state that the check refuses is recorded too. The engine calls this check only when decode hooks exist, so an arm
    with no hooks has no such record. That is why the second record point exists.
  - source 'init': the return of Engine._init_offload_moe_cache, in every port arm. It also records what the engine
    holds for the rung's identity: the decode hook class, whether learned eviction is present, and the overlap switch
    of freetoken.layers.moe. These values come from the live objects, not from the flags.

The wrappers change no value and no control flow of the engine and have no code on the decode path: both run at startup,
before graph capture. Without REPLICATION_CACHE_DIR this module does nothing.
"""
import importlib.abc
import importlib.util
import json
import os
import sys

_DIR = os.environ.get('REPLICATION_CACHE_DIR')
_TARGET = 'freetoken.engine.engine'
_MOE = 'freetoken.layers.moe'


def _state(cache) -> dict:
    layer_layout = getattr(cache, 'layer_layout', None)
    return dict(decode_target=cache.decode_target, cpu_layer_ids=sorted(int(i) for i in cache.cpu_layer_ids),
                locked_hybrid=sorted(int(i) for i in cache.locked_hybrid),
                pool_sizes=[int(s) for s in cache.pool_sizes], cache_size=int(cache.cache_size),
                num_layers=int(cache.num_layers),
                distinct_layouts=None if layer_layout is None else len(set(layer_layout)))


def _identity(engine) -> dict:
    hooks = engine.decode_hooks
    moe = sys.modules.get(_MOE)
    return dict(hooks_type=None if hooks is None else type(hooks).__name__,
                learned=getattr(hooks, 'learned', None) is not None,
                overlap=None if moe is None else bool(moe._OVERLAP_RESIDENT))


def _wrap(module) -> None:
    own = dict(pid=os.getpid(), calls=[])

    def append(record: dict) -> None:
        if own['pid'] != os.getpid():   # a forked child starts its own record, as in the EXP587a hook
            own.update(pid=os.getpid(), calls=[])
        own['calls'].append(record)
        path = os.path.join(_DIR, f"cache_{own['pid']}.json")
        with open(path + '.tmp', 'w') as f:
            json.dump(dict(pid=own['pid'], calls=own['calls']), f)
        os.replace(path + '.tmp', path)

    check = module._refuse_hooks_bypass

    def _refuse_hooks_bypass(cache):
        append(dict(_state(cache), source='check'))
        return check(cache)

    module._refuse_hooks_bypass = _refuse_hooks_bypass
    init = module.Engine._init_offload_moe_cache

    def _init_offload_moe_cache(self, config):
        cache = init(self, config)
        append(dict(_state(cache), **_identity(self), source='init'))
        return cache

    module.Engine._init_offload_moe_cache = _init_offload_moe_cache


class _Finder(importlib.abc.MetaPathFinder):
    """As in the EXP587a hook: find the engine module with the remaining finders, wrap it after its code ran."""

    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.loader is not None:
            run = spec.loader.exec_module

            def exec_module(module):
                run(module)
                _wrap(module)

            spec.loader.exec_module = exec_module
        return spec


if _DIR:
    sys.meta_path.insert(0, _Finder())
