"""CPU tests of the fast R584 scorer (EXP586): shared_evict_fast.FastSharedEvict on its torch path against
shared_evict.SharedEvict, the F8 split planning, the wb/wb64 concatenation and memory count, and the hooks in
fast mode (switch parsing and refusals, F1 order: scores-ready before the raw copies, the same buffers as the
shared mode on the same inputs, no read of a copied raw field). With the fast switch absent, the shared mode
gives the buffers of the EXP585 commit eba298d.

The Triton kernels are not run here: libdevice calls do not run in the Triton CPU interpreter. The GPU tests
(tests/test_gpu_exp586_fast.py) compare them with the SharedEvict kernels.
Run:  CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider tests/test_shared_evict_fast_cpu.py
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from commitmoe_prefetch import freetoken as ft
from commitmoe_prefetch import shared_evict as se
from commitmoe_prefetch import shared_evict_fast as sf
from test_shared_evict_cpu import (CPU, HERE, SMALL, TINY, StubBank, cfg, feed, load, make_bundle,  # noqa: F401
                                   make_hooks, state, stub_device)

SHARED_BASE = 'eba298d'  # the EXP585 commit of freetoken.py's shared mode


def pair(path: Path, f: dict, f8: str, top_k: int = 2) -> tuple[se.SharedEvict, sf.FastSharedEvict]:
    meta, t = load(path, f, top_k=top_k)
    return se.SharedEvict(meta, dict(t), CPU, se.reference_linear), sf.FastSharedEvict(meta, dict(t), CPU, f8, kernels=False)


def token(f: dict, seed: int, top_k: int) -> dict[str, torch.Tensor]:
    """Inputs of one scored token; row 0 repeats one routed id to check that a repeat counts once."""
    g = torch.Generator().manual_seed(seed)
    L, E, D = f['n_layer'], f['n_exp'], f['hidden_dim']
    routes = torch.stack([torch.randperm(E, generator=g)[:top_k] for _ in range(L)])
    routes[0, -1] = routes[0, 0]
    return dict(routes=routes, resident=(torch.rand(L, E, generator=g) < 0.4).float(),
                h=torch.randn(L, D, generator=g).half(),
                tid=torch.randint(0, f['token_table_rows'], (1,), generator=g),
                upcoming=torch.randint(0, f['token_table_rows'], (1,), generator=g),
                pos=torch.rand(1, 1, generator=g))


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal values, NaN equal to NaN (keep holds NaN on a non-finite token in both paths)."""
    return a.shape == b.shape and a.dtype == b.dtype and bool(((a == b) | (a.isnan() & b.isnan())).all())


def run_both(plain, fast, f, inp, step, table):
    """One token on both scorers; returns (keep, valid) of each and the fast non-finite count."""
    s = torch.tensor([step], dtype=torch.int32)
    plain.advance(inp['routes'], s)
    want = plain.scores(plain.own(inp['resident']), inp['h'], inp['tid'], inp['upcoming'], inp['pos'], table)
    valid_want = int(step > 0 and bool(torch.isfinite(want).all()))
    keep = torch.zeros(f['n_layer'], f['n_exp'])
    valid, nonfinite = torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int64)
    fast.score(inp['routes'], s, inp['resident'], inp['h'], inp['tid'], inp['upcoming'], inp['pos'], table,
               keep, valid, nonfinite)
    return want, valid_want, keep, int(valid), int(nonfinite)


# --- the scorer ---------------------------------------------------------------------------------

@pytest.mark.parametrize('f, top_k, f8', [(TINY, 2, 'blockn'), (SMALL, 8, 'blockn'), (SMALL, 8, 'splitk')])
def test_fast_equals_shared_over_tokens_and_a_reset(tmp_path, f, top_k, f8):
    """(TINY's fc1 has 5 K groups, too few for the 4-way split; the constructor refuses it.)"""
    """Steps 0..3, a non-finite token, a reset, then step 0 and 1 of a new request: equal keep, valid,
    non-finite count and state after every token."""
    path = make_bundle(tmp_path / 's.pt', f)
    plain, fast = pair(path, f, f8, top_k)
    table = torch.randn(f['token_table_rows'], f['tid_emb_dim']).bfloat16()
    nonfinite = 0
    for i, step in enumerate([0, 1, 2, 3, 4, 'reset', 0, 1]):
        if step == 'reset':
            plain.reset()
            fast.reset()
            continue
        inp = token(f, 50 + i, top_k)
        if i == 4:
            inp['h'][f['hidden_layers'][0], 1] = float('nan')
        want, valid_want, keep, valid, count = run_both(plain, fast, f, inp, step, table)
        nonfinite += count
        assert same(keep, want), i
        assert valid == valid_want, i
        for name in ('ema', 'age', 'routed'):
            assert torch.equal(getattr(fast, name), getattr(plain, name)), (i, name)
    assert nonfinite == 1


def test_step_zero_after_state_holds_the_state(tmp_path):
    path = make_bundle(tmp_path / 's.pt', TINY)
    _, fast = pair(path, TINY, 'blockn')
    inp = token(TINY, 1, 2)
    fast.advance_and_own(inp['routes'], torch.tensor([3], dtype=torch.int32), inp['resident'])
    before = [t.clone() for t in (fast.ema, fast.age, fast.routed)]
    x = fast.advance_and_own(token(TINY, 2, 2)['routes'], torch.tensor([0], dtype=torch.int32), inp['resident'])
    assert all(torch.equal(a, b) for a, b in zip(before, (fast.ema, fast.age, fast.routed)))
    E = TINY['n_exp']
    assert torch.equal(x[:, :E].float(), fast.routed) and torch.equal(x[:, 3 * E:].float(), inp['resident'])


def test_age_saturates_at_far(tmp_path):
    path = make_bundle(tmp_path / 's.pt', TINY)
    _, fast = pair(path, TINY, 'blockn')
    routes = torch.zeros(TINY['n_layer'], 2, dtype=torch.int64)  # only expert 0 is ever routed
    for _ in range(3):
        fast.advance_and_own(routes, torch.tensor([1], dtype=torch.int32), torch.zeros(3, 8))
    assert fast.age[:, 0].eq(0).all() and fast.age[:, 1:].eq(TINY['far']).all()


def test_readout_writes_u_as_heads(tmp_path):
    """The torch readout with u: cumulative logits non-decreasing in the horizon, z of the mix."""
    path = make_bundle(tmp_path / 's.pt', SMALL)
    plain, fast = pair(path, SMALL, 'blockn', 8)
    L, E = SMALL['n_layer'], SMALL['n_exp']
    x = torch.randn(L, 4 * E).bfloat16()
    wbo = (torch.randn(L, 4 * E) * 30).bfloat16()  # large values cross the softplus threshold 20
    keep, u = torch.zeros(L, E), torch.zeros(L, E, 4)
    fast.readout(wbo, x, keep, u)
    assert (u.diff(dim=-1) >= 0).all()
    assert torch.equal(keep, plain.mix(u)) and int(fast.bad) == 0


def test_wb_and_wb64_are_concatenated_and_counted(tmp_path):
    path = make_bundle(tmp_path / 's.pt', SMALL)
    meta, t = load(path, SMALL, top_k=8)
    for f8 in sf.F8:
        fast = sf.FastSharedEvict(meta, dict(t), CPU, f8, kernels=False)
        plain = se.SharedEvict(meta, dict(t), CPU, se.reference_linear)
        assert not {k for k in fast.t if k.startswith(('wb_', 'wb64_'))}
        for part in ('_q', '_s', '_b'):
            assert torch.equal(fast.t['wbx' + part], torch.cat([t['wb' + part], t['wb64' + part]], 1))
        extra = [fast.ell2, fast.x, fast.bad] + list(fast.partial.values())
        assert fast.nbytes == plain.nbytes + sum(b.numel() * b.element_size() for b in extra)
        assert set(fast.partial) == (set(sf.F8_SPLITS) if f8 == 'splitk' else set())


def test_parent_methods_and_bad_variants_are_refused(tmp_path):
    path = make_bundle(tmp_path / 's.pt', TINY)
    meta, t = load(path, TINY)
    with pytest.raises(ValueError, match='F8 variant'):
        sf.FastSharedEvict(meta, dict(t), CPU, 'fast', kernels=False)
    with pytest.raises(ValueError, match='power of two'):
        sf.FastSharedEvict(dict(meta, features=dict(meta['features'], n_exp=6)), dict(t), CPU, 'blockn')
    fast = sf.FastSharedEvict(meta, dict(t), CPU, 'blockn', kernels=False)
    for name in ('advance', 'own', 'heads', 'mix', 'scores'):
        with pytest.raises(NotImplementedError):
            getattr(fast, name)()


def test_split_groups():
    assert sf.split_groups(45, 4) == 12 and sf.split_groups(16, 8) == 2 and sf.split_groups(10, 4) == 3
    for groups, splits in ((45, 1), (4, 8), (9, 4)):
        with pytest.raises(ValueError, match='cannot be split'):
            sf.split_groups(groups, splits)
    # the declared production splits: fc1 has 45 K groups (K 5697), h_proj 16 (K 2048), group 128
    assert sf.split_groups(45, sf.F8_SPLITS['fc1']) == 12 and sf.split_groups(16, sf.F8_SPLITS['h_proj']) == 2


# --- the hooks ----------------------------------------------------------------------------------

def hooks_pair(cfg, monkeypatch, tmp_path, f8='blockn'):
    stub_device(monkeypatch, ft)
    path = make_bundle(tmp_path / 's.pt', SMALL, group=cfg.serving_reference.int8_group)
    out = []
    for fast in (None, f8):
        h = make_hooks(ft, cfg, shared=(path, 1), shared_fast=fast)
        h.cache = SimpleNamespace(slot_for_id=torch.full((40, 256), -1, dtype=torch.int32))
        out.append(h)
    StubBank.log.clear()
    return out


def test_fast_env_switch():
    on = {ft.SHARED_EVICT_ENV: '/b.pt', ft.EVICT_START_ENV: '1'}
    assert ft.shared_fast_from_env({}) is None and ft.shared_fast_from_env(on) is None
    assert ft.shared_fast_from_env(dict(on, **{ft.SHARED_EVICT_FAST_ENV: 'splitk'})) == 'splitk'
    with pytest.raises(ValueError, match='needs FREETOKEN_SHARED_EVICT'):
        ft.shared_fast_from_env({ft.SHARED_EVICT_FAST_ENV: 'blockn'})
    with pytest.raises(ValueError, match='is not one of'):
        ft.shared_fast_from_env(dict(on, **{ft.SHARED_EVICT_FAST_ENV: '1'}))


def test_hooks_refuse_fast_without_shared_and_a_bad_variant(cfg):
    with pytest.raises(ValueError, match='needs FREETOKEN_SHARED_EVICT'):
        ft.PrefetchHooks(cfg, {}, 16, shared_fast='blockn')
    with pytest.raises(ValueError, match='is not one of'):
        ft.PrefetchHooks(cfg, {}, 16, shared=(Path('/b.pt'), 1), shared_fast='1')


def test_fast_mode_publishes_before_the_raw_copies(cfg, monkeypatch, tmp_path):
    plain, fast = hooks_pair(cfg, monkeypatch, tmp_path)
    assert isinstance(fast.shared, sf.FastSharedEvict) and type(plain.shared) is se.SharedEvict
    for h in (plain, fast):
        copy = h._copy_raw
        h._copy_raw = lambda copy=copy: (StubBank.log.append('copies'), copy())
    for h, want in ((plain, ['copies', 'scores_ready', 'bank']), (fast, ['scores_ready', 'copies', 'bank'])):
        StubBank.log.clear()
        feed(h, 3, 0)
        h._score()
        assert StubBank.log == want


def test_fast_score_reads_no_copied_raw_field(cfg, monkeypatch, tmp_path):
    """Two fast hooks on the same inputs, one with garbage in every copied raw field: equal keep and valid."""
    stub_device(monkeypatch, ft)
    path = make_bundle(tmp_path / 's.pt', SMALL, group=cfg.serving_reference.int8_group)
    clean, dirty = (make_hooks(ft, cfg, shared=(path, 1), shared_fast='blockn') for _ in range(2))
    for step in (1, 2, 3):
        for h in (clean, dirty):
            feed(h, step, 70 + step)
        for name in ('h', 'h_prev', 'win', 'prevwin', 'curr', 'rl'):
            dirty.raw[name].fill_(7)
        for h in (clean, dirty):
            h._score_shared()
        assert torch.equal(clean.learned.keep, dirty.learned.keep) and int(dirty.learned.valid) == 1
        assert int(clean.learned.valid) == int(dirty.learned.valid)


def test_fast_hooks_equal_shared_hooks(cfg, monkeypatch, tmp_path):
    """Same calls on shared and fast hooks (two requests, a non-finite token): every buffer equal."""
    plain, fast = hooks_pair(cfg, monkeypatch, tmp_path)
    for uid, steps in ((5, range(4)), (6, range(3))):
        for step in steps:
            for h in (plain, fast):
                if step == 0:
                    h.begin_request(uid)
                h.before_decode(step)
                feed(h, step, 60 + 10 * uid + step)
                if (uid, step) == (5, 2):
                    h.score_inputs['h'][6, 3] = float('inf')
                h._score()
                h.after_decode(step, torch.tensor([11 + step]), torch.tensor([12 + step]))
            a, b = state(plain), state(fast)
            assert a.keys() == b.keys()
            assert all(same(a[k], b[k]) for k in a), (uid, step, [k for k in a if not same(a[k], b[k])])
            for name in ('ema', 'age', 'routed'):
                assert torch.equal(getattr(plain.shared, name), getattr(fast.shared, name)), name
            for k in plain.bank.raw:
                assert torch.equal(plain.bank.raw[k], fast.bank.raw[k]), k
            if (uid, step) == (5, 2):  # begin_request clears the count, so check it here
                assert int(fast.nonfinite_tokens) == 1 and int(fast.learned.valid) == 0


def load_shared_base(monkeypatch):
    """freetoken.py as committed at SHARED_BASE, loaded as a sibling module of the package."""
    root = HERE.parents[1]
    src = subprocess.run(['git', '-C', str(root), 'show',
                          f'{SHARED_BASE}:commitmoe_prefetch/commitmoe_prefetch/freetoken.py'],
                         check=True, capture_output=True, text=True).stdout
    spec = importlib.util.spec_from_loader('commitmoe_prefetch.freetoken_shared_base', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'commitmoe_prefetch'
    monkeypatch.setitem(sys.modules, spec.name, mod)
    exec(compile(src, f'{SHARED_BASE}:freetoken.py', 'exec'), mod.__dict__)
    return mod


def test_shared_mode_without_fast_equals_the_exp585_commit(cfg, monkeypatch, tmp_path):
    base = load_shared_base(monkeypatch)
    stub_device(monkeypatch, ft)
    stub_device(monkeypatch, base)
    path = make_bundle(tmp_path / 's.pt', SMALL, group=cfg.serving_reference.int8_group)
    hooks = [make_hooks(m, cfg, shared=(path, 1)) for m in (base, ft)]
    for h in hooks:
        h.cache = SimpleNamespace(slot_for_id=torch.full((40, 256), -1, dtype=torch.int32))
    for step in range(4):
        for h in hooks:
            if step == 0:
                h.begin_request(5)
            h.before_decode(step)
            feed(h, step, 80 + step)
            h._score()
            h.after_decode(step, torch.tensor([11 + step]), torch.tensor([12 + step]))
        a, b = state(hooks[0]), state(hooks[1])
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), [k for k in a if not torch.equal(a[k], b[k])]
