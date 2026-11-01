"""CPU tests of the shared learned eviction (EXP585): the bundle loader and its refusals, the eviction state
against the training recurrence, the readout, and the hooks (scores published before the prefetch bank, step
0 invalid, non-finite tokens to LRU, request resets, env switches). With the switch off, and with the r36
learned-eviction path, the hooks give the same buffers as the fork base 616acb4 on the same inputs; in shared
mode the prefetch bank reads the same inputs and gives the same outputs as with the switch off.

The GPU kernels (qgemv, the in-graph slot-map snapshot) are not run here; the GPU correctness window (P0-P2)
checks them. The hooks tests need the published config (COMMITMOE_PREDICTOR_DIR) and skip without it.
Run:  CUDA_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider tests/test_shared_evict_cpu.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from commitmoe_prefetch import bank_loader
from commitmoe_prefetch import freetoken as ft
from commitmoe_prefetch import shared_evict as se
from commitmoe_prefetch.spec import PredictorConfig

HERE = Path(__file__).resolve().parent
PUBLISHED = Path(os.environ.get('COMMITMOE_PREDICTOR_DIR', '/nonexistent'))
BUNDLE = Path(os.environ.get('EXP585_BUNDLE', '/nonexistent'))
BASE = '616acb4'
GROUP = 4
CPU = torch.device('cpu')
# Real model geometry (40 x 256, top 8, D 2048, the release table) with small trunk and head widths.
SMALL = dict(n_layer=40, n_exp=256, n_feat=4, layer_dim=2, hidden_dim=2048, hidden_proj=3, trunk_hidden=8,
             trunk_dim=8, evict_hidden=4, tid_emb_dim=32, token_table_rows=248070, hidden_layers=[6, 39],
             ema_beta=0.9, far=255, n_tok=256, horizons=[8, 16, 32, 64])
TINY = dict(SMALL, n_layer=3, n_exp=8, hidden_dim=16, token_table_rows=10, tid_emb_dim=2, hidden_layers=[0, 2])


def make_bundle(path: Path, f: dict = TINY, group: int = GROUP, seed: int = 0, release: str = 'a' * 64,
                meta_changes: dict | None = None, tensor_changes=None) -> Path:
    """A bundle of the exporter's format with random weights quantized by the bank's rule."""
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, (dtype, shape) in se.schema(f, group).items():
        if dtype == torch.int8:
            continue
        if name.endswith('_s'):
            w = torch.randn(shape[:2] + (shape[2] * group,), generator=g)[..., :se.schema(f, group)[name[:-2] + '_q'][1][2]]
            tensors[name[:-2] + '_q'], tensors[name] = bank_loader.quantize(w * 0.3, group)
        else:
            tensors[name] = torch.randn(shape, generator=g) * 0.3
    meta = dict(format=se.FORMAT, group=group, features=dict(f), release_sha256=release)
    meta.update(meta_changes or {})
    if tensor_changes:
        tensor_changes(tensors)
    torch.save(dict(meta=meta, tensors=tensors), path)
    return path


def load(path: Path, f: dict = TINY, release: str | None = 'a' * 64, group: int = GROUP, table=None,
         position_scale: int = 256, top_k: int = 2):
    table = torch.zeros(f['token_table_rows'], f['tid_emb_dim'], dtype=torch.bfloat16) if table is None else table
    return se.load_bundle(path, release, f['n_layer'], f['n_exp'], top_k, f['hidden_dim'], table, position_scale, group)


def scorer(path: Path, f: dict = TINY) -> se.SharedEvict:
    meta, t = load(path, f)
    return se.SharedEvict(meta, t, CPU, se.reference_linear)


# --- bundle loader --------------------------------------------------------------------------------

def test_load_accepts_a_bundle_of_the_schema(tmp_path):
    meta, t = load(make_bundle(tmp_path / 'b.pt'))
    assert meta['format'] == se.FORMAT and {k: (v.dtype, tuple(v.shape)) for k, v in t.items()} == se.schema(TINY, GROUP)


def _drop(name):
    return lambda t: t.pop(name)


def _strided(name):
    def change(t):
        t[name] = t[name].transpose(1, 2).contiguous().transpose(1, 2)
        assert not t[name].is_contiguous()
    return change


def _set(name, idx, value):
    def change(t):
        t[name][idx] = value
    return change


@pytest.mark.parametrize('kwargs,message', [
    (dict(meta_changes=dict(format='exp526')), 'expected exp585_shared_evict_v1'),
    (dict(meta_changes=dict(group=128)), 'INT8 group 128'),
    (dict(release='b' * 64), 'served release'),
    (dict(meta_changes=dict(features=dict(TINY, horizons=[8, 16, 32]))), 'horizons'),
    (dict(meta_changes=dict(features=dict(TINY, n_tok=512))), 'position scale 256'),
    (dict(meta_changes=dict(features=dict(TINY, hidden_layers=[0, 3]))), r'hidden layers \[3\]'),
    (dict(meta_changes=dict(features=dict(TINY, ema_beta=1.0))), 'ema_beta 1.0'),
    (dict(meta_changes=dict(features={k: v for k, v in TINY.items() if k != 'far'})), r"without \['far'\]"),
    (dict(tensor_changes=_drop('wb64_b')), r"schema at \['wb64_b'\]"),
    (dict(tensor_changes=lambda t: t.__setitem__('extra', torch.zeros(1))), r"schema at \['extra'\]"),
    (dict(tensor_changes=lambda t: t.__setitem__('mix_alpha', t['mix_alpha'].double())), 'mix_alpha'),
    (dict(tensor_changes=lambda t: t.__setitem__('wide', t['wide'][:, :3])), 'wide'),
    (dict(tensor_changes=_set('fc1_b', (0, 1), float('nan'))), 'fc1_b non-finite'),
    (dict(tensor_changes=_set('mix_alpha', (1, 2), float('inf'))), 'mix_alpha non-finite'),
    (dict(tensor_changes=_set('we_s', (0, 0, 0), 0.0)), 'we_s has a scale <= 0'),
    (dict(tensor_changes=_set('wb_q', (0, 0, 0), -128)), 'wb_q has code -128'),
    # Same shape and values, transposed strides: qgemv would read it in the wrong order.
    (dict(tensor_changes=_strided('wb64_q')), 'wb64_q is not C-contiguous'),
    (dict(tensor_changes=_strided('fc1_s')), 'fc1_s is not C-contiguous'),
])
def test_load_refuses(tmp_path, kwargs, message):
    release = kwargs.pop('release', 'a' * 64)
    path = make_bundle(tmp_path / 'b.pt', **kwargs)
    with pytest.raises(ValueError, match=message):
        se.load_bundle(path, release, TINY['n_layer'], TINY['n_exp'], 2, TINY['hidden_dim'],
                       torch.zeros(TINY['token_table_rows'], TINY['tid_emb_dim'], dtype=torch.bfloat16), 256, GROUP)


def test_load_refuses_another_model_and_table(tmp_path):
    path = make_bundle(tmp_path / 'b.pt')
    table = torch.zeros(TINY['token_table_rows'], TINY['tid_emb_dim'], dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='3 x 8 experts, the model has 4 x 8'):
        se.load_bundle(path, 'a' * 64, 4, 8, 2, TINY['hidden_dim'], table, 256, GROUP)
    with pytest.raises(ValueError, match='hidden width 16, the model has 32'):
        se.load_bundle(path, 'a' * 64, 3, 8, 2, 32, table, 256, GROUP)
    with pytest.raises(ValueError, match='served table torch.float32'):
        se.load_bundle(path, 'a' * 64, 3, 8, 2, TINY['hidden_dim'], table.float(), 256, GROUP)


@pytest.mark.skipif(not (BUNDLE.is_file() and (PUBLISHED / 'model.safetensors').is_file()),
                    reason='EXP585_BUNDLE or the published release absent')
def test_production_bundle_loads_against_the_release():
    from safetensors import safe_open
    cfg = PredictorConfig.model_validate(json.loads((PUBLISHED / 'config.json').read_text()))
    release = json.loads((PUBLISHED / 'manifest.json').read_text())['files']['model.safetensors']
    with safe_open(str(PUBLISHED / 'model.safetensors'), framework='pt') as fh:
        table = fh.get_tensor('table')
    meta, t = se.load_bundle(BUNDLE, release, cfg.base_model.num_layers, cfg.base_model.num_experts,
                             cfg.base_model.top_k, cfg.base_model.hidden_size, table, cfg.features.position_scale,
                             cfg.serving_reference.int8_group)
    assert t['we_q'].shape == (40, 256, 2048) and meta['sources']['head_sha256'].startswith('a1a3aafa')
    s = se.SharedEvict(meta, t, CPU, se.reference_linear)
    assert all(v.is_contiguous() for v in s.t.values())  # the layout qgemv reads


def test_age_table_is_the_reference_bytes(tmp_path):
    """The scorer's table is se.age_table's CPU bytes (copied, not recomputed on the device)."""
    s = scorer(make_bundle(tmp_path / 'b.pt'))
    ref = se.age_table(255)
    assert torch.equal(s.age_table, ref) and torch.equal(ref, torch.log2(1 + torch.arange(256).float()) / 8)
    assert ref[0] == 0 and ref[255] == 1


# --- state, features, readout ---------------------------------------------------------------------

def test_state_follows_the_training_recurrence(tmp_path):
    """300 tokens (age saturates at 255): ema and age equal exp565_shared_trunk.scan_features's loop,
    bit for bit, and step 0 changes nothing."""
    s = scorer(make_bundle(tmp_path / 'b.pt'))
    L, E, K = 3, 8, 2
    g = torch.Generator().manual_seed(1)
    s.advance(torch.randint(0, E, (L, K), generator=g), torch.tensor([0], dtype=torch.int32))
    assert not s.ema.any() and s.age.eq(255).all() and not s.routed.any()
    ema, age = torch.zeros(L, E), torch.full((L, E), 255, dtype=torch.int16)
    for t in range(300):
        routes = torch.randint(0, E - 2, (L, K), generator=g)  # experts 6 and 7 are never routed
        r = torch.zeros(L, E, dtype=torch.bool).scatter_(1, routes, True)
        ema = ema * 0.9 + r.float() * (1 - 0.9)
        age = torch.where(r, torch.zeros_like(age), (age + 1).clamp(max=255))
        s.advance(routes, torch.tensor([t + 1], dtype=torch.int32))
        assert torch.equal(s.ema, ema) and torch.equal(s.age, age.long()) and torch.equal(s.routed, r.float())
    agef = torch.log2(1 + age.float()) / 8
    own = s.own(torch.ones(L, E))
    assert torch.equal(own[..., 2], agef) and own[..., 2][:, 6:].eq(1).all()
    assert torch.equal(own[..., 0], r.float()) and torch.equal(own[..., 1], ema) and own[..., 3].eq(1).all()
    s.reset()
    assert not s.ema.any() and s.age.eq(255).all() and not s.routed.any()


def test_skip_adds_exact_products_in_feature_order():
    g = torch.Generator().manual_seed(2)
    own = torch.randn(3, 8, 4, generator=g).bfloat16().float()
    w3, w1 = torch.randn(3, 4, 3, generator=g).bfloat16().float(), torch.randn(3, 4, generator=g).bfloat16().float()
    want3 = (((own[..., 0:1] * w3[:, None, 0] + own[..., 1:2] * w3[:, None, 1]) + own[..., 2:3] * w3[:, None, 2])
             + own[..., 3:4] * w3[:, None, 3])
    assert torch.equal(se.skip(own, w3), want3.bfloat16().float())
    want1 = ((own[..., 0] * w1[:, None, 0] + own[..., 1] * w1[:, None, 1]) + own[..., 2] * w1[:, None, 2]) \
        + own[..., 3] * w1[:, None, 3]
    assert torch.equal(se.skip(own, w1), want1.bfloat16().float())


def test_reference_linear_is_the_qgemv_contract():
    g = torch.Generator().manual_seed(3)
    w, x, b = torch.randn(2, 5, 9, generator=g), torch.randn(2, 9, generator=g), torch.randn(2, 5, generator=g)
    codes, scale = bank_loader.quantize(w, GROUP)
    dq = codes.float() * scale.float().repeat_interleave(GROUP, -1)[..., :9]
    want = (torch.einsum('hk,hnk->hn', x.bfloat16().float(), dq) + b).bfloat16()
    assert torch.equal(se.reference_linear(x, codes, scale, b, GROUP), want)


def test_heads_are_cumulative_and_the_mix_is_the_mixture_logit(tmp_path):
    s = scorer(make_bundle(tmp_path / 'b.pt'))
    g = torch.Generator().manual_seed(4)
    s.advance(torch.randint(0, 8, (3, 2), generator=g), torch.tensor([1], dtype=torch.int32))
    table = torch.randn(10, 2, generator=g).bfloat16()
    args = (s.own(torch.randint(0, 2, (3, 8), generator=g).float()), torch.randn(3, 16, generator=g).half(),
            torch.tensor([3]), torch.tensor([7]), torch.tensor([[2 / 256]]), table)
    u = s.heads(*args)
    assert u.shape == (3, 8, 4) and u.dtype == torch.float32 and bool((u.diff(dim=-1) >= 0).all())
    z = s.scores(*args)
    p = (torch.softmax(s.t['mix_alpha'], -1).unsqueeze(1) * torch.sigmoid(u.double())).sum(-1)
    assert z.shape == (3, 8) and torch.allclose(z.double(), torch.log(p / (1 - p)), atol=1e-5)
    assert torch.equal(z, s.scores(*args))  # deterministic


def test_mix_is_finite_and_ordered_at_extreme_logits(tmp_path):
    """Signed and extreme cumulative logits give finite, ordered keep scores; equal inputs tie exactly."""
    s = scorer(make_bundle(tmp_path / 'b.pt'))
    u = torch.tensor([-200., -50., -5., 0., 5., 50., 200., 200.]).view(1, 8, 1).expand(3, 8, 4).contiguous()
    z = s.mix(u)
    assert bool(torch.isfinite(z).all()) and bool((z[:, 1:] >= z[:, :-1]).all())
    assert torch.equal(z[:, 6], z[:, 7]) and bool((z[:, :3] < 0).all()) and bool((z[:, 4:] > 0).all())


# --- hooks ----------------------------------------------------------------------------------------

@pytest.fixture
def cfg():
    if not (PUBLISHED / 'config.json').is_file():
        pytest.skip('published folder absent')
    return PredictorConfig.model_validate(json.loads((PUBLISHED / 'config.json').read_text()))


class StubBank:
    """Records its inputs and calls; returns fixed probabilities; holds a BF16 table like the bank."""
    log: list = []

    def __init__(self, cfg, state, device):
        self.nbytes, self.raw = 0, None
        self.out = torch.linspace(0, 1, len(cfg.target_layers) * cfg.base_model.num_experts).view(
            len(cfg.target_layers), -1)
        g = torch.Generator().manual_seed(5)
        self.buffers = dict(table=torch.randn(cfg.features.token_table_rows, cfg.features.token_embedding_dim,
                                              generator=g).bfloat16())

    def __call__(self, raw):
        StubBank.log.append('bank')
        self.raw = {k: v.clone() for k, v in raw.items()}
        return self.out


class StubEvent:
    def record(self, *args):
        StubBank.log.append('scores_ready')


def stub_device(monkeypatch, module):
    monkeypatch.setattr(module, 'Int8Bank', StubBank)
    monkeypatch.setattr(torch.cuda, 'Stream', lambda **kwargs: None)  # no device on the CPU runner
    monkeypatch.setattr(torch.cuda, 'Event', StubEvent)
    monkeypatch.setattr(module, 'qlinear', se.reference_linear, raising=False)


def make_hooks(module, cfg, release='a' * 64, **kwargs):
    hooks = module.PrefetchHooks(cfg, {}, demand_stripes=16, release_sha256=release, **kwargs)
    hooks.reserve(SimpleNamespace(device=CPU))
    return hooks


@pytest.fixture
def shared(cfg, monkeypatch, tmp_path):
    stub_device(monkeypatch, ft)
    path = make_bundle(tmp_path / 's.pt', SMALL, group=cfg.serving_reference.int8_group)
    hooks = make_hooks(ft, cfg, shared=(path, 1))
    hooks.cache = SimpleNamespace(slot_for_id=torch.full((40, 256), -1, dtype=torch.int32))
    StubBank.log.clear()
    return hooks


def feed(hooks, step: int, seed: int) -> None:
    """End-of-token inputs of decode step ``step - 1`` as after_decode leaves them, then step ``step``."""
    g = torch.Generator().manual_seed(seed)
    hooks.score_inputs['routes'].copy_(torch.stack([torch.randperm(256, generator=g)[:8] for _ in range(40)]))
    hooks.score_inputs['h'].copy_(torch.randn(40, 2048, generator=g).half())
    hooks.score_inputs['rl'].copy_(torch.randn(40, 256, generator=g).half())
    hooks.score_inputs['prev'].copy_(torch.randint(0, 256, (40, 8), generator=g))
    hooks.raw['tid'].fill_(int(torch.randint(0, 1000, (1,), generator=g)))
    hooks.raw['upcoming_tid'].fill_(int(torch.randint(0, 1000, (1,), generator=g)))
    hooks.raw['pos'].fill_(step / 256)
    hooks.step.fill_(step)
    if getattr(hooks, 'shared', None) is not None:
        hooks.resident.copy_((torch.rand(40, 256, generator=g) < 0.3).float())


def test_shared_mode_sizes_and_order(shared):
    assert shared.learned.keep.shape == (40, 256) and shared.learned.start_layer == 1
    assert shared.learned.target_of_layer.tolist() == list(range(40)) and shared.evict_cfg is None
    feed(shared, 3, 0)
    shared._score()
    # a2 order: the keep scores are published before the prefetch bank runs
    assert StubBank.log == ['scores_ready', 'bank']
    assert int(shared.learned.valid) == 1 and bool(torch.isfinite(shared.learned.keep).all())


def test_shared_step_zero_is_invalid_and_holds_the_state(shared):
    feed(shared, 0, 0)
    shared._score()
    assert int(shared.learned.valid) == 0 and int(shared.nonfinite_tokens) == 0
    assert not shared.shared.ema.any() and shared.shared.age.eq(255).all()
    feed(shared, 1, 0)
    shared._score()
    assert int(shared.learned.valid) == 1 and shared.shared.ema.gt(0).sum() == 40 * 8


def test_shared_nonfinite_token_falls_back_to_lru(shared):
    feed(shared, 2, 0)
    shared.score_inputs['h'][6, 5] = float('nan')
    shared._score()
    assert int(shared.learned.valid) == 0 and int(shared.nonfinite_tokens) == 1
    feed(shared, 3, 1)
    shared._score()
    assert int(shared.learned.valid) == 1 and int(shared.nonfinite_tokens) == 1


def test_shared_request_boundary_resets_everything(shared):
    for step in range(1, 4):
        feed(shared, step, step)
        shared._score()
    assert int(shared.learned.valid) == 1 and shared.shared.ema.any()
    shared.begin_request(9)
    assert int(shared.learned.valid) == 0 and not shared.learned.keep.any() and not shared.resident.any()
    assert not shared.shared.ema.any() and shared.shared.age.eq(255).all() and not shared.shared.routed.any()
    assert int(shared.step) == 0
    shared.score_inputs['routes'].copy_(torch.arange(8).expand(40, 8))  # stale routes must not enter
    shared._score()  # the new request's step 0
    assert int(shared.learned.valid) == 0 and not shared.shared.ema.any()


def test_shared_first_scored_token_matches_a_fresh_scorer(shared):
    """After a request with three tokens and a reset, the first scored token of the next request gives the
    keep scores of a scorer that never saw the first request."""
    for step in range(1, 4):
        feed(shared, step, 10 + step)
        shared._score()
    shared.begin_request(9)
    feed(shared, 1, 20)
    shared._score()
    s = shared.shared
    fresh = se.SharedEvict(*load(shared.shared_cfg[0], SMALL, group=s.group, table=shared.bank.buffers['table'],
                                 top_k=8), CPU, se.reference_linear)
    fresh.advance(shared.score_inputs['routes'], torch.tensor([1], dtype=torch.int32))
    want = fresh.scores(fresh.own(shared.resident), shared.score_inputs['h'], shared.raw['tid'],
                        shared.raw['upcoming_tid'], shared.raw['pos'][:1], shared.bank.buffers['table'])
    assert torch.equal(shared.learned.keep, want)


def test_prefetch_inputs_and_outputs_unchanged_in_shared_mode(cfg, monkeypatch, tmp_path):
    """Same captured inputs, switch off vs shared mode: the bank reads the same raw, gives the same probs,
    and the prefetch EMA prior moves the same way."""
    stub_device(monkeypatch, ft)
    off = make_hooks(ft, cfg)
    on = make_hooks(ft, cfg, shared=(make_bundle(tmp_path / 's.pt', SMALL, group=cfg.serving_reference.int8_group), 1))
    on.cache = SimpleNamespace(slot_for_id=torch.full((40, 256), -1, dtype=torch.int32))
    for step in range(3):
        for h in (off, on):
            feed(h, step, 30 + step)
            h._score()
        for k in off.bank.raw:
            assert torch.equal(off.bank.raw[k], on.bank.raw[k]), k
        assert torch.equal(off.probs, on.probs) and torch.equal(off.raw['prior_before'], on.raw['prior_before'])


def load_base(monkeypatch):
    """freetoken.py as committed at the fork base, loaded as a sibling module of the package."""
    root = HERE.parents[1]
    src = subprocess.run(['git', '-C', str(root), 'show', f'{BASE}:commitmoe_prefetch/commitmoe_prefetch/freetoken.py'],
                         check=True, capture_output=True, text=True).stdout
    spec = importlib.util.spec_from_loader('commitmoe_prefetch.freetoken_base', loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'commitmoe_prefetch'
    monkeypatch.setitem(sys.modules, spec.name, mod)
    exec(compile(src, f'{BASE}:freetoken.py', 'exec'), mod.__dict__)
    return mod


def state(hooks) -> dict[str, torch.Tensor]:
    out = {f'live.{k}': v for k, v in hooks.live.items()} | {f'in.{k}': v for k, v in hooks.score_inputs.items()}
    out |= {f'raw.{k}': v for k, v in hooks.raw.items()}
    out |= dict(probs=hooks.probs, step=hooks.step, gen=hooks.demand.gen, unscored=hooks.unscored_ids)
    if hooks.learned is not None:
        out |= dict(keep=hooks.learned.keep, valid=hooks.learned.valid, nonfinite=hooks.nonfinite_tokens)
    return {k: v.clone() for k, v in out.items()}


@pytest.mark.parametrize('mode', ['off', 'r36'])
def test_control_paths_equal_the_fork_base(cfg, monkeypatch, tmp_path, mode):
    """The switch off and the r36 learned-eviction path give the fork base's buffers on the same calls."""
    base = load_base(monkeypatch)
    stub_device(monkeypatch, ft)
    stub_device(monkeypatch, base)
    from commitmoe_prefetch import evict
    monkeypatch.setattr(evict, 'qlinear', se.reference_linear)
    kwargs = {}
    if mode == 'r36':
        torch.manual_seed(0)
        layers = [0, 1, 8, 39]
        torch.save(dict(version=2, layers=layers, source_layers=[6, 6, 8, 39], horizon_tokens=8,
                        release_sha256='a' * 64, weight=torch.randn(4, 256, cfg.architecture.up_dim).bfloat16(),
                        bias=torch.randn(4, 256).bfloat16()), tmp_path / 'r.pt')
        kwargs = dict(evict=(tmp_path / 'r.pt', 1))
    hooks = [make_hooks(m, cfg, **kwargs) for m in (base, ft)]
    H, up = len(cfg.target_layers), cfg.architecture.up_dim
    for h in hooks:  # the stub bank has no trunk/head split; the r36 path calls both
        h.bank.trunk = lambda raw, b=h.bank: (b(raw)[:, :1] + torch.linspace(-1, 1, H * up).view(H, up)).bfloat16()
        h.bank.head = lambda g, b=h.bank: b.out
    for step in range(4):
        for h in hooks:
            if step == 0:
                h.begin_request(5)
            h.before_decode(step)
            feed(h, step, 40 + step)
            h._score()
            h.after_decode(step, torch.tensor([11 + step]), torch.tensor([12 + step]))
        a, b = state(hooks[0]), state(hooks[1])
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), [k for k in a if not torch.equal(a[k], b[k])]


def test_env_switches():
    assert ft.shared_evict_from_env({}) is None
    assert ft.shared_evict_from_env({ft.SHARED_EVICT_ENV: '/b.pt', ft.EVICT_START_ENV: '1'}) == (Path('/b.pt'), 1)
    with pytest.raises(ValueError, match='two eviction rules'):
        ft.shared_evict_from_env({ft.SHARED_EVICT_ENV: '/b.pt', ft.EVICT_ENV: '/r.pt', ft.EVICT_START_ENV: '1'})
    with pytest.raises(ValueError, match='must give the first learned layer'):
        ft.shared_evict_from_env({ft.SHARED_EVICT_ENV: '/b.pt'})
    assert ft.evict_from_env({ft.SHARED_EVICT_ENV: '/b.pt', ft.EVICT_START_ENV: '1'}) is None


def test_hooks_refuse_two_rules_and_a_bad_start(cfg):
    with pytest.raises(ValueError, match='two eviction rules'):
        ft.PrefetchHooks(cfg, {}, 16, evict=(Path('/r.pt'), 1), shared=(Path('/b.pt'), 1))
    with pytest.raises(ValueError, match='is not a layer 0..39'):
        ft.PrefetchHooks(cfg, {}, 16, shared=(Path('/b.pt'), 40))
    ft.PrefetchHooks(cfg, {}, 16, timing=Path('/t.pt'), shared=(Path('/b.pt'), 1))  # timing works with either rule


def test_hooks_refuse_a_bundle_of_another_release(cfg, monkeypatch, tmp_path):
    stub_device(monkeypatch, ft)
    path = make_bundle(tmp_path / 's.pt', SMALL, group=cfg.serving_reference.int8_group, release='c' * 64)
    with pytest.raises(ValueError, match='served release'):
        make_hooks(ft, cfg, shared=(path, 1))


def test_bind_refuses_a_slot_map_of_another_shape(shared):
    """bind() with a slot map that is not (layers, experts); the stage setup before the check has no banks."""
    cache = SimpleNamespace(num_layers=40, num_experts=256, slot_for_id=torch.zeros(39, 256, dtype=torch.int32),
                            banks=[], _copy_src_ptrs=[torch.zeros(1, dtype=torch.int64)], _copy_feat_bytes=torch.zeros(1))
    shared.demand.check_cache = lambda c: None
    with pytest.raises(ValueError, match='slot map'):
        shared.bind(None, cache)
