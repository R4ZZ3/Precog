"""CPU tests of commitmoe_prefetch: release-file identity, manifest checks, settings checks,
feature layout of the in-graph scoring, and the end-of-token updates.

Needs the FreeToken fork installed. The tests that read the published predictor need
COMMITMOE_PREDICTOR_DIR (see PUBLISHED below) and skip without it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from commitmoe_prefetch import bank_loader, hub
from commitmoe_prefetch import freetoken as ft
from commitmoe_prefetch.spec import PredictorConfig

HERE = Path(__file__).resolve().parent
# A local copy of the published predictor (huggingface-cli download RASMUS/qwen3.5-35b-a3b-fp8-expert-predictor
# --local-dir DIR). The tests that need it skip when COMMITMOE_PREDICTOR_DIR is unset.
PUBLISHED = Path(os.environ.get('COMMITMOE_PREDICTOR_DIR', '/nonexistent'))
needs_published = pytest.mark.skipif(not (PUBLISHED / 'config.json').is_file(), reason='published folder absent')


@needs_published
def test_package_copies_equal_the_release_code():
    """The Hub repo ships the reference spec, scorer and loader; the package must run the same code."""
    pkg = HERE.parent / 'commitmoe_prefetch'
    for name in ('spec.py', 'reference_scorer.py'):
        assert (pkg / name).read_text() == (PUBLISHED / name).read_text(), name
    # bank_loader differs only in the relative import of spec
    want = (PUBLISHED / 'bank_loader.py').read_text().replace('from spec import', 'from .spec import')
    assert (pkg / 'bank_loader.py').read_text() == want


def _folder(tmp_path, gate=True, tamper=False):
    for name, data in (('config.json', b'{}'), ('model.safetensors', b'weights')):
        (tmp_path / name).write_bytes(data)
    files = {name: hub._sha256(tmp_path / name) for name in hub.DATA_FILES}
    if tamper:
        (tmp_path / 'model.safetensors').write_bytes(b'other weights')
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(files=files, gate=dict(passed=gate))))
    return tmp_path


def test_verify_accepts_a_matching_folder(tmp_path):
    hub.verify(_folder(tmp_path))


@pytest.mark.parametrize('change,message', [(dict(gate=False), 'passed release gate'),
                                            (dict(tamper=True), 'differs from the manifest')])
def test_verify_refuses(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        hub.verify(_folder(tmp_path, **change))


def test_verify_refuses_a_folder_without_manifest(tmp_path):
    with pytest.raises(ValueError, match='no manifest.json'):
        hub.verify(tmp_path)


def test_resolve_refuses_a_file(tmp_path):
    path = tmp_path / 'x.safetensors'
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='got a file'):
        hub.resolve(str(path))


@pytest.fixture
def cfg():
    if not (PUBLISHED / 'config.json').is_file():
        pytest.skip('published folder absent')
    return PredictorConfig.model_validate(json.loads((PUBLISHED / 'config.json').read_text()))


@needs_published
def test_published_folder_passes_verify():
    hub.verify(PUBLISHED)


def _changed(cfg, **serving):
    data = cfg.model_dump()
    data['serving_reference'].update(serving)
    return PredictorConfig.model_validate(data)


def test_refuses_wrong_cap_count_and_zero_stripes(cfg):
    with pytest.raises(ValueError) as err:
        ft.PrefetchHooks(_changed(cfg, section_caps=(2, 2)), {}, demand_stripes=0)
    assert 'section caps' in str(err.value) and 'at least 1' in str(err.value)


class _StubBank:
    """Records the raw inputs and returns fixed probabilities."""

    def __init__(self, cfg, state, device):
        self.nbytes, self.raw = 0, None
        self.out = torch.full((len(cfg.target_layers), cfg.base_model.num_experts), 0.5)

    def __call__(self, raw):
        self.raw = {k: v.clone() for k, v in raw.items()}
        return self.out


@pytest.fixture
def hooks(cfg, monkeypatch):
    monkeypatch.setattr(ft, 'Int8Bank', _StubBank)
    monkeypatch.setattr(torch.cuda, 'Stream', lambda **kwargs: None)  # no device on the CPU runner
    hooks = ft.PrefetchHooks(cfg, {}, demand_stripes=16)
    hooks.reserve(SimpleNamespace(device=torch.device('cpu')))
    return hooks


def test_reserve_sizes(hooks, cfg):
    assert hooks.sections == 7 and hooks.first == 6 and hooks.heads == 34
    assert hooks.demand.ids.numel() == 8 and hooks.records.shape == (7, 4)
    # every target L reads logits of L-1 .. L-3; layer 5's logits were never stored
    assert hooks.rl_layers == frozenset(range(3, 39)) - {5}


def test_score_reads_the_documented_layers(hooks):
    """Fill each layer's features with its layer number and check which layer every raw block reads."""
    inputs = hooks.score_inputs
    layers = torch.arange(40)
    inputs['h'].copy_(layers[:, None].half().expand_as(inputs['h']))
    inputs['rl'].copy_(layers[:, None].half().expand_as(inputs['rl']))
    inputs['routes'].copy_(layers[:, None].expand_as(inputs['routes']))
    inputs['prev'].copy_(100 + layers[:, None].expand_as(inputs['prev']))
    hooks._score()
    raw, targets = hooks.bank.raw, torch.arange(6, 40)
    assert torch.equal(raw['h'][:, 0].long(), targets)
    assert torch.equal(raw['h_prev'][:, 0].long(), targets - 1)
    assert torch.equal(raw['curr'][:, 0], targets)
    for d in range(1, 4):
        assert torch.equal(raw['win'][:, d - 1, 0], targets - d)
        assert torch.equal(raw['rl'][:, (d - 1) * 256].long(), targets - d)
    for d in range(1, 6):
        assert torch.equal(raw['prevwin'][:, d - 1, 0], 100 + targets - d)
    assert torch.equal(hooks.probs, hooks.bank.out)


def test_prior_holds_at_step_zero_and_moves_after(hooks):
    hooks.score_inputs['routes'][:, :] = torch.arange(8)
    hooks._score()  # step 0: no decoded token yet
    assert not hooks.raw['prior_before'].any()
    hooks.step.fill_(1)
    hooks._score()
    prior = hooks.raw['prior_before']
    assert torch.allclose(prior[:, :8], torch.full((34, 8), 1 - 0.9)) and not prior[:, 8:].any()


def test_end_of_token_updates(hooks, cfg):
    hooks.live['h'].fill_(3)
    hooks.before_decode(4)
    assert int(hooks.step) == 4 and int(hooks.demand.gen) == 5
    hooks.after_decode(4, torch.tensor([17]), torch.tensor([cfg.features.token_table_rows + 2]))
    assert hooks.score_inputs['h'].eq(3).all()
    assert int(hooks.raw['tid']) == 17
    assert int(hooks.raw['upcoming_tid']) == 0  # an id without a table row is replaced, not looked up
    assert int(hooks.unscored_ids) == 1
    assert torch.allclose(hooks.raw['pos'], torch.full((34, 1), 5 / 256))


def test_reset_restores_the_capture_state(hooks):
    """A recapture after serving must start from step 0: the prior holds and nothing is copied."""
    hooks.before_decode(4)
    hooks.busy.fill_(1)
    hooks.layer.fill_(12)
    hooks.begin_request(-1)
    assert int(hooks.step) == 0 and int(hooks.demand.gen) == 1
    assert int(hooks.busy) == 0 and int(hooks.layer) == -1
    hooks.score_inputs['routes'][:, :] = torch.arange(8)
    hooks._score()
    assert not hooks.raw['prior_before'].any()


def test_begin_request_clears_request_state(hooks):
    hooks.raw['prior_before'].fill_(1)
    hooks.demand.ids.fill_(3)
    hooks.records.fill_(9)
    hooks.begin_request(7)
    assert not hooks.raw['prior_before'].any() and hooks.demand.ids.eq(-1).all() and hooks.records.eq(-1).all()


def test_load_bank_shapes(cfg):
    if not (PUBLISHED / 'model.safetensors').is_file():
        pytest.skip('published weights absent')
    loaded_cfg, state = bank_loader.load(PUBLISHED)
    assert loaded_cfg == cfg and state['proj_w'].shape[0] == 34


def test_logger_reaches_the_server_log():
    """FreeToken's loggers do not propagate, so the request summaries need a FreeToken logger."""
    import logging
    assert ft.logger.isEnabledFor(logging.INFO) and ft.logger.handlers


# --- learned eviction bundle loader (evict.EvictRows), versions 1 and 2 -------------------------
# A small synthetic geometry: 40 layers, heads on L06-L39 as the release, 4 experts, 8 trunk units.
_TARGETS, _E, _UP, _GROUP, _L = list(range(6, 40)), 4, 8, 4, 40


def _bundle(path, version, layers, source_layers=None, **changes):
    torch.manual_seed(0)
    data = dict(version=version, layers=list(layers), horizon_tokens=8, release_sha256='a' * 64,
                weight=torch.randn(len(layers), _E, _UP).bfloat16(), bias=torch.randn(len(layers), _E).bfloat16())
    if source_layers is not None:
        data['source_layers'] = list(source_layers)
    data.update(changes)
    torch.save(data, path)
    return path


def _rows(path, release='a' * 64):
    from commitmoe_prefetch.evict import EvictRows
    return EvictRows(path, _TARGETS, _UP, _E, _GROUP, torch.device('cpu'), release, _L)


def test_evict_v1_reads_its_own_head(tmp_path):
    # The exporter's provenance entry 'sources' is not a source-layer list and is ignored.
    rows = _rows(_bundle(tmp_path / 'b.pt', 1, [8, 11, 39], sources={8: {'sha256': 'x'}}))
    assert rows.source_layers == [8, 11, 39] and rows.heads.tolist() == [2, 5, 33]


def test_evict_v2_rows_share_one_source(tmp_path):
    rows = _rows(_bundle(tmp_path / 'b.pt', 2, [0, 1, 5, 8], source_layers=[6, 6, 6, 8]))
    assert rows.layers == [0, 1, 5, 8] and rows.heads.tolist() == [0, 0, 0, 2]


def test_evict_v2_codes_equal_v1_codes_of_the_same_rows(tmp_path):
    """Version 2 changes only which trunk row is read; the quantized rows are the version 1 ones."""
    v1 = _rows(_bundle(tmp_path / 'a.pt', 1, [8, 11]))
    v2 = _rows(_bundle(tmp_path / 'b.pt', 2, [8, 11], source_layers=[8, 11]))
    for name in ('codes', 'scale', 'bias', 'heads'):
        assert torch.equal(getattr(v1, name), getattr(v2, name)), name


@pytest.mark.parametrize('version, layers, sources, changes, message', [
    (1, [0, 8], None, {}, r'source layers \[0\] have no prefetch head'),
    (1, [8], [8], {}, 'version 1 bundle has source_layers'),
    (2, [0, 8], None, {}, 'needs integer source_layers'),
    (2, [0, 8], [6], {}, '1 source layers for 2 owner layers'),
    (2, [0, 8], [5, 8], {}, r'source layers \[5\] have no prefetch head'),
    (2, [8, 0], [8, 6], {}, 'ascending and unique'),
    (2, [8, 8], [8, 8], {}, 'ascending and unique'),
    (2, [0, 40], [6, 6], {}, r'owner layers \[40\] are not layers 0..39'),
    (2, [0, 8], [6.0, 8], {}, 'needs integer source_layers'),
    (2, [0, 8], [6, 8], dict(horizon_tokens=16), 'horizon 16 tokens'),
    (2, [0, 8], [6, 8], dict(release_sha256='b' * 64), 'served release'),
    (2, [0, 8], [6, 8], dict(weight=torch.full((2, _E, _UP), float('nan')).bfloat16()), 'non-finite weight'),
    (2, [0, 8], [6, 8], dict(bias=torch.zeros(2, _E)), 'bias torch.float32'),
    (3, [8], [8], {}, 'version 3'),
])
def test_evict_bundle_refusals(tmp_path, version, layers, sources, changes, message):
    with pytest.raises(ValueError, match=message):
        _rows(_bundle(tmp_path / 'b.pt', version, layers, sources, **changes))


@pytest.mark.parametrize('bundle', [p for p in os.environ.get('EXP526_EVICT_BUNDLES', '').split(':') if p])
def test_evict_released_bundles_load(bundle):
    """The exported bundles (EXP526_EVICT_BUNDLES, ':'-separated) load against the published release."""
    from commitmoe_prefetch.evict import EvictRows
    if not (PUBLISHED / 'config.json').is_file():
        pytest.skip('published folder absent')
    cfg = PredictorConfig.model_validate(json.loads((PUBLISHED / 'config.json').read_text()))
    release_sha256 = json.loads((PUBLISHED / 'manifest.json').read_text())['files']['model.safetensors']
    rows = EvictRows(Path(bundle), list(cfg.target_layers), cfg.architecture.up_dim, cfg.base_model.num_experts,
                     cfg.serving_reference.int8_group, torch.device('cpu'), release_sha256,
                     cfg.base_model.num_layers)
    assert len(rows.layers) == len(rows.source_layers) == rows.codes.shape[0]
