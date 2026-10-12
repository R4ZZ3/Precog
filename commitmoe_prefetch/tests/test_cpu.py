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
