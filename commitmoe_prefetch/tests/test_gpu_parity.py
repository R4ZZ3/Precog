"""GPU parity of the package against the research code of the measured run (experiments/).

- The int8 bank: Int8Bank equals exp493_qbank.QBank built from the same buffers, bitwise,
  on the published parity fixture inputs.
- capture_layer, select_scored, retire and publishing_pump equal the research kernels on
  the same state. The pump runs with no demand copy busy and no section expired, so both
  copy every selected expert in full and the result does not depend on timing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

HERE = Path(__file__).resolve().parent
# The research code (exp483*_kernels.py) lives in the CommitMoE research workspace, not in this repo.
# Set COMMITMOE_EXPERIMENTS_DIR to its experiments/ folder to run these tests; they skip without it.
EXPERIMENTS = Path(os.environ.get('COMMITMOE_EXPERIMENTS_DIR', '/nonexistent'))
# A local copy of the published predictor (huggingface-cli download RASMUS/qwen3.5-35b-a3b-fp8-expert-predictor
# --local-dir DIR). The tests that need it skip when COMMITMOE_PREDICTOR_DIR is unset.
PUBLISHED = Path(os.environ.get('COMMITMOE_PREDICTOR_DIR', '/nonexistent'))
E, K, D, L = 256, 8, 2048, 40


@pytest.fixture(scope='module')
def legacy():
    if not (EXPERIMENTS / 'exp483_kernels.py').is_file():
        pytest.skip('research code absent (COMMITMOE_EXPERIMENTS_DIR)')
    sys.path.insert(0, str(EXPERIMENTS))
    try:
        import exp483_kernels
        import exp483b_kernels
        yield exp483_kernels, exp483b_kernels
    finally:
        sys.path.remove(str(EXPERIMENTS))


def test_int8_bank_equals_research_bank(legacy):
    from safetensors.torch import load_file

    from commitmoe_prefetch import bank_loader
    from commitmoe_prefetch.qbank import Int8Bank
    if not (PUBLISHED / 'parity_fixture.safetensors').is_file():
        pytest.skip('published folder absent')
    import exp493_qbank

    cfg, state = bank_loader.load(PUBLISHED)
    dev = torch.device('cuda')
    bank = Int8Bank(cfg, state, dev)
    research = exp493_qbank.QBank.__new__(exp493_qbank.QBank)
    torch.nn.Module.__init__(research)
    research.compute_dtype = torch.bfloat16
    for name, t in bank.buffers.items():
        research.register_buffer(name, t)
    fixture = load_file(str(PUBLISHED / 'parity_fixture.safetensors'))
    raw_names = ('h', 'h_prev', 'win', 'prevwin', 'curr', 'prior_before', 'pos', 'rl', 'tid', 'upcoming_tid')
    samples = sorted({k.split('.')[0] for k in fixture if k.split('.', 1)[-1] in raw_names})
    assert samples, sorted(fixture)[:10]
    for sample in samples:
        raw = {name: fixture[f'{sample}.{name}'].to(dev) for name in raw_names}
        want = research(raw)
        got = bank(raw)
        assert torch.equal(want, got), sample


def test_capture_layer_equals_research(legacy):
    from commitmoe_prefetch import kernels
    old, _ = legacy
    dev = torch.device('cuda')
    hidden = torch.randn(1, D, dtype=torch.bfloat16, device=dev)
    logits = torch.randn(1, E, dtype=torch.bfloat16, device=dev)
    ids = torch.randint(0, E, (1, K), dtype=torch.int32, device=dev)
    out = [(torch.zeros(L, D, dtype=torch.float16, device=dev), torch.zeros(L, E, dtype=torch.float16, device=dev),
            torch.zeros(L, K, dtype=torch.int64, device=dev)) for _ in range(2)]
    for layer, keep in ((7, True), (5, False)):
        old.capture_layer[(1,)](*out[0], hidden, logits if keep else hidden, ids, LAYER=layer, KEEP_RL=keep)
        kernels.capture_layer[(1,)](*out[1], hidden, logits if keep else hidden, ids, LAYER=layer, KEEP_RL=keep,
                                    D=D, D_POW2=D, E=E, E_POW2=E, K=K, K_POW2=K)
    for want, got in zip(*out):
        assert torch.equal(want, got)


def _bank_rows(dev):
    """Host rows of every bank and the per-layer base pointers into them.

    The rows have the FP8 expert sizes of the served model (gate_up, its scales, down, its scales,
    in bytes), so a copy spans several CHUNK_BYTES chunks as in serving. One full row per
    (layer, expert) would need 40 x 256 x 3.00 MiB = 30 GiB, more than the GPU. So the layer blocks
    overlap: layer l starts at row l, and expert e of layer l is row l + e (296 rows, 888 MiB). Both
    kernels read through the same pointers, and a read at another layer or expert lands on another
    random row, unless the layer and expert errors cancel exactly.
    """
    sizes = [2 * 512 * D, 8 * 16 * 4, D * 512, 16 * 4 * 4]
    hosts = [torch.randint(0, 255, (L + E, s), dtype=torch.uint8, device=dev) for s in sizes]
    ptrs = torch.tensor([[h.data_ptr() + layer * s for h, s in zip(hosts, sizes)] for layer in range(L)],
                        dtype=torch.int64, device=dev)
    return sizes, hosts, ptrs


@pytest.mark.parametrize('section,cap', [(0, 2), (3, 2), (6, 3)])
def test_select_retire_pump_equal_research(legacy, section, cap):
    from commitmoe_prefetch import kernels
    old_select, old_pump = legacy[0], legacy[1]
    dev = torch.device('cuda')
    torch.manual_seed(section)
    sizes, hosts, host_ptrs = _bank_rows(dev)  # hosts must stay alive while the kernels read them
    sizes_t = torch.tensor(sizes, dtype=torch.int64, device=dev)
    slot_for_id = torch.where(torch.rand(L * E, device=dev) < 0.7, 1, -1).to(torch.int32)
    probs = torch.rand(34, E, device=dev)
    step_value = 3
    step = torch.full((1,), step_value, dtype=torch.int32, device=dev)
    gen = torch.full((1,), step_value + 1, dtype=torch.int64, device=dev)
    busy = torch.zeros(1, dtype=torch.int32, device=dev)
    layer = torch.full((1,), section * 5, dtype=torch.int32, device=dev)
    chunks = -(-sum(s // 8 for s in sizes) // (kernels.CHUNK_BYTES // 8))

    def stages():
        bufs = [torch.zeros(8, s, dtype=torch.uint8, device=dev) for s in sizes]
        return bufs, torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64, device=dev)

    def stage_state():
        ids = torch.full((10,), -1, dtype=torch.int32, device=dev)
        ids[:8] = torch.arange(8, dtype=torch.int32, device=dev)  # stale entries that retire must clear
        return ids, torch.full((10,), 7, dtype=torch.int64, device=dev)

    # research path: per-(step, section) rows for the first 256 steps
    old_bufs, old_ptrs = stages()
    old_ids, old_ready = stage_state()
    records = torch.zeros((256, 7, 4, 4), dtype=torch.int64, device=dev)
    cursors7 = torch.zeros((256, 7), dtype=torch.int32, device=dev)
    stripe_cursors = torch.zeros((256, 7, 8), dtype=torch.int32, device=dev)
    old_retire, old_publish = old_pump.retire, old_pump.publishing_pump
    old_retire[(1,)](old_ids, old_ready, SECTION=section)
    old_select.select_scored[(1,)](slot_for_id, probs, step, records, cursors7, SECTION=section, num_warps=8,
                                   FIRST_STEP=1, MIN_SCORE=0.0)
    old_publish[(8,)](host_ptrs, old_ptrs, sizes_t, busy, layer, step, records, stripe_cursors,
                      torch.zeros((256, 7, 52), dtype=torch.int32, device=dev),
                      torch.zeros((256, 7, 4), dtype=torch.int32, device=dev),
                      torch.zeros((256, 7, 52, 8, 3), dtype=torch.int64, device=dev),
                      torch.zeros(3, dtype=torch.int64, device=dev), old_ids, old_ready, gen,
                      SECTION=section, CAP=cap, STRIPES=8, num_warps=8, FIRST_STEP=1)

    # package path: one row per section
    new_bufs, new_ptrs = stages()
    new_ids, new_ready = stage_state()
    new_records = torch.full((7, 4), -1, dtype=torch.int64, device=dev)
    new_cursors = torch.full((7, 8), 5, dtype=torch.int32, device=dev)  # stale; select must clear them
    payload = torch.full((7, 4), 5, dtype=torch.int32, device=dev)
    stats = torch.zeros(4, dtype=torch.int64, device=dev)
    kernels.retire[(1,)](new_ids, new_ready, SECTION=section, SLOTS=4)
    kernels.select_scored[(1,)](slot_for_id, probs, step, new_records, new_cursors, payload, SECTION=section,
                                FIRST=6, FIRST_STEP=1, MIN_SCORE=0.0, E=E, SPAN_POW2=2048, num_warps=8)
    kernels.publishing_pump[(8,)](host_ptrs, new_ptrs, sizes_t, busy, layer, step, new_records, new_cursors,
                                  payload, stats, new_ids, new_ready, gen, SECTION=section, CAP=cap,
                                  FIRST_STEP=1, NUM_BANKS=4, E=E, QUOTA=4 * chunks, num_warps=8)
    torch.cuda.synchronize()

    assert torch.equal(records[step_value, section, :, 0], new_records[section])
    assert torch.equal(old_ids[:8], new_ids[:8]) and torch.equal(old_ready[:8], new_ready[:8])
    for want, got in zip(old_bufs, new_bufs):
        assert torch.equal(want, got)
    published = int((new_ready[:8] == step_value + 1).sum())
    assert published == min(cap, int((new_records[section] >= 0).sum())) and int(stats[3]) == published
