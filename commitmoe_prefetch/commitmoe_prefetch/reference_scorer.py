"""Plain-PyTorch reference of the BF16 bank: feature assembly, scoring, EMA update.

This is the executable form of config.json's feature contract, for anyone porting the
predictor to another runtime. It mirrors experiments/exp483_bank.assemble and
experiments/exp460_runtime.PrecisionBank.score operation for operation, including the
FP16 roundings, so on the CPU it reproduces the served BF16 bank bit for bit
(check_equivalence.py). The served int8 bank differs from it by the int8 weights and by
adding biases in FP32 before the BF16 rounding (experiments/exp493_qbank.QBank.score).

Raw inputs for H heads, head i targeting layer L = target_layers[i], all from the end of token t:
  h        (H, D) fp16  token t's MoE input hidden state at layer L
  h_prev   (H, D) fp16  the same at layer L-1
  win      (H, 3, k)    token t's routed expert ids at layers L-1, L-2, L-3
  prevwin  (H, 5, k)    token t-1's routed expert ids at layers L-1 .. L-5. At the first decode
                        step the deployed runtime passed all zeros, which mark expert 0 (not
                        "no expert"); a port should do the same to match the measured run.
  curr     (H, k)       token t's routed expert ids at layer L
  prior_before (H, E) fp32  the EMA prior before token t (zeros at reset)
  pos      (H, 1) fp32  (decode step t + 1) / 256
  rl       (H, 3E) fp16 token t's router logits at layers L-1, L-2, L-3, concatenated; the
                        layers in router_logit_layers_zeroed (layer 5) are zeros
  tid      (1,)         token t's id (the token consumed by step t)
  upcoming_tid (1,)     the already-sampled id of token t+1
Token ids must be text ids below token_table_rows; scores() raises on any other id.
"""
import torch
import torch.nn.functional as F


def assemble(raw: dict[str, torch.Tensor], num_experts: int, ema_beta: float):
    n = raw['h'].shape[0]
    h = raw['h'].float()
    hd = (raw['h'] - raw['h_prev']).float()  # the subtraction rounds to FP16, as in training
    win = torch.zeros(n, raw['win'].shape[1], num_experts, device=h.device).scatter_(2, raw['win'], 1).flatten(1)
    prev = torch.zeros(n, raw['prevwin'].shape[1], num_experts, device=h.device).scatter_(2, raw['prevwin'], 1).flatten(1)
    sticky = torch.zeros(n, num_experts, device=h.device).scatter_(1, raw['curr'], 1)
    prior = advance_prior(raw['prior_before'], raw['curr'], ema_beta).half().float()
    feats = torch.cat([h, win, prior, raw['pos'], raw['rl'].float(), prev, hd, sticky], 1)
    return feats, h, hd


def advance_prior(prior: torch.Tensor, curr: torch.Tensor, ema_beta: float) -> torch.Tensor:
    """prior * beta, then + (1 - beta) at token t's experts (FP32). The same update, applied
    after scoring, carries the prior from token t to token t+1."""
    out = prior * ema_beta
    return out.scatter_(1, curr, out.gather(1, curr) + (1 - ema_beta))


def linear(state, x, name):
    out = torch.bmm(state[name + '_w'], x.unsqueeze(2)).squeeze(2)
    bias = state.get(name + '_b')
    return out if bias is None else out + bias


def scores(state: dict[str, torch.Tensor], raw: dict[str, torch.Tensor], num_experts: int,
           ema_beta: float) -> torch.Tensor:
    """(H, E) fp32 uncalibrated head outputs (logits before the per-head calibration)."""
    rows = state['table'].shape[0]
    for name in ('tid', 'upcoming_tid'):
        # F.embedding would raise an IndexError on most ids out of range, but a negative id
        # is legal Python indexing elsewhere; reject both explicitly with the reason.
        if bool(((raw[name] < 0) | (raw[name] >= rows)).any()):
            raise ValueError(f'{name} {raw[name].tolist()}: the token table covers text ids 0..{rows - 1} only')
    dtype = torch.bfloat16
    feats, h, hd = assemble(raw, num_experts, ema_beta)
    feats, h, hd = feats.to(dtype), h.to(dtype), hd.to(dtype)
    h_hat = h + linear(state, F.gelu(linear(state, torch.cat([h, hd], 1), 'fh_down')), 'fh_up')
    trv = torch.bmm(state['W'], h_hat.unsqueeze(2)).squeeze(2) + state['router_b']
    emb = F.embedding(raw['tid'], state['table']).expand(len(h), -1)
    nxt = F.embedding(raw['upcoming_tid'], state['table']).expand(len(h), -1)
    x = torch.cat([feats, trv, emb, nxt], 1)
    return linear(state, F.gelu(linear(state, linear(state, x, 'proj'), 'up')), 'out').float()


def probabilities(state: dict[str, torch.Tensor], raw: dict[str, torch.Tensor], num_experts: int,
                  ema_beta: float) -> torch.Tensor:
    """(H, E) fp32 calibrated probabilities that each expert is routed at the target layer for token t+1."""
    return torch.sigmoid(scores(state, raw, num_experts, ema_beta) * state['a'] + state['b'])
