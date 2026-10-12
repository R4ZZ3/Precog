# Expert prefetch for Qwen3.5-35B-A3B-FP8

This branch is a fork of [FreeToken](https://github.com/FlashML-org/FreeToken) (Apache-2.0) at
commit 2d8b9cc. It adds engine options for expert offloading and an optional learned expert
predictor. All changes are flags. Without them, the engine behaves as FreeToken 2d8b9cc.

The predictor supports one model only: `Qwen/Qwen3.5-35B-A3B-FP8`. The engine flags work with any
model that FreeToken serves with `--moe-strategy offload`, but we measured them only on this model.

## What the options do

When the experts do not all fit in VRAM, FreeToken keeps a GPU cache of expert slots and copies a
missing expert from host memory when a token routes to it. Decode then waits for the copy.

| Flag | Effect |
|---|---|
| `--moe-demand-copy-stripes 16` | Copies each missing expert as 16 parallel stripes on a high-priority stream. Decode only: prefill streams whole layers as upstream does. |
| `--expert-predictor <folder or HF repo>` | Loads the learned predictor. At the end of each token, it scores the experts that the next token can route to and copies up to 16 non-resident experts into a private 32 MiB stage. A later miss reads the staged copy if it is complete. Needs `--moe-demand-copy-stripes`. |
| `--pretouch-host-banks` | Writes each host expert bank once when it is allocated, before the weights are read. It is for load time. We did not measure it alone. |
| `--expert-load bounded` | Reads one checkpoint shard at a time. This keeps host RAM use bounded during load. We did not measure it alone. |

The predictor is [RASMUS/qwen3.5-35b-a3b-fp8-expert-predictor](https://huggingface.co/RASMUS/qwen3.5-35b-a3b-fp8-expert-predictor)
(34 int8 heads, about 200 MiB of VRAM). Its training prompts were 99% coding prompts.

## Install

Requirements are the same as FreeToken's ([install.md](install.md)): Linux x86_64, an NVIDIA GPU,
driver r580+ (CUDA 13), Python 3.10 or later, and host RAM for the expert banks (30 GiB for this model).

```bash
git clone <this fork> && cd <this fork>
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
uv pip install -e ./commitmoe_prefetch   # only for --expert-predictor
```

## Run

```bash
ft serve --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --text-model-only \
  --moe-demand-copy-stripes 16 \
  --expert-predictor RASMUS/qwen3.5-35b-a3b-fp8-expert-predictor \
  --pretouch-host-banks --expert-load bounded
```

To run without the predictor, remove `--expert-predictor`. Give no slot count. With no cache flag,
the engine sizes the expert slots and the KV cache from free VRAM (`--moe-cache-auto`, 90% of the
VRAM that is free before load, `--memory-ratio` changes it). The predictor's memory comes from the
expert slots, so the total VRAM stays the same. On our RTX 4080 16 GB, the predictor cost 67 of
3237 expert slots and 50 of 8284 KV pages.

`--text-model-only` keeps the vision tower out of pinned host memory. Use it for text-only serving,
or when CUDA cannot pin the 30 GiB of expert banks plus the vision tower.

## Limits

- The predictor serves one running request (`--max-running-requests 1`) with CUDA graphs at batch
  size 1. The engine refuses other settings when `--expert-predictor` is set.
- The predictor was trained on decode steps 0-255. Its position input exceeds the training range
  after 255 decode tokens. We did not measure long responses.
- A miss that the predictor did not stage is still copied on demand. The output does not depend on
  the predictor: in every benchmark run, the output was the same with and without it.

## Benchmark

Setup: one RTX 4080 16 GB under WSL2, `Qwen/Qwen3.5-35B-A3B-FP8`, one request at a time, 256 decode
tokens with thinking on, sampling temperature 1.0, top-k 20, top-p 0.95. Every engine sized its cache
from the same free VRAM (`--memory-ratio 0.9`) and used 13.49 GiB. Upstream is FreeToken 2d8b9cc with
`--expert-load serial`. One server per engine served 50 public prompts (10 per domain, domains mixed),
in two orders. None of these prompts was in the predictor's training data. Decode speed is the time
from the first to the last streamed event, divided by the generated tokens minus one.

The server command for the fork with the predictor (remove `--expert-predictor` for the fork alone):

```bash
ft serve --model Qwen/Qwen3.5-35B-A3B-FP8 --moe-strategy offload --moe-cache-auto \
  --memory-ratio 0.9 --max-running-requests 1 --cuda-graph-max-bs 1 --max-seq-len-override 8448 \
  --cache-type naive --text-model-only \
  --moe-demand-copy-stripes 16 --pretouch-host-banks --expert-load bounded \
  --expert-predictor RASMUS/qwen3.5-35b-a3b-fp8-expert-predictor
```

`--cache-type naive` turns off prefix reuse, so that every engine runs the same prefill. We used it
to compare outputs between engines. You do not need it for normal serving.

Decode tokens/s, the median over the 10 prompts of a domain:

| Domain | FreeToken 2d8b9cc | This fork | This fork + predictor |
|---|---|---|---|
| Coding (MBPP, BigCodeBench) | 33.1 | 62.1 | 63.3 |
| Math (MATH-500) | 29.8 | 58.0 | 58.5 |
| Chat (MT-Bench) | 35.2 | 64.6 | 63.9 |
| Finnish (MT-Bench, translated) | 36.9 | 65.8 | 65.2 |
| Summarization (CNN/DailyMail) | 32.2 | 60.7 | 61.0 |

- The fork without the predictor was faster than upstream on all 50 prompts. Per domain, the median of
  the per-prompt speed ratios was 1.80x to 1.95x.
- The predictor gave about 1% lower decode time per token on coding and math (faster on 9 of 10
  prompts in each). On chat, Finnish and summarization, we detected no clear difference. These effects
  are small, and we did not repeat them on independent servers.
- The median time to the first streamed event fell from 2.6 s (upstream) to 1.4 s (fork). We did not
  find the cause. The striped copies run only in decode.
- All three engines gave the same output text for every prompt.

Your speed depends on your GPU, your PCIe link and your free VRAM. With more free VRAM, more experts
stay on the GPU, and the predictor has fewer misses to hide.
