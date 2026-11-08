#!/usr/bin/env bash
#
# Replication setup: build this engine, download the pinned model and predictor, check every hash.
#
# What it does, in order:
#   1. Checks the host: Linux x86_64, an NVIDIA driver of major version 580 or later (CUDA 13), nvcc 13.x (the
#      engine's C++/CUDA extensions compile at install time, with SASS for sm_120), uv, git, host RAM and disk.
#   2. Checks that the engine code (python/, commitmoe_prefetch/, setup.py, pyproject.toml) is unchanged from the
#      tested engine commit. The replication files themselves live on top of that commit.
#   3. Creates the virtual environment .venv at the repository root with Python 3.12 and installs the engine with
#      the documented commands (uv pip install -e ".[accel]" and -e ./commitmoe_prefetch).
#   4. Downloads Qwen/Qwen3.5-35B-A3B-FP8 (about 35 GB) and the expert predictor at pinned revisions into the
#      Hugging Face cache (HF_HOME is respected), and checks the sha256 of every predictor file.
#   5. Checks the prompt files and the R584 eviction bundle by sha256, and that the bundle was exported against the
#      downloaded predictor release.
#   6. Writes replication/local_paths.json, which run_bench.py reads.
#
# It never stops a process and never changes a system setting. It is safe to run again: uv and the Hugging Face
# cache skip work that is already done.
#
# Usage, from the repository root:   bash replication/setup.sh
set -euo pipefail

# ---- pins (the values of the RTX 4080 runs) ------------------------------------------------------------------
ENGINE_COMMIT="16d7cd485d68599972e3b9c2ae6798c32d72cb49"
MODEL_REPO="Qwen/Qwen3.5-35B-A3B-FP8"
MODEL_REVISION="9d1823d2dee688a6b25e77009dc727688c44936e"
PREDICTOR_REPO="RASMUS/qwen3.5-35b-a3b-fp8-expert-predictor"
PREDICTOR_REVISION="d6a9f5984fbc1a65cc2e2976ddbdb6f5180c5411"
PY_VERSION="3.12"            # the version of the tested environment
MIN_DRIVER_MAJOR=580         # CUDA 13 needs driver r580 or later (docs/expert-predictor.md)
# Host RAM: the engine pins all 40 x 256 expert banks (32.2 GB) plus the host embedding tables (about 1 GB), and the
# bounded weight reader holds up to about 7.4 GB more while loading. 48 GB is the smallest total that leaves room
# for the OS. run_bench.py checks the available memory again before every serve.
MIN_MEM_TOTAL_GB=48
MIN_DISK_FREE_GB=45          # the model is about 35 GB; the predictor is 0.35 GB

PREDICTOR_SHA256="
6d1d968fb225eca367cb7f0b8831ab012a35d92b547e945e17ef8e7b05c3e5cc  LICENSE
862f73e7a0d143e22e853e58de31bf772d9b51dfb4a7b71da2bff3a28a09440a  NOTICE
54fc0b913f44bd6a19554e956254850071cc38795d68735934d8ec98f818db1a  README.md
198b03800f11c9d04b6e10e01a810b58777d0045c13eccd17a14e8d825247193  bank_loader.py
bd0a3ea3b6c8d9a1099ce2f42543e8831a7dade8137d5b3216ed7aa75e2a791b  config.json
a794508507c48884a3c6bc008eb5a51e963e50b61d5b8e3d462322bb706f4718  manifest.json
32c631a276fcf1d34cb079e253c371c3745f05f59ea3c217c855f85fd7cccecf  model.safetensors
09718946bf69e8f3c77b5af45b4ad664c1ccc631435a14c1c6858a903e1832d1  parity_fixture.safetensors
60e60a1706cf14260dbfb962081c218cb87c51d329af3e61422583d23737d2f8  reference_scorer.py
8e65f523884fccba36fa7f7aad643675badaff570bea8a3b8d5ad7b9512a9eed  spec.py"
LOCAL_SHA256="
7c0225333c0e4b7a5d5b808711770f9b81bf49e0f34c30127969b273ddce9af6  prompts/code40.jsonl
3ff3d8bb381ab9f3556d814f69df3f81c2db0efd2468164dd3a1d7bf60eb9084  prompts/math_qa80.jsonl
87421c2054eca01450ba34b88e7fad5e8c5ff206a8ac5616938f6b6df59344f7  assets/r584_shared_evict_v1.pt"

say()  { printf '==> %s\n' "$*"; }
die()  { printf '[setup stopped] %s\n' "$*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "$HERE/.." && pwd -P)"
VENV="$REPO/.venv"
PY="$VENV/bin/python"

# ---- 1. host checks --------------------------------------------------------------------------------------------
say "1/6 host checks"
[ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ] || die "needs Linux x86_64, found $(uname -s) $(uname -m)"
command -v git >/dev/null || die "git not found"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver (r${MIN_DRIVER_MAJOR} or later)"
command -v uv >/dev/null || die "uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
GPU_LINE="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
say "GPU: $GPU_LINE"
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" = "1" ] || \
  say "note: more than one GPU. run_bench.py uses GPU 0 (CUDA_VISIBLE_DEVICES=0)."
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
[ "$DRIVER_MAJOR" -ge "$MIN_DRIVER_MAJOR" ] || die "driver $DRIVER_MAJOR is older than r$MIN_DRIVER_MAJOR (CUDA 13)"
NVCC="$(command -v nvcc || true)"
[ -n "$NVCC" ] || for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-13*/bin/nvcc; do [ -x "$c" ] && NVCC="$c" && break; done
[ -n "$NVCC" ] && [ -x "$NVCC" ] || die "nvcc not found: install the CUDA 13 toolkit (the engine compiles its CUDA kernels)"
NVCC_VERSION="$("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
case "$NVCC_VERSION" in 13.*) ;; *) die "nvcc $NVCC_VERSION found at $NVCC; the engine needs CUDA 13.x (torch cu130)";; esac
CUDA_HOME_DIR="$(cd "$(dirname "$NVCC")/.." && pwd -P)"
say "nvcc $NVCC_VERSION, CUDA_HOME=$CUDA_HOME_DIR"
MEM_TOTAL_GB="$(awk '/^MemTotal:/ {printf "%d", $2 * 1024 / 1e9}' /proc/meminfo)"
[ "$MEM_TOTAL_GB" -ge "$MIN_MEM_TOTAL_GB" ] || die "host RAM ${MEM_TOTAL_GB} GB is below ${MIN_MEM_TOTAL_GB} GB (pinned expert banks)"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"
DISK_FREE_GB="$(df -B1 --output=avail "$HF_CACHE" | tail -1 | awk '{printf "%d", $1 / 1e9}')"
say "RAM ${MEM_TOTAL_GB} GB, ${DISK_FREE_GB} GB free at $HF_CACHE"

# ---- 2. engine identity ----------------------------------------------------------------------------------------
say "2/6 engine identity"
git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || die "$REPO is not a git clone; clone the repository with git"
git -C "$REPO" merge-base --is-ancestor "$ENGINE_COMMIT" HEAD || die "HEAD does not contain engine commit $ENGINE_COMMIT"
ENGINE_PATHS=(python commitmoe_prefetch setup.py pyproject.toml)
git -C "$REPO" diff --quiet "$ENGINE_COMMIT" -- "${ENGINE_PATHS[@]}" || \
  die "engine code differs from the tested commit $ENGINE_COMMIT (git diff $ENGINE_COMMIT -- ${ENGINE_PATHS[*]})"
[ -z "$(git -C "$REPO" status --porcelain -- "${ENGINE_PATHS[@]}")" ] || die "uncommitted changes in the engine code"
say "engine code equals $ENGINE_COMMIT"

# ---- 3. environment and install --------------------------------------------------------------------------------
say "3/6 virtual environment and engine build (the first build compiles CUDA kernels and takes several minutes)"
[ -x "$PY" ] || uv venv --python "$PY_VERSION" "$VENV"
( cd "$REPO" && CUDA_HOME="$CUDA_HOME_DIR" PATH="$CUDA_HOME_DIR/bin:$PATH" uv pip install --python "$PY" -e ".[accel]" )
( cd "$REPO" && uv pip install --python "$PY" -e ./commitmoe_prefetch )
"$PY" - <<'EOF'
import torch, triton, freetoken, commitmoe_prefetch  # noqa: F401  (import check)
assert torch.cuda.is_available(), 'torch sees no CUDA device'
cap = torch.cuda.get_device_capability(0)
print(f'==> torch {torch.__version__}, CUDA {torch.version.cuda}, triton {triton.__version__}, '
      f'GPU 0 {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}')
EOF

# ---- 4. downloads ----------------------------------------------------------------------------------------------
say "4/6 downloads (the model is about 35 GB; ${DISK_FREE_GB} GB free)"
HF="$VENV/bin/hf"
[ -x "$HF" ] || die "the hf command is missing from the environment (huggingface_hub>=1.5 provides it)"
if ! MODEL_DIR="$("$HF" download "$MODEL_REPO" --revision "$MODEL_REVISION" --quiet)"; then
  die "model download failed"
fi
[ "$DISK_FREE_GB" -ge "$MIN_DISK_FREE_GB" ] || say "note: less than ${MIN_DISK_FREE_GB} GB was free before the download"
PRED_DIR="$("$HF" download "$PREDICTOR_REPO" --revision "$PREDICTOR_REVISION" --quiet)" || die "predictor download failed"
MODEL_DIR="$(printf '%s' "$MODEL_DIR" | tail -1)"; PRED_DIR="$(printf '%s' "$PRED_DIR" | tail -1)"
[ -f "$MODEL_DIR/config.json" ] || die "model folder $MODEL_DIR has no config.json"
case "$MODEL_DIR" in *"/snapshots/$MODEL_REVISION") ;; *) die "model folder $MODEL_DIR is not revision $MODEL_REVISION";; esac
say "model: $MODEL_DIR"
say "predictor: $PRED_DIR"

# ---- 5. hashes -------------------------------------------------------------------------------------------------
say "5/6 sha256 checks"
( cd "$PRED_DIR" && printf '%s\n' "$PREDICTOR_SHA256" | sed '/^$/d' | sha256sum --check --strict --quiet ) \
  || die "a predictor file differs from its pinned sha256"
( cd "$HERE" && printf '%s\n' "$LOCAL_SHA256" | sed '/^$/d' | sha256sum --check --strict --quiet ) \
  || die "a prompt file or the R584 bundle differs from its pinned sha256"
"$PY" - "$HERE/assets/r584_shared_evict_v1.pt" <<'EOF'
import sys, torch
meta = torch.load(sys.argv[1], map_location='cpu', weights_only=True)['meta']
want = '32c631a276fcf1d34cb079e253c371c3745f05f59ea3c217c855f85fd7cccecf'   # the predictor's model.safetensors
assert meta.get('release_sha256') == want, f"bundle exported against {meta.get('release_sha256')}, not {want}"
print('==> R584 bundle matches the predictor release')
EOF

# ---- 6. paths for run_bench.py ---------------------------------------------------------------------------------
say "6/6 writing replication/local_paths.json"
"$PY" - "$HERE/local_paths.json" "$PY" "$CUDA_HOME_DIR" "$MODEL_DIR" "$PRED_DIR" <<'EOF'
import json, sys
out, python, cuda_home, model, predictor = sys.argv[1:]
json.dump(dict(python=python, cuda_home=cuda_home, model=model, predictor=predictor), open(out, 'w'), indent=1)
EOF
say "setup complete. Next: $PY replication/run_bench.py"
