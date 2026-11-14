#!/usr/bin/env bash
# setup.sh [ace] [applio]: install ACE-Step 1.5 and/or Applio (RVC) on a Kaggle GPU machine.
# Everything goes in /kaggle/temp (fast scratch disk that isn't saved), so outputs in /kaggle/working stay small.
# Same versions as the band laptop: PyTorch 2.10.0 built for CUDA 12.6, which runs on every Kaggle GPU
# (T4, P100, L4). Pinned commits so a run next month installs exactly what this one did.
set -euo pipefail
T=/kaggle/temp
ACE_COMMIT=ca1e85f        # ACE-Step 1.5, 2026-08-29
APPLIO_COMMIT=6d5d951     # Applio, 2026-11-04
TORCH="torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0"
CU=https://download.pytorch.org/whl/cu126
mkdir -p $T
command -v uv >/dev/null || pip install -q uv
export UV_LINK_MODE=copy UV_CACHE_DIR=$T/uv-cache MPLBACKEND=Agg

clone() {  # clone REPO DIR COMMIT
  [ -d "$2/.git" ] && return
  git clone -q --filter=blob:none "$1" "$2" && git -C "$2" checkout -q "$3"
}

if [[ " $* " == *" ace "* ]]; then
  t0=$SECONDS
  clone https://github.com/ace-step/ACE-Step-1.5 $T/ACE-Step-1.5 $ACE_COMMIT
  cd $T/ACE-Step-1.5
  # Upstream's lockfile, minus its CUDA 12.8 torch, then the CUDA 12.6 torch the laptop uses.
  uv sync --frozen -q --no-install-package torch --no-install-package torchvision --no-install-package torchaudio
  uv pip install -q --python .venv/bin/python $TORCH --index-url $CU
  # Fetch the turbo model now, once, so two GPU workers don't both download it at the same time.
  .venv/bin/python -c "from acestep.model_downloader import ensure_main_model as e; print(e())"
  echo "ACE-Step ready in $((SECONDS - t0)) s"
fi

if [[ " $* " == *" applio "* ]]; then
  t0=$SECONDS
  clone https://github.com/IAHispano/Applio $T/Applio $APPLIO_COMMIT
  cd $T/Applio
  [ -x $T/venv-applio/bin/python ] || uv venv -q $T/venv-applio --python 3.12
  printf 'torch==2.10.0\ntorchaudio==2.10.0\n' > $T/applio-overrides.txt   # same override as the laptop
  uv pip install -q --python $T/venv-applio/bin/python -r requirements-engine.txt --override $T/applio-overrides.txt \
     --extra-index-url $CU --index-strategy unsafe-best-match
  # fp16 training on Turing and newer (T4, L4); fp32 on the P100, like the laptop's Pascal card.
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  prec=$([ "${cap%%.*}" -ge 7 ] && echo fp16 || echo fp32)
  echo "{\"precision\": \"$prec\"}" > $T/applio-config.json
  $T/venv-applio/bin/python core.py prerequisites --pretraineds-hifigan --models --no-exe > $T/applio-prereq.log 2>&1 \
    || { tail -20 $T/applio-prereq.log; exit 1; }
  echo "Applio ready in $((SECONDS - t0)) s (training precision $prec)"
fi
