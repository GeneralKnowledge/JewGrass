#!/usr/bin/env bash
# One-command song: gen.sh CAPTION_FILE LYRICS_FILE SECONDS SEED OUTPUT_NAME [BPM]
# Output: ~/band/outputs/OUTPUT_NAME.wav. Uses --no-sync so uv keeps the Pascal (cu126) torch.
# Defaults are the settings that work on the GTX 1060 (fp32 compute, DiT loaded straight to GPU, never offloaded);
# see the BAND_* env vars at the top of gen.py to change them.
set -euo pipefail
[ $# -ge 5 ] || { echo "usage: $0 caption.txt lyrics.txt seconds seed name [bpm]"; exit 1; }
cap=$(realpath "$1"); lyr=$(realpath "$2")
cd ~/band/ACE-Step-1.5
exec uv run --no-sync python ../gen.py "$cap" "$lyr" "$3" "$4" "$5" "${6:-}"
