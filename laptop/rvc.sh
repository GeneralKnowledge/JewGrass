#!/usr/bin/env bash
# rvc.sh <applio core.py command> [options]: run Applio's CLI with fp32 precision (the GTX 1060 has no usable
# fp16/bf16), low priority (heavy.sh) and, for GPU commands, the shared GPU lock.
# ~/band/tools/rvc/applio-config.json contains: {"precision": "fp32"}
export APPLIO_CONFIG_FILE=~/band/tools/rvc/applio-config.json
cd ~/band/tools/rvc/Applio
exec flock ~/band/.gpu.lock ~/band/heavy.sh ../.venv/bin/python core.py "$@"
