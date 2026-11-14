#!/usr/bin/env bash
while true; do
  echo "=== $(date +%T)"
  nvidia-smi --query-gpu=memory.used,temperature.gpu,utilization.gpu --format=csv,noheader
  free -m | awk 'NR>1{print $1,$3}'
  sleep 15
done
