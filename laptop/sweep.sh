#!/usr/bin/env bash
# sweep.sh SECONDS...: one-shot full-lyrics runs at each length; stops at the first failure.
cd ~/band; rm -f sweep.end
for s in "$@"; do
  echo "=== SWEEP ${s}s $(date +%T)" >> monitor.log
  ~/band/run.sh len$s ~/band/full-caption.txt ~/band/your-landlord-full-lyrics.txt $s 1111 your-landlord-len$s 140
  grep -q "OUTPUT=" len$s.log || break
  [ -f stop.hot ] && break
done
date +%s > sweep.end
