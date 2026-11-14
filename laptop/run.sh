#!/usr/bin/env bash
# run.sh LOGNAME [gen.sh args...]: runs gen.sh in this shell, logs to ~/band/LOGNAME.log, stamps start/end.
cd ~/band; log=$1; shift
rm -f run.end; date +%s > run.start
HF_HUB_DISABLE_XET=1 ~/band/heavy.sh ~/band/gen.sh "$@" > ~/band/$log.log 2>&1; echo EXIT=$? >> ~/band/$log.log
date +%s > run.end
