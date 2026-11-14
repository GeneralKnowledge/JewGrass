#!/usr/bin/env bash
# heavy.sh CMD...: run a CPU/GPU-heavy job so the rest of the laptop (network, claude remote-control) stays responsive.
#  - nice 15 and idle I/O class: everything else gets the CPU and disk first
#  - 8 of 12 CPU threads for numeric libraries (torch, numpy, BLAS)
#  - oom_score_adj 500: if memory runs out, the kernel kills this job before claude or system services
echo 500 > /proc/self/oom_score_adj 2>/dev/null || true
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8
exec nice -n 15 ionice -c3 "$@"
