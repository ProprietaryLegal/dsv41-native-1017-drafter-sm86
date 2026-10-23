#!/usr/bin/env bash
# Execute on x86host. Each acquisition ends within 14 minutes including compilation.
set -euo pipefail
# (site-specific paths file removed for publication)
[[ $(hostname) == x86-host ]] || { echo 'must run on x86host' >&2; exit 2; }
mkdir -p ~/dsv41-offload/locks
exec 9>~/dsv41-offload/locks/gpu-DRAFTER1.lock
flock -w 1800 -E 75 9
export KX_GROUPED_LOCK_FD=9
export CUDA_VISIBLE_DEVICES=GPU-DRAFTER1-UUID
exec timeout --signal=TERM --kill-after=10 830 "$@"
