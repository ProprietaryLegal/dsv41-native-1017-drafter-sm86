# N2 aggregate evidence

These JSON files contain aggregate measurements and comparison statistics, not
model weights, captured tokens, tensors or element-level fixture values.

- `qualification-summary.json`: 171 component checks, eight graph cases,
  full-emulated model probes for both rank paths, published packed-FP8 dense
  and expert-projection oracles, exact shared/post-TP bit checks.
- `final_accept_all.json`, `final_accept_vs_kx.json`: realized acceptance
  estimates on the same 538 determined live captured blocks. Standard errors
  do not establish an acceptance gain or statistical equivalence.
- `final_compare_*.json`: captured draft/routing agreement (453 rounds,
  568 total blocks). Prefix differences require V1 state/causal classification.
- `old_vs_fresh_kx.json`: prior native arm reproduces its earlier saved replay.
- `repeated-summary.json`: three interleaved settled solo repetitions of each
  arm/rank; median of three run p50s, their full values, kernel gaps,
  telemetry, source hashes and memory/load ranges. Ranges are not confidence
  intervals. Solo collective stand-ins do not measure physical TP2 latency.

Source was frozen at `5904787` before final timing. Later changes are evidence,
reporting or profiler synchronization only. Full private detail and original
run JSONs/logs remain under `~/dsv41-offload/n2/{results,logs}` on X86host.

CPU-only aggregate reproduction on X86host:

```bash
# (site-specific paths file removed for publication)
cd ~/dsv41-offload/n2/repo
CUDA_VISIBLE_DEVICES= ~/dsv41-offload/e/venv/bin/python -m offload.dspark.native.tools.n2.collect_qualification \
  ~/dsv41-offload/n2/results --output /tmp/n2-qualification-summary.json
CUDA_VISIBLE_DEVICES= ~/dsv41-offload/e/venv/bin/python -m offload.dspark.native.tools.n2.collect_repeats \
  ~/dsv41-offload/n2/results/repeated --output /tmp/n2-repeated-summary.json
```

To rerun timing, launch `native/tools/n2/repeat.sh` from detached tmux on X86host.
It uses `run_locked.sh` for each separate bounded acquisition of GPU-DRAFTER1.
Use a fresh `N2_REPEAT_OUTPUT` directory to preserve these receipts. Physical
pair runs belong to the coordinator's separate `pair.sh` handoff and window.
