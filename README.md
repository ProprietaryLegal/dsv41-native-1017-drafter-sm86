# Native DeepSeek-V4.1-Flash-1017 drafting on RTX 3090 SM86: MXFP4 / FP8 kernels for DGX Spark offload

Run native-format draft expert and dense weights on an x86 host with two NVLinked RTX 3090s, for a target served on two DGX Spark (GB10) nodes. The fastest historical arm beats the deployed quantized drafter's step time on the real pair while measured acceptance remains equal within noise. The strict fidelity gate still fails, and the arithmetic fix is in progress.

**Hugging Face:** [Card and files](https://huggingface.co/ProprietaryLegal/dsv41-native-1017-drafter-sm86) · **Stack overview:** [GitHub](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · **Collection:** [All entries](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e)

<!-- pli:blurb:start -->

> Published by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai). PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work and discuss what your organization needs.

<!-- pli:blurb:end -->

## Status

2026-10-23 — opt-in research: merged for opt-in integration only; fast on the real NVLink pair and acceptance-equal in offline replay, but the strict fidelity gate FAILS, the fix remains in progress, latency may change, and native arms are not serving in production.

The latest corrected pilots pass the state, unexplained-disagreement and KL checks but still fail the attention gate. An official GPU attention component control also exceeds the reference attention band. That is an unresolved reference/arithmetic issue, not a waiver. There are no final corrected latency numbers to publish. The tables below describe the earlier arithmetic and must not be treated as a performance guarantee for the fix.

This source copy contains the earlier benchmark/factory implementation. The newer arm-aware serving integration and active fidelity fix are not included here. It is not a turnkey production worker.

## What it is

DeepSeek supplies a draft model with stages `mtp.0–2`, Markov and confidence heads, and 2,401 draft tensors. Routed experts use MXFP4; dense matrices use FP8 E4M3 with E8M0 32×32 block scales. The model architecture shares embedding and vocabulary head with the target. **The measured native arms here retain the deployed EXL3 K6 draft head**; native expert/dense weights do not mean every tensor uses the official head representation. The target weights, sampler and verification remain unchanged. In the arm table, M and T denote row counts for dense and expert kernel shapes.

SM86 has no FP8/FP4 tensor-core instructions. These kernels decode the released values into BF16 operands and accumulate in FP32. They preserve activation quantize/dequantize operations, FP32 routers/biases and declared rounding boundaries. Full arithmetic fidelity is separately gated, especially for attention and head computation.

| Arm | Routed experts | Dense operations | Purpose |
|---|---|---|---|
| `native-packed` | MXFP4 decoded inside GEMM | Packed FP8 | Small packed reference |
| `native-expanded` | Exactly decoded BF16 | Exactly decoded BF16 | Larger representation control |
| `native-hybrid-kx` | Register-fed MXFP4 | BF16 | Earlier native comparison arm |
| `native-hybrid-kx2` | Fused MXFP4 MoE, static shape dispatch | Selected FP8 at M5/M10/M15 and main projection M8; BF16 elsewhere | Fastest measured historical arm; paired BF16 shared projection at T20 |

## Why

Consumer Ampere GPUs cannot execute the model's native low-bit operations directly. Decoding values inside small kernels reduces memory traffic while retaining known source values. This makes native-format drafting practical enough to study against the EXL3 control. Precision alone has not demonstrated an acceptance gain on the measured target.

## When to use it

Use it for kernel research, format decoding, numerical comparisons and controlled native-drafter experiments on an isolated NVLink pair. Keep the K6 head label, exact weight provenance and full fidelity gate visible in any result.

## When not to use it

Do not replace a production drafter with these arms yet. Component agreement, graph/eager equality and equal acceptance do not establish whole-drafter fidelity. Do not extrapolate solo-shard timings to complete remote rounds or engine token rates.

## Requirements

- Two RTX 3090s with an NV4 bridge and peer access for physical TP2. Solo emulation uses local collective stand-ins and has different capacity constraints.
- Native packed/hybrid arms need more memory than the EXL3 control; the measured pair footprints are below. The expanded arm requires substantially more memory and cannot fit full two-rank emulation on one card.
- Historical kernel environment: PyTorch 2.10.0+cu126, CUDA 12.6, Triton 3.6 and an SM86 build. External CUTLASS headers are pinned in the grouped-kernel donor lock. The control and K6 head need the matching exllamav3 extension and checkpoint metadata.
- Official shards and a compatible local EXL3 control package, supplied separately. The fetch scripts pin and verify official shard hashes. No model weights or capture fixtures ship here.
- Configure the sample GPU admission checks, data roots and lock locations for your own hardware before GPU use. Publication placeholders intentionally do not identify working devices.

## Measured effect

All measurements below are historical, before the active fidelity correction. They concern the draft worker, with the public [EXL3 2.75 bpw target](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark) used for captured acceptance. B denotes batch size; it is not an end-to-end concurrency throughput test. Context consists of benchmark windows or captured prefixes as specified.

Physical-pair tests ran on 2026-10-23 UTC with two NVLinked RTX 3090s at 325 W/card, synthetic benchmark windows, k = 5 and B1–B4. Each arm ran once, collecting 200 CUDA events per case after 0.5 s settling. Values are draft-step p50 in µs; output counts are shown per arm:

| arm | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| deployed EXL3 drafter, TP2, n_out 5 | 2,028 | 2,512 | 2,896 | 3,722 |
| native-hybrid-kx, n_out 5 | 2,246 | 2,804 | 3,151 | 3,737 |
| **native-hybrid-kx2, n_out 5** | **1,782 (−12 %)** | **2,457 (−2 %)** | **2,821 (−3 %)** | **3,447 (−7 %)** |
| deployed EXL3 drafter, TP2, n_out 4 | 2,000 | 2,443 | 2,820 | 3,280 |
| native-hybrid-kx2, n_out 4 | 1,825 (−8.7 %) | 2,442 (−0.1 %) | 2,812 (−0.3 %) | 3,099 (−5.5 %) |

Solo tests ran on 2026-10-23 UTC on one RTX 3090 at 325 W, using TP2 emulation with local collective stand-ins, synthetic windows, B1–B4 and n_out 5. Each cell is the median of 3 interleaved run p50s, each with 200 events after 0.5 s settling, in µs. These do not measure NVLink collectives:

| rank | arm | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| 0 | deployed EXL3 | 1,734.7 | 2,225.2 | 2,503.7 | 3,298.8 |
| 0 | native-hybrid-kx2 | 1,650.7 (−4.8 %) | 2,070.5 (−7.0 %) | 2,347.0 (−6.3 %) | 3,076.1 (−6.8 %) |
| 1 | deployed EXL3 | 1,815.6 | 2,317.8 | 2,658.2 | 3,365.3 |
| 1 | native-hybrid-kx2 | 1,753.0 (−3.5 %) | 2,184.8 (−5.7 %) | 2,380.4 (−10.5 %) | 3,176.0 (−5.6 %) |

An independent interleaved solo repetition reproduced the direction of the speed effect, but not every original three-run range. Those observed ranges are not confidence intervals. The complete pair comparison is a single run per arm.

Component checks ran on 2026-10-23 UTC on an RTX 3090 with full native weights and TP2 emulation. The complete arithmetic audits cover both ranks, stages 0–2 and T5/T10/T15/T20; each listed check ran as one suite or audit, without a long-context serve. Exceptions are stated below.

| Numerical check | Historical result | Gate | Scope exception |
|---|---|---|---|
| Focused component/integration suite | 171 passed | All pass | — |
| Graph vs eager, B1–B4 × n_out 4/5 | 8 / 8 exact IDs and confidence; ranks agree | Exact | — |
| Expert projection vs prior native oracle | 12 BF16 differences in 2,073,600 values; worst 99.9942 % agreement; all candidate values match FP64 rounding | ≥ 99.99 % | — |
| Shared/post-TP epilogues vs prior operations | 48 / 48 bit-exact, including signed zero | Exact against prior operations | — |
| Routed output vs FP64 | Worst relative norm 3.3871e-8 | ≤ 1e-6 | — |
| Selected dense output vs prior packed FP8 | Worst 99.9072 % BF16 agreement | ≥ 99.9 % | — |
| Required FP32 dense partials vs FP64 | Worst relative norm 7.7068e-8 | ≤ 1e-6 | — |
| Inherited single-pass vs official online attention | 38,799 / 81,920 BF16 outputs differ; relative norm 0.00299785 | Demonstrates a policy difference | Independent CPU synthetic case with identical inputs and FP64 dots; no inference context/concurrency |

The FP32 bounds are normwise. They do not imply a universal pointwise relative bound near cancellation. Epilogue bit identity is against the prior native operations, including their signed-zero behavior, rather than a claim of universal literal-official bits. These component gates do not supersede the failed strict fidelity gate.

Acceptance uses a 2026-10-22 production capture replayed on 2026-10-23: the public target above, temperature 1, 453 rounds and 538 determinable live blocks, with the captured context/batch mix. Each worker score comes from one offline replay on the physical RTX 3090 pair.

| Drafter | Accepted per round |
|---|---:|
| Engine drafter during capture | 1.810 |
| EXL3 control, replayed | 1.801 |
| `native-hybrid-kx` | 1.823 |
| `native-hybrid-kx2` | 1.807 |

Acceptance is equal within noise, with standard errors ~0.02–0.03. This is an offline committed-continuation estimate, not a live native-engine A/B or an acceptance gain. Capture receipts contain 542 live blocks; the report's determinable scoring subset contains 538. Different draft representations can change individual blocks while preserving the average acceptance rate.

Memory tests ran on 2026-10-23 UTC on the physical NVLink RTX 3090 pair at 325 W/card, with one load per arm and B1–B4 benchmark graphs over synthetic windows. Values are card A / card B. Load peaks are allocated memory; serving footprints are observed nvidia-smi readings. Load times are rounded:

| arm | load time | load peak (allocated) | serving (nvidia-smi) |
|---|---|---|---|
| deployed EXL3 drafter | 61 s | 4,604 / 4,598 MiB | 5,214 / 5,004 MiB |
| native-hybrid-kx | 332 s | 6,833 / 6,827 MiB | 7,584 / 7,540 MiB |
| **native-hybrid-kx2** | **330 s** | **7,348 / 7,344 MiB** | **8,090 / 7,926 MiB** |

Warm-cache native admission/prepacking took ~160–170 s per rank in separate solo runs. This is a different load condition from the pair table.

The following development screens ran on 2026-10-23 UTC on a solo RTX 3090 at 325 W. They compare kernel shapes, without an inference context or concurrent serve.

| Kernel selection finding | Historical effect | Shape / repeats |
|---|---|---|
| Selected FP8 dense and fused epilogues vs prior native arm | About 300 µs dense work and about 140 µs epilogue work saved per rank | B1, n_out 5; repeated profile comparison, 3 runs |
| Alternative expert mode at T15 | About 250 µs vs 216 µs; slower | Rank 0 stage-specific shape screen; repeat count not reported |
| FP8 main projection at M128 | About 563 µs vs 268 µs BF16; slower | Same-shard screen; synthetic rows; repeat count not reported |

Register-tile/split sweeps did not beat the delivered dispatch. FP8 is selected only where measured useful. No unfinished corrected timing results are included.

## Upstream claims

[DeepSeek's official inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/model.py) and kernels define the model maker's draft arithmetic and native formats. They are the reference, not an upstream claim that this SM86 implementation is equivalent or faster. HeNryous supplies the quantized image lineage; its results are not mixed into our tables.

## Deploy

This is a research setup recipe. It does not enable native production serving.

Keep the `offload/` package level. From the standalone repository root, inspect [NATIVE.md](code/offload/dspark/native/NATIVE.md), [the tensor manifest](code/offload/dspark/native/NATIVE_MANIFEST.json) and the kernel READMEs first. The arithmetic narrative in NATIVE.md predates the failed full-fidelity review; the status on this page controls qualification.

Supply official draft shards, shared embedding/head shards, tokenizer/control data and donor dependencies in the paths expected by your configured local copy. The hash-checking [draft fetch script](code/offload/dspark/native/fetch_official.sh) and [shared fetch script](code/offload/dspark/native/fetch_official_shared.sh) identify what is needed. Their download directories are fixed examples; configure them before running.

After configuring GPU checks and build/data paths for an isolated pair, the benchmark module accepts:

```bash
cd code
PYTHONPATH=. python -m offload.dspark.native.tools.native_bench \
  native-hybrid-kx2 pair /data/results/native-pair.json 0 200
PYTHONPATH=. python -m offload.dspark.native.tools.native_graphcheck \
  native-hybrid-kx2 pair /data/results/native-graph.json
```

Use `native_parity` only with your own compatible capture fixtures and external verifier. Do not treat graph equality as fidelity. The staged worker server still constructs the quantized control factory; native arm-aware identity/HELLO integration is absent in this copy. There is no supported native service-switch command in this release.

## Rollback

For research, stop the experiment and choose the control `k2-tp2` arm in the benchmark factory. For a future serving integration, restore the previously qualified EXL3 worker, its matching identity, source/binary set and engine configuration, then recheck HELLO and a smoke request. Do not silently substitute a different arm under the same identity. No native production switch has been made here.

## Limits and known issues

- Strict attention fidelity remains unresolved. The latest fix changes attention and FP32 head/Markov arithmetic; its final performance and broader qualification remain pending.
- The earlier worker used a single-pass softmax with the sink inside its maximum. Official attention processes online key blocks, rounds probabilities per block and adds the sink last. This explains a policy difference but is not yet a complete diagnosis of the residual gate failure.
- The latest corrected pilots remove the head self-check flags and pass the KL band, but fail attention. An official GPU component control also fails that band; no reference relaxation is assumed.
- Native-format expert/dense values, the retained K6 head representation and whole-drafter arithmetic fidelity are separate facts.
- Physical-pair worker benchmarks are not complete in-engine native serving, remote-round cost or target-throughput qualification.
- The standalone tree retains fixed paths, GPU admission placeholders and deployment-specific locks. It lacks the transport package and newer serving/fidelity modules. Configure external dependencies and review the integration before running GPU tools.
- Fused-MoE component latency targets were only partly met, even though the integrated historical arm is faster. Expanded weights impose a much larger memory cost.

## Repository layout

| Path | Purpose |
|---|---|
| [code/offload/dspark/native](code/offload/dspark/native) | Native format decoders, reference arithmetic, loaders and TP2 factory |
| [native tools](code/offload/dspark/native/tools) | Solo/pair benchmarks, graph equality and captured replay |
| [aggregate evidence](code/offload/dspark/native/tools/n2/evidence) | Component qualification and repeated benchmark summaries |
| [MXFP4 kernels](code/offload/dspark/kernels/mxfp4) | Register-fed packed experts and decoder reference |
| [grouped kernels](code/offload/dspark/kernels/grouped) | Expanded BF16 experts, dense/head alternatives, CUTLASS adapter |
| [fused MoE](code/offload/dspark/kernels/fusedmoe) | MXFP4 expert pipeline, shape dispatch and exact epilogues |
| [FP8 dense](code/offload/dspark/kernels/fp8dense) | Byte-preserving prepack, selected small-row dense kernels |
| [worker](code/offload/dspark/worker) | Shared control model, protocol, TP2/peer machinery and EXL3 adapter |
| [CONTRACT.md](code/offload/dspark/CONTRACT.md) | Worker/target data boundary and protocol assumptions |

## Tests

Run the CPU format/native suite from `code/`, with CUDA hidden and cache writes disabled in the source tree:

```bash
cd code
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python -m pytest -q -p no:cacheprovider offload/dspark/native/tests
```

The independent documentation check reran the command above: **5 passed, 10 skipped, 3 warnings**. The warnings concern deprecated output resizing in the format decoder under the installed PyTorch. Weight-shard and GPU tests skip without their prerequisites. The earlier publication reported **11 passed, 17 skipped** for its standalone CPU checks; it did not specify the identical command/scope, so that count is retained separately. The focused full GPU/component suite recorded **171 passed** and graph/eager **8 / 8**; those are separate suites and are not reproduced by the CPU command. They do not establish fidelity. Use each kernel's qualification script only in an isolated, configured GPU test environment.

## Credits and licences

PLI Labs additions are Apache-2.0. DeepSeek supplies the model and arithmetic reference under MIT. HeNryous/vLLM supply the Apache-2.0 engine/plugin lineage. The MXFP4, fused-MoE and FP8-dense register mapping derives from [ro99/strata](https://github.com/ro99/strata), Apache-2.0. All three kernel directories retain NOTICE/LICENSE.strata files.

The grouped adapter uses external [CUTLASS](https://github.com/NVIDIA/cutlass), BSD-3-Clause. The EXL3 runtime and included accumulator patch target [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3), MIT. No full donor engine is bundled. [NOTICE](NOTICE) inventories the included derivatives and patch, with full licence texts in [LICENSES](LICENSES).

## Related

| Entry | GitHub | Hugging Face |
|---|---|---|
| Async host-flag handoff | [Source](https://github.com/ProprietaryLegal/dsv41-async-hostflag-handoff) | [Card](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) |
| Remote RTX 3090 drafter | [Source](https://github.com/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) | [Card](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) |
| NVLink drafter | [Source](https://github.com/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) | [Card](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) |
| Remote Engram | [Source](https://github.com/ProprietaryLegal/dsv41-remote-engram-rdma) | [Card](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma) |
| ofl transport | [Source](https://github.com/ProprietaryLegal/ofl-rdma-transport) | [Card](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) |
| Benchmark tools | [Source](https://github.com/ProprietaryLegal/dsv41-longctx-bench-tools) | [Card](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools) |
| Exact acceptance ceiling | [Source](https://github.com/ProprietaryLegal/dsv41-exact-acceptance-ceiling) | [Card](https://huggingface.co/ProprietaryLegal/dsv41-exact-acceptance-ceiling) |
| Full stack overview | [Source](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) | [Card](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) |
| Offload findings | [Source](https://github.com/ProprietaryLegal/dsv41-spark-offload-findings) | [Card](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings) |

Upstream forks: [HeNryous lineage](https://github.com/ProprietaryLegal/deepseek-v41-1017-dgx-spark) · [sudoingX lineage](https://github.com/ProprietaryLegal/dgx-spark-deepseek-v41).

[Collection](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e) · [Hub](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload).

<!-- pli:footer:start -->

---

## About PLI Labs

“Your proprietary work product is your proprietary intelligence.” PLI Labs is the open research arm of Proprietary Legal Intelligence. We publish legal model prunes, serving recipes for owned hardware, and open-source tools so readers can inspect our methods and test our claims before hiring us.

Founded by a practicing trial lawyer, PLI builds AI systems around a legal organization's own work product: retrieval, staged workflows, validation, and attorney review. The approach grew from systems running on live matters in a litigation practice. Our consulting work supports deployment through approved frontier APIs, private cloud, or open-weight models on hardware you own. Explore the research or contact us about your organization's work.

[Website](https://proprietarylegal.ai) · [PLI Labs](https://proprietarylegal.ai/labs.html) · [Research](https://proprietarylegal.ai/research/index.html) · [GitHub](https://github.com/ProprietaryLegal) · [Hugging Face](https://huggingface.co/ProprietaryLegal) · [Contact](https://proprietarylegal.ai/contact.html)

Proprietary Legal Intelligence LLC is a technology consultancy, not a law firm, and does not provide legal services or legal advice.

<!-- pli:footer:end -->
