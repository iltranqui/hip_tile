# ROADMAP — training `hip_darknet` end to end

The goal: train a Darknet YOLO model **from scratch** on a labelled dataset and prove the result, entirely in Rust on the AMD GPU — no C/C++, no `hipcc`, no Darknet binary. The first target is `LegoGears_v2` (5 classes, 224×160, YOLOv4-tiny-shaped cfg).

Implementation lives in `crates/hip_darknet` (grown, not a new crate) with every new GPU primitive as a tile op in `crates/hip-tile`. Process lives in [`workflows/train-darknet-model/`](workflows/train-darknet-model/CONTEXT.md); the current run is `workflows/train-darknet-model/runs/legogears_v2/`.

## Contents

- [Acceptance](#acceptance)
- [Decisions](#decisions)
- [Stages](#stages)
- [Status](#status)
- [Risks](#risks)

## Acceptance

Measured on the 80-image validation split of `~/src/DarkSuite/datasets/LegoGears_v2/`, compared against that dataset's own recorded training history (`LegoGears_v2_loss.csv`, `LegoGears_v2_map.csv`):

| Gate | Bar |
|---|---|
| mAP@0.5 (final) | ≥ 0.99 (reference 0.9955) |
| mAP@0.5:0.95 | recorded and stable (reference only logs 0.5) |
| Loss trajectory | same band as the reference at equal images-seen |
| Learning-rate curve | matches the reference csv by construction (hard check of the schedule port) |
| Stability | no NaN/inf in loss, weights or deltas across the run |
| Reproducibility | fixed seed, command line recorded, resume produces the same weights |
| Round trip | the produced `.weights` loads through the inference path and passes the ground-truth detection test |
| Performance | measured per `_shared/benchmarking.md`, GPU faster than the CPU reference trainer by a stated factor |

## Decisions

Recorded because they constrain every stage; change them here, not in code.

| Decision | Value | Why |
|---|---|---|
| Init | from scratch | reproduces the reference run's starting point |
| Batch | 4 | user decision; comfortable memory on the 680M iGPU at 224×160 |
| Iterations | 12000 with `steps=9600,10800`, `scales=0.2,0.1`, `burn_in=1000`, `max_batches=12000` | matches the reference's 48 000 images seen and the same LR fractions |
| Optimizer | SGD with momentum only | user decision; no Adam, no per-parameter adaptive anything |
| Precision | f32 end to end | user decision; no f64 master weights, no mixed precision, no f64 accumulation |
| Checkpoint format | Darknet `.weights`, BN unfolded on save | loadable by DarkSuite and by our own loader |
| Resume format | our own, documented | Darknet's `optimizer.state` layout is unpublished; no compatibility claimed |
| Augmentation | only what this cfg enables: crop jitter + resize, hue/saturation/exposure, and flip (off by default) | mosaic/mixup/cutmix/angle are rejected, never silently ignored |
| Ops | new primitives in `crates/hip-tile`, kernels stay thin | user decision; keeps `kernels/` declarative |
| Loss comparison | per-image loss column, never raw `loss` across batch sizes | Darknet's cost is a batch-wide sum |
| Verification | f32 CPU reference, two independent CIoU derivations, and finite differences at a stated tolerance | no f64 oracle exists under the f32 decision |

## Stages

| # | Stage | Gate before moving on | Contract |
|---|---|---|---|
| 01 | training semantics | facts cited from Darknet's C++, loss scaling pinned | [`01_train-semantics/`](workflows/train-darknet-model/01_train-semantics/CONTEXT.md) |
| 02 | CPU backward (f32) | CIoU two-formula agreement + finite differences + analytic identities, all at stated tolerances | [`02_cpu-backward/`](workflows/train-darknet-model/02_cpu-backward/CONTEXT.md) |
| 03 | tile ops + GPU kernels | GPU vs CPU parity per op/kernel, FD where it applies, batch-1 inference suites unchanged | [`03_gpu-backward/`](workflows/train-darknet-model/03_gpu-backward/CONTEXT.md) |
| 04 | GPU training step | GPU gradients == CPU gradients, loss match, timing probe accepted | [`04_executor/`](workflows/train-darknet-model/04_executor/CONTEXT.md) |
| 05 | CLI, checkpoints, metrics | weights round trip, csv schema matches the reference, resume equality | [`05_cli-checkpoints/`](workflows/train-darknet-model/05_cli-checkpoints/CONTEXT.md) |
| 06 | the run | the acceptance table above | [`06_run/`](workflows/train-darknet-model/06_run/CONTEXT.md) |
| 07 | bench + docs | numbers per method, every CONTEXT.md true again | [`07_bench-docs/`](workflows/train-darknet-model/07_bench-docs/CONTEXT.md) |

Nothing moves to the next stage until a person has read the previous stage's run file.

## Status

| Stage | State | File |
|---|---|---|
| 01 training semantics | written, awaiting the human read | `runs/legogears_v2/01_train-semantics.md` |
| 02 CPU backward | **done** — 11 numeric gradient checks + 49 unit tests + learning check pass | `runs/legogears_v2/02_cpu-backward.md` |
| 03 tile ops + kernels | **done except one open case** — `store_tile_add` + 12 kernels GPU-verified (util, maxpool/upsample/route gradients, BN stats/backward, `[yolo]` delta for the CIoU path); `conv_bwd_dw/dx` verified at batch 1 and 2 (batched `dX` bug fixed); the device MSE box path has a 4-entry mismatch, unused by this cfg | `runs/legogears_v2/03_gpu-backward.md` |
| 04 GPU training step | not started — unblocked (needs batched forward + `TrainModel`) | — |
| 05 CLI, checkpoints, metrics | **done** except split-run resume (needs a long run) — the CPU-side of it | `runs/legogears_v2/05_cli-checkpoints.md` |
| 06 the run | blocked on 03/04 for the 12 000-iteration budget; dataset caveats recorded in stage 05 | — |
| 07 bench + docs | not started | — |

Working today: `hip_darknet train --data …/LegoGears_v2.data --iters N --batch N --out DIR` trains on the CPU with Darknet-faithful deltas, writes Darknet-format `.weights` and reference-compatible `loss.csv`/`map.csv`, and can resume. The CPU path is the reference implementation for the GPU trainer; at ~5.5 s/image it is not the run owner.

Two dataset facts that change stage 06 (recorded in `runs/legogears_v2/05_cli-checkpoints.md`): the train and valid lists are the same 80 images, and the only shipped weights file (`…_last.weights`) produces no detections, so there is no trustworthy checkpoint to score against.

## Risks

| Risk | Why it bites | Mitigation |
|---|---|---|
| CIoU delta in f32 | likeliest way to "train but worse" | two independent derivations must agree, FD at a stated f32 tolerance, GPU-vs-CPU exactness — all before any long run |
| f32 momentum drift over 12000 iterations | no f64 master copy | `max|w|` and a weight-norm checksum per checkpoint, so drift shows as a discontinuity rather than a mystery |
| Batch-4 batch norm statistics | noisier than the reference's batch 16 | documented as an expected shift; fallback is 2-step accumulation to effective batch 8, recorded as a decision |
| No atomics/accumulate primitive in `hip-tile` | scatter-based `dX`/`dW` would need them | gather-`dX` and one-block-per-tile `dW`; block-partial + reduction pass is the deterministic alternative if a spec needs it |
| Batched forward touching inference kernels | could regress the passing parity suites | batch-1 path kept byte-identical, suites re-run unchanged in stage 03 |
| Wall clock on 12 CUs | ~48 000 forward-equivalents plus backward | 200-iteration timing probe at stage 04 precedes the full run |
| Loss-scale confusion across batch sizes | would make stage 06 unreadable | per-image loss column decided in stage 01 |
