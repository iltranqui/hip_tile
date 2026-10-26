---
status: review
stage: workflows/train-darknet-model/03_gpu-backward
date: 2026-10-26
---

# legogears_v2 — GPU backward (first slice)

Contract: [`workflows/train-darknet-model/03_gpu-backward/CONTEXT.md`](../../../03_gpu-backward/CONTEXT.md). This file covers the kernels that are written and GPU-verified; the convolution gradients and the `[yolo]` delta are the next slice (see [Remaining](#remaining)).

## Contents

- [New tile op](#new-tile-op)
- [Op to kernel map](#op-to-kernel-map)
- [Parity](#parity)
- [Remaining](#remaining)

## New tile op

| Op | Why it cannot live in a kernel | Location |
|---|---|---|
| `store_tile_add` | gradients fan in from several layers/blocks, so the store must accumulate; `store_tile` overwrites by design | `crates/hip-tile/src/ops/memory.rs`, its own `impl<T: Element + AddAssign>` block so `load_tile`/`store_tile` stay bound-free |

## Op to kernel map

| Op | Kernel | Batched? | Notes |
|---|---|---|---|
| fill | `backward::util::zero` | flat | one element per thread |
| accumulate | `backward::util::{axpy_acc, acc}` | flat | `y += a*x`, `y += x` |
| SGD step | `backward::util::sgd_update` | flat | `u += -decay·batch·w; w += (lr/batch)·u; u *= momentum`; `decay = 0` for biases/scales |
| rolling BN stats | `backward::util::running_update` | flat | `0.9·old + 0.1·new` |
| maxpool gradient | `backward::maxpool_bwd` | per image | **gather**: each input element examines the ≤ `size²` windows containing it and takes the delta only if that window selected it — race-free, no argmax buffer |
| upsample gradient | `backward::upsample_bwd` | per image | **gather**: each input sums its `stride²` upstream gradients |
| route/concat gradient | `backward::route_bwd` | per image | `dst += src` on a channel-block view (`group_id` handled by the caller's view) |
| BN batch statistics | `backward::bn_stats` | yes | one block per channel; `[batch x spatial]` view, unbiased variance (`/(n-1)`) |
| BN backward | `backward::bn_bwd` | yes | one block per channel, two passes; dscale + dx in the Darknet form |

Device-side `f32` has no `sqrt`/`powf`: `bn_bwd` uses `core::intrinsics::sqrt::<f32>` (the device build now enables `core_intrinsics`) and computes `(v+ε)^1.5` as `(v+ε)·√(v+ε)`.

## Parity

`apps/runner/tests/backward_ops.rs`, GPU vs CPU (f64 accumulation where the math is exact, the ported Darknet formula for BN):

| Kernel | Cases | Tolerance | Result |
|---|---|---|---|
| `util` (zero/axpy/acc/sgd/running) | 1000 elements, decay and no-decay forms | 1e-5 rel | pass |
| `maxpool_bwd` | 3×10×12 s2; 2×9×7 s2; **8×8 size 3 stride 2** (overlapping windows); 5×5 size 3 stride 1 | 1e-5 | pass |
| `upsample_bwd` | 3×4×5 stride 2 | 1e-6 | pass |
| `route_bwd` | 6-channel dst, group 1 of 2, poisoned neighbours | exact | pass |
| `bn_stats` | 3×4×40; 1×2×7 (batch of one ⇒ var 0); 9×3×300 (ragged tail) | 1e-5 / 1e-4 | pass |
| `bn_bwd` | 3×2×24, non-unit scales, `x_norm` from the forward definition | 2e-4 dx, 1e-4 dscale | pass |

Every gradient kernel accumulates into its destination rather than overwriting it, and the tests poison the destination to prove it — that is what lets several branches feed one buffer (the Darknet ordering quirk documented in `_shared/darknet-semantics.md#gradient-routing`).

Three real bugs were caught by these tests, which is the argument for writing them first: the flat kernels used the wrong grid axis (`grid` is `(x, y)` and `block_id()` is `(y, x)`), the maxpool gather indexed the input without its channel offset, and `bn_stats`/`bn_bwd` treated the channel stride as 1 instead of `spatial`.

## Remaining

- `conv_bwd_dw`: needs the batched im2col gather plus an accumulating GEMM (a `hip-tile` op), then one block per output tile sweeping the batch.
- `conv_bwd_dx`: transpose-convolution gather (no scatter), with weights prepacked per `(ky, kx)`.
- `yolo_delta`: the `[yolo]` delta tensor on device, against `train::deltas::yolo_delta` as the oracle.
- Batched forward (`conv`/`maxpool`/`upsample`/`concat` at `batch > 1`) and `TrainModel` wiring (stage 04).

## Test output

```
cargo test -p runner --release       34 passed (6 backward_ops, 5 gemm, 5 memory, 1 rocblas, 3 saxpy, 14 yolo_ops)
cargo clippy --workspace --all-targets   0 warnings
```
