# hip-tile

Pure-Rust tile kernels for AMD GPUs (cuTile-style), plus a Darknet YOLO runner built on them. Tested on the Radeon 680M iGPU (gfx1031).

## Contents

- [What it is](#what-it-is)
- [Crates](#crates)
- [Running](#running)
  - [runner](#runner)
  - [hip_darknet](#hip_darknet)
- [Results](#results)

## What it is

Device code is ordinary Rust compiled for the `amdgcn` target (pinned nightly, see `rust-toolchain.toml`). `hip-tile` provides tile types and ops (load/store, map, mma, transpose, reduce) and a `#[kernel]` macro; `hip-rt` loads the resulting code object through the HIP runtime. No C++, no hand-written assembly.

## Crates

| Path | Role |
|---|---|
| `crates/hip-sys` | raw HIP / rocBLAS FFI |
| `crates/hip-rt` | safe host runtime: device, module, stream, event, device buffers |
| `crates/hip-tile` | tile abstraction used inside kernels |
| `crates/hip-tile-macros` | `#[kernel]` proc macro (device entry + host launcher) |
| `crates/hip-tile-build` | `build.rs` helper: compiles the kernels crate for gfx1031, sets the ROCm rpath |
| `kernels` | gemm, copy, transpose, saxpy, and the YOLO ops (implicit-GEMM conv, maxpool, upsample, concat) |
| `apps/runner` | CLI / benches / tests for the kernels |
| `crates/hip_darknet` | Darknet `.cfg`/`.weights` loader, CPU reference, YOLO decode + NMS, GPU executor |

## Running

Wrap builds in a memory cap and keep jobs at 4: `systemd-run --user --scope -q -p MemoryHigh=3G cargo ... -j 4`.

### runner

```sh
cargo run -p runner --release -- gemm --n 1024 --bench   # gemm vs rocBLAS
cargo test -p runner --release               # kernel parity tests
```

### hip_darknet

```sh
D=/path/to/LegoGears_v2
cargo run -p hip_darknet --release -- --gpu \
  --cfg $D/LegoGears_v2.cfg --weights $D/LegoGears_v2_last.weights --names $D/LegoGears.names \
  --out out/ image1.png image2.png
cargo run -p hip_darknet --release -- --gpu --bench 200 ... image.png   # ms/frame
cargo test -p hip_darknet --release                                     # CPU + GPU parity
cargo test -p hip_darknet --release --no-default-features               # CPU only
```

Drop `--gpu` for the CPU path. The `gpu` feature is on by default.

## Results

Radeon 680M (gfx1031), f32. The iGPU clock shares a power budget with the CPU, so numbers are best-of-N with an idle CPU.

| Benchmark | Result |
|---|---|
| Tile GEMM 1024^3 | 2.05 TFLOP/s (rocBLAS 2.69; 0.76x) |
| Tile GEMM 2048^3 | 2.28 TFLOP/s (rocBLAS 1.91; 1.19x) |
| Conv (implicit GEMM), real YOLO layers | 250-970 GFLOP/s (56x40 maps best, 7x5 maps worst) |
| LegoGears YOLOv4-tiny, CPU forward (1 thread, reference) | 445 ms/frame |
| LegoGears, GPU forward (upload + kernels + sync) | 4.9 ms/frame (about 90x) |
| LegoGears, GPU end to end (resize, upload, download, decode) | 5.5 ms/frame (about 81x) |
| GPU vs CPU per-layer max relative error | 1.7e-4 (layer 32), all layers under 1e-3 |
| GPU vs CPU detections, 10 labelled images | identical (count, class, IoU >= 0.99, conf within 1e-3) |
| Ground truth, 10 validation images (38 boxes), IoU 0.5 | recall 1.000, precision 1.000, 0 class errors |
