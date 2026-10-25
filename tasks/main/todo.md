# hip-tile todo (branch: main)

Plan: `~/.claude/plans/crystalline-dreaming-lightning.md`. Run the LegoGears YOLOv4-tiny Darknet model on AMD gfx1031 in pure Rust, using hip-tile (cuTile-style tile kernels).

## Contents

- [Phase A: hip-tile](#phase-a-hip-tile)
- [Phase B: LegoGears YOLO](#phase-b-legogears-yolo)
- [Review](#review)

## Phase A: hip-tile

- [x] Bootstrap: git, toolchain pin, workspace skeleton, task files
- [x] Spike: tile fragment typing, dynamic LDS, ISA check (1347.7 GFLOP/s)
- [x] `hip-sys` + `hip-rt`, `hip-tile-build`, `#[kernel]` macro, runner + saxpy (verified in scratch copy)
- [x] `hip-tile` core ops: load/store, map, mma, transpose, reduce (tile GEMM ~1475 GFLOP/s)
- [x] Step 0: generalized thread layout, `kernels/src/{copy,transpose,gemm}.rs`, real-repo build + tests + clippy, lessons

## Phase B: LegoGears YOLO

- [x] 1. `crates/hip_darknet` (cargo package `hip_darknet`, lib + bin): cfg, weights + BN fold, shapes, CPU reference forward, yolo decode + NMS, image; Rust unit tests + ground-truth label check (no Darknet binary)
- [x] 2. GPU kernels: conv (implicit GEMM + fused bias/leaky), maxpool, upsample, concat; per-op parity
- [ ] 3. GPU executor in `hip_darknet` (`--gpu`): per-layer parity, detections equal to the CPU path
- [ ] 4. App polish: annotated PNG, `--bench`, README results
- [ ] 5. Optional: fused route stores, per-layer conv tiles, f16

## Review

(filled in at the end)
