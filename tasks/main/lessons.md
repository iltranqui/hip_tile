# hip-tile lessons (branch: main)

Rules learned while building this project.

## Contents

- [Environment](#environment)
- [Decisions](#decisions)

## Environment

- The AMD GPU is the Radeon 680M iGPU; the HIP runtime reports it as gfx1031. Build device code with `-Ctarget-cpu=gfx1031`.
- The amdgcn target requires an explicit `-C target-cpu`, or `core` fails to build.
- The rustc amdgcn output is `<crate>.elf` (not `.so`), loadable directly by `hipModuleLoad`/`hipModuleLoadData`.
- `sudo` cannot run through the `!` prompt prefix: give the user commands to run in their own terminal.
- Don't run `find /` on WSL: it walks `/mnt/c` and hangs. Search `/opt/rocm`, `/usr` only.

## Corrections

- The YOLO work is a direct Rust port: never invoke `/usr/bin/darknet` or `darknet_*` binaries for references or tests. Darknet's C++ source may be read to port semantics; verify with Rust unit tests and the dataset's ground-truth label files.

## Decisions

Spike, 2026-10-25 (`scratchpad/spike`): 64×64×16 block tile, 256 threads, 4×4 accumulator per thread, padded LDS operands.
Result at N=1024: 1347.7 GFLOP/s, max relative error 2.3e-6, 57 VGPRs, no scratch spills.

- Tile fragment typing: use `generic_const_exprs` (`[f32; M / 16 * (N / 16)]` with `where [(); ...]:` bounds). It compiles cleanly; the `TileShape` macro-trait fallback is not needed.
- LDS: use `core::intrinsics::gpu::gpu_launch_sized_workgroup_mem::<T>()` (features `gpu_intrinsics`, `gpu_launch_sized_workgroup_mem`); pass the byte size as `sharedMemBytes` at launch.
- FMA: `core::intrinsics::fma(a, b, c)` (generic over float); `fmaf32` no longer exists.
- Barrier: `fence(Release)` + `core::arch::amdgpu::s_barrier()` + `fence(Acquire)`. The fence is system-scope and emits `buffer_gl0_inv`/`buffer_gl1_inv`; acceptable for now, revisit for performance.

Device side, 2026-10-25:

- `generic_const_exprs` is not usable: it ICEs under the next solver (`rustc_infer` `infer/mod.rs:1694`, unreachable) and overflows under `-Znext-solver=no`. Replaced by a fixed register array `[T; FRAG_MAX = 64]` per tile; only `frag_len(M, N)` slots are used, loops unroll, SROA removes the rest (no spills, no scratch). No `where [(); ..]:` bounds anywhere. Tiles above 64 elements per thread (e.g. 256x128) fail a const assert.
- `#[derive(Clone, Copy)]` on `Tensor<T>` adds a `T: Copy` bound that breaks `KernelArg`; use manual impls (done in `tensor.rs`).
- `core::intrinsics::abort` no longer exists on this nightly: use `core::process::abort_immediate()` with `#![feature(abort_immediate)]`.
- Thread layout rule: `TC` = largest power of two <= `256 / min(M, 16)` that divides `N`; `TR = 256 / TC`; `M % TR == 0`. The layout depends on `(M, N)`, so BK=8 operand tiles (64x8, 8x64) work. Checked 64x16, 16x64, 64x64, 64x8, 8x64, 32x32, 1x256, 64x32, 32x64 (copy and transpose, ragged edges): all exact.
- Barrier: `asm!("s_waitcnt lgkmcnt(0)", "s_barrier")` (feature `asm_experimental_arch`) replaces `fence` + `s_barrier` + `fence`. Valid because `mma`/`transpose`/`reduce` only exchange data through LDS. 128x128x16 gemm: 1941 -> 2040 GFLOP/s. Any future op that exchanges data through global memory needs a real fence.
  - Reverted 2026-10-25: the repo CLAUDE.md rule forbids hand-written AMDGCN inline asm in kernels. `sync_block` is back to `fence` + `s_barrier` + `fence` (pure Rust); accept the ~5% gemm cost. Never use `asm!` in hip-tile or kernels.
- gemm at N=1024 (f32, gfx1031), block tile BMxBNxBK, same correctness (max rel err 1.75e-6):

  | config | GFLOP/s | VGPRs |
  |---|---|---|
  | spike 64x64x16 | 1355 | 57 |
  | tile API 64x64x16 | 1453 | 63 |
  | 64x64x8 / 64x64x32 | 1271 / 1516 | 66 / 61 |
  | 128x64x16 / 64x128x16 | 1647 / 1723 | 81 / 68 |
  | 64x256x16 / 256x64x16 | 1866 / 1663 | - |
  | 128x128x8 | 1805 | 127 |
  | 128x128x16 (kept) | 1941-1966 (2040 with asm barrier) | 110 |
  | 128x128x32 | 1984 | - |

  Final through the host runner: N=1024 1965 GFLOP/s (rocBLAS 1548), N=2048 2283 (rocBLAS 1399). Register prefetch of the next k-tile gave no gain (1941 vs 1941); not kept. BK=32 gives about 2% for double the LDS; not kept.
- Launchers live at `kernels::<file>::<fn>::launch(module, stream, grid, args...)` (the macro emits `pub mod <fn>` inside the file module); gemm grid is `(ceil(n/BN), ceil(m/BM))`.
- `hip_tile_build::link_rocm()` in `build.rs` sets the rpath for libamdhip64 / librocblas so test and runner binaries run without `LD_LIBRARY_PATH`.
- Gotcha: zsh does not word-split `$var` in `set -- $var`/`for`; use bash for the scratch bench scripts.
- Benchmarks on the 680M iGPU are bimodal: the GPU clock follows a power budget shared with the CPU (e.g. a concurrent cargo build). Same binary, same kernel: tile gemm 1285 vs 2118 GFLOP/s, rocBLAS 1538 vs 2706. Compare kernels only within one run, take the best of several repetitions after a real warm-up (runner `--bench` does 10 warm-up, best of 5x20), and keep the CPU idle. At high clocks the 128x128x16 tile gemm is about 0.8x rocBLAS (2.1 vs 2.4-2.7 TFLOP/s); the earlier "beats rocBLAS" numbers were a clock artifact.
- YOLO kernels (`kernels::yolo::*`): conv is implicit GEMM (B tile via `Tile::from_fn` gather, no im2col buffer), one `#[kernel]` per tile config generated by a `macro_rules!` (kernel symbol names must be unique; `#[kernel]` cannot be generic). Best GFLOP/s on the real layers: 64x64 tiles ~850-970 (56x40 maps), 64x32 ~600 (14x10), small 7x5 maps only 250-320 (few blocks, long K loop; split-K would be the next step). Auto choice: `Config::choose`.
- Darknet semantics verified in the C++ source for the port (see hip_darknet): inference BN epsilon is 1e-5 (fold in f64, `darknet_network.cpp` `fuse_conv_batchnorm`); `nms_kind=greedynms` means DIoU-NMS (`box_diou`, exponent 0.6), threshold 0.45; maxpool window origin is `-(padding/2)` with padding default `size-1`; route indices >= 0 are absolute; resize is corner-aligned bilinear and a no-op at net size; no letterbox unless `letter_box` is set.
