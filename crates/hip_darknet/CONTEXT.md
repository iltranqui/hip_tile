# hip_darknet — Darknet YOLO port

One job: load a Darknet model and detect objects, on the CPU (reference) or the GPU, in pure Rust.

## Contents

- [Files](#files)
- [Dependencies](#dependencies)
- [Rules for changes](#rules-for-changes)
- [Human check](#human-check)

## Files

- `cfg.rs`, `weights.rs` (BN fold), `model.rs` (layers, shapes), `reference.rs` (CPU forward, every layer), `yolo.rs` (decode, DIoU-NMS), `image.rs`, `error.rs`, `lib.rs` (`detect_cpu`, `detect_gpu`).
- Feature `gpu` (default): `gpu.rs` (`GpuModel`), `draw.rs`, `build.rs` (kernels ELF).
- `main.rs`: CLI (`--gpu`, `--out`, `--bench`, `--thresh`, `--nms`).
- `tests/detect.rs` (ground truth), `tests/gpu.rs` (per-layer and detection parity).
- `README.md`: usage and API.

## Dependencies

- Depends on: `image` (png, jpeg), `thiserror`; with `gpu`: `hip-sys`, `hip-rt`, `hip-tile`, `kernels`, build-dep `hip-tile-build`.

## Rules for changes

- Read `_shared/rules.md` first.
- Model work goes through `workflows/port-darknet-model/`; Darknet behaviour comes from `_shared/darknet-semantics.md`.
- Never run the Darknet binary. Keep `main.rs` thin; logic lives in the library.
- Keep `--no-default-features` (CPU only) building.

## Human check

`cargo test -p hip_darknet --release -- --nocapture` and `cargo test -p hip_darknet --release --no-default-features` pass.
