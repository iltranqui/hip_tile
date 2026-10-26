//! Kernels for the training backward pass, in the same f32 NCHW convention as
//! [`crate::yolo`]: every activation is a `[C x H*W]` row-major matrix, so a channel slice of a
//! larger buffer is a view at a row offset.
//!
//! Accumulation, not overwriting: gradients fan in from several layers, so the kernels that
//! write a gradient add to it (`store_tile_add`) and the driver zeros the buffers once per
//! iteration. This is the one place the port deliberately differs from Darknet's `col2im`, which
//! zeroes first (see `_shared/darknet-semantics.md#gradient-routing`).
//!
//! - [`util`]: fill, accumulate, the SGD step, rolling BN statistics.
//! - [`maxpool_bwd`], [`upsample_bwd`]: gather formulations, race-free without atomics.
//! - [`route_bwd`]: channel-block accumulate for route/concat fan-in.
//! - [`bn_stats`], [`bn_bwd`]: batch-norm statistics and gradients.
//! - [`conv_bwd_dw`], [`conv_bwd_dx`]: the two convolution gradients.
//! - [`yolo_delta`]: the `[yolo]` delta tensor.

pub mod bn_bwd;
pub mod bn_stats;
pub mod maxpool_bwd;
pub mod route_bwd;
pub mod upsample_bwd;
pub mod util;

// Still to write (ROADMAP stage 03): `conv_bwd_dw`, `conv_bwd_dx`, `yolo_delta`. They need the
// batched im2col gather and the accumulating GEMM (a `hip-tile` op), so they land in their own
// `add-kernel` runs rather than in this module's first slice.
