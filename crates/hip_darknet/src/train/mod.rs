//! Training in pure Rust: Darknet's box math and deltas, a training-mode forward/backward, and
//! the SGD step. f32 end to end by decision (`ROADMAP.md`).
//!
//! This is the CPU reference for the GPU trainer: every gradient here is checked numerically in
//! `tests/gradients.rs`, since the Darknet binary may never be run (`_shared/rules.md`).

pub mod checkpoint;
pub mod deltas;
pub mod driver;
pub mod net;

pub use deltas::{box_ciou, box_diou, box_giou, box_iou, dx_box_iou, head_cost, yolo_delta, Box, DxRep};
pub use driver::{train, LOSS_CSV_HEADER, MAP_CSV_HEADER, LossRow, TrainOptions, TrainSummary};
pub use net::{learning_rate, ConvState, TrainNet, BN_EPS, ROLLING_MOMENTUM};
