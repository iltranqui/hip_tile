# Darknet semantics

Inference behaviour of Darknet that a Rust port must reproduce, each verified in the C++ source under `~/src/DarkSuite/darknet/src-lib/`. Read this in stage 01 of a model port; add any newly verified layer here (one home per fact).

## Contents

- [Weights](#weights)
- [Layers](#layers)
- [YOLO decode and NMS](#yolo-decode-and-nms)
- [Images](#images)

## Weights

- Header (`weights.cpp:1006-1023`): `major`, `minor`, `revision` as i32, then `seen` as u64 when `major*10 + minor >= 2` (else u32).
- Per conv layer (`weights.cpp:768-785`): biases, then (batch-norm layers only) scales, rolling mean, rolling variance, then weights `[filters][in/groups][kh][kw]`.
- Batch-norm folding for inference (`darknet_network.cpp:2905-2940`, `fuse_conv_batchnorm`): `ε = 1e-5`, computed in f64 then rounded to f32: `w' = w·scale/√(var+ε)`, `b' = bias − mean·scale/√(var+ε)`. The 1e-6 in `convolutional_layer.cpp` is the training path.
- A correct loader consumes the file exactly to EOF.

## Layers

- **convolutional** (`darknet_cfg.cpp:2022-2027`): `pad=1` means `size/2` per side. Default activation is `logistic`. Leaky slope 0.1 (`activations.hpp:129-133`).
- **maxpool** (`darknet_cfg.cpp:2336+`, `maxpool_layer.cpp:129,370-395`): `padding` defaults to `size-1`; output `(w + padding − size)/stride + 1`; window origin `−(padding/2)` (integer division); out-of-image taps are ignored.
- **route** (`route_layer.cpp:120-137`, `darknet_cfg.cpp:2204-2250`): negative indices are relative to the current layer, indices `≥ 0` are absolute. With `groups`/`group_id`, every source contributes channel block `group_id` of `groups` equal blocks (contiguous in CHW), concatenated.
- **upsample**: nearest neighbour, `stride` 2.

## YOLO decode and NMS

- `get_yolo_box` (`yolo_layer.cpp:54-88`): `x = (col + x')/lw`, with `x' = σ(tx)·s − 0.5·(s−1)` for `s = scale_x_y` (`yolo_layer.cpp:991`); `w = exp(clamp(tw, ±10))·anchor_w/net_w`, floored at 1e-6.
- Objectness threshold on `σ(obj)`; class probability `σ(obj)·σ(cls)`, zeroed below the threshold (`yolo_layer.cpp:1479-1515`).
- `nms_kind=greedynms` is DIoU-NMS (`box_diou`, hard-coded exponent 0.6; `box.cpp:1063-1081, 2102-2109`). `beta_nms` only affects `diounms`. `default` is plain-IoU `do_nms_sort`.
- NMS threshold 0.45 (`detector.cpp:2524`); the kind comes from the last yolo layer (`detector.cpp:2599-2606`). Sorting is per class by probability; later boxes are suppressed.

## Images

- Resize (`darknet_image.cpp:1720-1790`): separable bilinear with corner-aligned scale `(in−1)/(out−1)`; a no-op at network size. RGB, `u8/255` into planar CHW f32.
- No letterbox unless the cfg sets `letter_box` (`detector.cpp:2557-2561`).
