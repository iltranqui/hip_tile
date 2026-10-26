//! The GPU training network: device-resident activations, weights and batch-norm state, with the
//! training-mode forward built from the stage-03 kernels. The CPU [`TrainNet`] is the reference:
//! a [`GpuTrainNet`] is created *from* one, so both start from identical weights and the per-layer
//! outputs can be compared one for one.
//!
//! Layout is NCHW with a batch: image `b` of a layer sits at `b * C*H*W` (`ROADMAP.md` decisions:
//! f32 end to end). Batch handling is per-image launches over offset views
//! (`kernels::yolo::batched`), so the batch-1 inference kernels are untouched.
//!
//! A batch-normalised conv is conv (linear, zero bias) → `bn_stats` → rolling update → `bn_fwd`
//! (normalise, scale/shift, leaky); a plain conv is the fused bias+activation conv.

use hip_rt::{Device, DeviceBuffer, Module, Stream};
use hip_tile::{Tensor, TensorMut};
use kernels::backward::bn_fwd::{BnFwdParams, launch_bn_fwd};
use kernels::backward::bn_stats::{BnStatsParams, launch_bn_stats};
use kernels::backward::util::launch_running_update;
use kernels::backward::yolo_act::{YoloActParams, launch_yolo_act};
use kernels::yolo::batched::{launch_concat_batched, launch_conv_batched, launch_maxpool_batched, launch_upsample_batched};
use kernels::yolo::conv::ConvParams;
use kernels::yolo::maxpool::PoolParams;

use crate::cfg::Activation;
use crate::error::{Error, Result};
use crate::gpu::{KERNELS_ELF, hip, upload};
use crate::model::{LayerKind, Model};
use crate::train::net::{ROLLING_MOMENTUM, TrainNet};

/// One conv layer's device state.
struct GpuConv {
    weights: DeviceBuffer<f32>,
    bias: DeviceBuffer<f32>,
    /// All zeros: the bias of the pre-normalisation conv of a BN layer.
    zero_bias: DeviceBuffer<f32>,
    scales: DeviceBuffer<f32>,
    mean: DeviceBuffer<f32>,
    var: DeviceBuffer<f32>,
    rolling_mean: DeviceBuffer<f32>,
    rolling_var: DeviceBuffer<f32>,
    /// The conv sum before normalisation, `[batch x C x H x W]` (BN layers only).
    sum: Option<DeviceBuffer<f32>>,
    /// Darknet's `x_norm`, kept for the backward pass (BN layers only).
    xhat: Option<DeviceBuffer<f32>>,
}

pub struct GpuTrainNet {
    model: Model,
    batch: usize,
    _device: Device,
    module: Module,
    stream: Stream,
    input: DeviceBuffer<f32>,
    /// Per layer activation; a `[yolo]` layer owns its (activated) output.
    outs: Vec<DeviceBuffer<f32>>,
    convs: Vec<Option<GpuConv>>,
}

impl GpuTrainNet {
    /// Copy `net`'s weights and batch-norm state to the device and allocate the activations.
    pub fn from_net(net: &TrainNet) -> Result<Self> {
        let model = net.model.clone();
        let batch = net.batch;
        let device = hip(Device::init())?;
        let module = hip(Module::from_elf(KERNELS_ELF))?;
        let stream = hip(Stream::new())?;
        let input = hip(DeviceBuffer::zeroed(batch * model.input_shape().len()))?;
        let mut outs = Vec::with_capacity(model.layers.len());
        let mut convs = Vec::with_capacity(model.layers.len());
        for (i, layer) in model.layers.iter().enumerate() {
            outs.push(hip(DeviceBuffer::zeroed(batch * layer.out_shape.len()))?);
            match (&layer.kind, &net.convs[i]) {
                (LayerKind::Conv(c), Some(st)) => {
                    if !matches!(c.cfg.activation, Activation::Linear | Activation::Leaky) {
                        return Err(Error::Unsupported(format!("layer {i}: GPU conv activation {:?}", c.cfg.activation)));
                    }
                    let n = batch * layer.out_shape.len();
                    let bn = c.cfg.batch_normalize;
                    convs.push(Some(GpuConv {
                        weights: hip(DeviceBuffer::from_slice(&st.weights))?,
                        bias: hip(DeviceBuffer::from_slice(&st.bias))?,
                        zero_bias: hip(DeviceBuffer::zeroed(st.bias.len()))?,
                        scales: hip(DeviceBuffer::from_slice(&st.scales))?,
                        mean: hip(DeviceBuffer::from_slice(&st.mean))?,
                        var: hip(DeviceBuffer::from_slice(&st.var))?,
                        rolling_mean: hip(DeviceBuffer::from_slice(&st.rolling_mean))?,
                        rolling_var: hip(DeviceBuffer::from_slice(&st.rolling_var))?,
                        sum: if bn { Some(hip(DeviceBuffer::zeroed(n))?) } else { None },
                        xhat: if bn { Some(hip(DeviceBuffer::zeroed(n))?) } else { None },
                    }));
                }
                _ => convs.push(None),
            }
        }
        Ok(GpuTrainNet { model, batch, _device: device, module, stream, input, outs, convs })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Output pointer of layer `i`'s source: the previous layer, or the network input.
    fn prev_ptr(&self, i: usize) -> *const f32 {
        if i == 0 { self.input.as_ptr() } else { self.outs[i - 1].as_ptr() }
    }

    /// Training-mode forward over the whole batch: upload `chw` (NCHW with the batch), queue every
    /// layer on the stream, synchronise once.
    pub fn forward(&mut self, chw: &[f32]) -> Result<()> {
        if chw.len() != self.input.len() {
            return Err(Error::Model(format!("input has {} floats, expected {}", chw.len(), self.input.len())));
        }
        hip(upload(&self.input, chw))?;
        let (md, st, batch) = (&self.module, Some(&self.stream), self.batch as u32);
        for (i, l) in self.model.layers.iter().enumerate() {
            let prev = self.prev_ptr(i);
            let (ins, outs) = (l.in_shape, l.out_shape);
            let (ohw, ihw) = ((outs.h * outs.w) as u32, (ins.h * ins.w) as u32);
            let out_ptr = self.outs[i].as_ptr().cast_mut();
            let y = TensorMut::new(out_ptr, outs.c as u32, ohw, ohw);
            match &l.kind {
                LayerKind::Conv(c) => {
                    let g = self.convs[i].as_ref().expect("conv state");
                    let k = c.k() as u32;
                    let w = Tensor::new(g.weights.as_ptr(), c.cfg.filters as u32, k, k);
                    let x = Tensor::new(prev, ins.c as u32, ihw, ihw);
                    let leaky = c.cfg.activation == Activation::Leaky;
                    let cp = |act| {
                        ConvParams::new(ins.c as u32, ins.h as u32, ins.w as u32, c.cfg.size as u32, c.cfg.stride as u32, c.cfg.padding as u32, act)
                    };
                    if c.cfg.batch_normalize {
                        let (sum, xhat) = (g.sum.as_ref().expect("bn sum"), g.xhat.as_ref().expect("bn xhat"));
                        let ch = outs.c as u32;
                        let zb = Tensor::vector(g.zero_bias.as_ptr(), ch);
                        let sum_y = TensorMut::new(sum.as_ptr().cast_mut(), ch, ohw, ohw);
                        hip(launch_conv_batched(md, st, batch, x, w, zb, sum_y, &cp(ConvParams::LINEAR)))?;
                        // Channel `c` of every image: `[C x spatial]` rows, image stride `C*spatial`.
                        let sv = Tensor::new(sum.as_ptr(), ch, ohw, ch * ohw);
                        let (mean, var) = (g.mean.as_ptr().cast_mut(), g.var.as_ptr().cast_mut());
                        hip(launch_bn_stats(md, st, sv, TensorMut::new(mean, 1, ch, ch), TensorMut::new(var, 1, ch, ch), &BnStatsParams { batch, spatial: ohw }))?;
                        hip(launch_running_update(md, st, TensorMut::new(g.rolling_mean.as_ptr().cast_mut(), 1, ch, ch), Tensor::new(g.mean.as_ptr(), 1, ch, ch), ROLLING_MOMENTUM))?;
                        hip(launch_running_update(md, st, TensorMut::new(g.rolling_var.as_ptr().cast_mut(), 1, ch, ch), Tensor::new(g.var.as_ptr(), 1, ch, ch), ROLLING_MOMENTUM))?;
                        let flat = |p: *const f32| Tensor::vector(p, batch * ch * ohw);
                        let vec_c = |b: &DeviceBuffer<f32>| Tensor::vector(b.as_ptr(), ch);
                        let act = if leaky { ConvParams::LEAKY } else { ConvParams::LINEAR };
                        hip(launch_bn_fwd(
                            md, st, flat(sum.as_ptr()), vec_c(&g.mean), vec_c(&g.var), vec_c(&g.scales), vec_c(&g.bias),
                            TensorMut::vector(xhat.as_ptr().cast_mut(), batch * ch * ohw),
                            TensorMut::vector(out_ptr, batch * ch * ohw),
                            &BnFwdParams { channels: ch, spatial: ohw, act },
                        ))?;
                    } else {
                        let bias = Tensor::vector(g.bias.as_ptr(), c.cfg.filters as u32);
                        let act = if leaky { ConvParams::LEAKY } else { ConvParams::LINEAR };
                        hip(launch_conv_batched(md, st, batch, x, w, bias, y, &cp(act)))?;
                    }
                }
                LayerKind::Maxpool(m) => {
                    let p = PoolParams::new(ins.h as u32, ins.w as u32, m.size as u32, m.stride as u32, Some(m.padding as u32));
                    hip(launch_maxpool_batched(md, st, batch, Tensor::new(prev, ins.c as u32, ihw, ihw), y, &p))?;
                }
                LayerKind::Upsample { stride } => {
                    hip(launch_upsample_batched(md, st, batch, Tensor::new(prev, ins.c as u32, ihw, ihw), y, ins.w as u32, *stride as u32))?;
                }
                LayerKind::Route { sources, groups, group_id } => {
                    let mut ch0 = 0usize;
                    for &s in sources {
                        let src_c = self.model.layers[s].out_shape.c;
                        let c = src_c / groups;
                        let src = unsafe { self.outs[s].as_ptr().add(group_id * c * ohw as usize) };
                        let dst = TensorMut::new(unsafe { out_ptr.add(ch0 * ohw as usize) }, c as u32, ohw, ohw);
                        hip(launch_concat_batched(md, st, batch, Tensor::new(src, c as u32, ohw, ohw), src_c * ohw as usize, dst, outs.c * ohw as usize))?;
                        ch0 += c;
                    }
                }
                LayerKind::Yolo(y_cfg) => {
                    let plain = outs.len() as u32;
                    let p = YoloActParams {
                        plain,
                        plane: ohw,
                        entries: (y_cfg.classes + 5) as u32,
                        scale_x_y: y_cfg.scale_x_y,
                    };
                    hip(launch_yolo_act(md, st, Tensor::vector(prev, batch * plain), TensorMut::vector(out_ptr, batch * plain), &p))?;
                }
            }
        }
        hip(self.stream.synchronize())
    }

    /// Download layer `i`'s output (`[batch x C x H x W]`), for parity checks.
    pub fn layer_out(&self, i: usize) -> Result<Vec<f32>> {
        hip(self.outs[i].copy_to_vec())
    }

    /// Download a conv layer's batch statistics and rolling statistics:
    /// `(mean, var, rolling_mean, rolling_var)`.
    pub fn bn_state(&self, i: usize) -> Result<Option<[Vec<f32>; 4]>> {
        let Some(g) = &self.convs[i] else { return Ok(None) };
        Ok(Some([
            hip(g.mean.copy_to_vec())?,
            hip(g.var.copy_to_vec())?,
            hip(g.rolling_mean.copy_to_vec())?,
            hip(g.rolling_var.copy_to_vec())?,
        ]))
    }
}
