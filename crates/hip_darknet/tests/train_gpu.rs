//! The GPU training forward against the CPU reference [`TrainNet`], layer by layer, on the real
//! LegoGears cfg with a batch of images. Skips when the DarkSuite dataset is not present.

#![cfg(feature = "gpu")]

use hip_darknet::cfg::Cfg;
use hip_darknet::train::{GpuTrainNet, TrainNet};

const DIR: &str = "/home/kerrigan/src/DarkSuite/datasets/LegoGears_v2";

/// Largest absolute difference as a fraction of the tensor's largest magnitude. (A per-element
/// relative error with a floor would inflate the figure ~100x on near-zero entries; this is the
/// honest f32 number.)
fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f64, |m, v| m.max(f64::from(v.abs()))).max(1e-9);
    got.iter().zip(want).map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs() / scale).fold(0.0, f64::max)
}

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32) / (1u64 << 24) as f32
    }
}

#[test]
fn gpu_train_forward_matches_cpu_per_layer() {
    let path = std::path::Path::new(DIR).join("LegoGears_v2.cfg");
    if !path.exists() {
        eprintln!("skipping: {} not found", path.display());
        return;
    }
    let cfg = Cfg::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let batch = 2;
    let mut net = TrainNet::new(&cfg, batch, 7).unwrap();
    // Non-trivial BN scale/shift so the affine part of the layer is actually tested.
    let mut next = lcg(11);
    for conv in net.convs.iter_mut().flatten() {
        for s in conv.scales.iter_mut() {
            *s = 0.5 + next();
        }
        for b in conv.bias.iter_mut() {
            *b = next() - 0.5;
        }
    }
    let mut next = lcg(3);
    let input: Vec<f32> = (0..net.input.len()).map(|_| next()).collect();

    let mut gpu = GpuTrainNet::from_net(&net).unwrap();
    net.set_input(&input).unwrap();
    net.forward().unwrap();
    gpu.forward(&input).unwrap();

    let mut worst = 0f64;
    for i in 0..net.model.layers.len() {
        let got = gpu.layer_out(i).unwrap();
        let e = rel_err(&got, &net.outs[i]);
        println!("layer {i:2} {:?}: err/scale {e:.3e}", net.model.layers[i].out_shape);
        assert!(e < 1e-4, "layer {i}: err {e:e}");
        worst = worst.max(e);
        if let Some([mean, var, rmean, rvar]) = gpu.bn_state(i).unwrap() {
            let conv = net.convs[i].as_ref().unwrap();
            for (name, g, w) in [("mean", &mean, &conv.mean), ("var", &var, &conv.var), ("rolling_mean", &rmean, &conv.rolling_mean), ("rolling_var", &rvar, &conv.rolling_var)] {
                // A plain (non-BN) conv leaves its statistics at their initial values on both sides.
                let e = rel_err(g, w);
                assert!(e < 1e-4, "layer {i} {name}: err {e:e}");
            }
        }
    }
    println!("worst layer err/scale {worst:.3e}");
}
