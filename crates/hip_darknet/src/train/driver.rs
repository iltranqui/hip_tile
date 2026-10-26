//! The training loop: sample → augment → forward → deltas → backward → SGD, with `loss.csv`,
//! `map.csv`, Darknet-format checkpoints and resume.
//!
//! This is the CPU device. The GPU trainer (stages 03/04 of `ROADMAP.md`) plugs into the same
//! loop through [`crate::train::net::TrainNet`]'s interface, so the file formats and the schedule
//! do not change between devices.

use crate::augment::{Augment, Rng};
use crate::cfg::Cfg;
use crate::data::{Dataset, Sample};
use crate::error::{Error, Result};
use crate::train::checkpoint;
use crate::train::net::{learning_rate, TrainNet};
use crate::yolo::DetectOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// What a run should do.
#[derive(Debug, Clone)]
pub struct TrainOptions {
    /// Iterations (one optimizer step each, `batch` images per step).
    pub iters: usize,
    /// Images per step; overrides the cfg's `batch` when `Some`.
    pub batch: Option<usize>,
    pub seed: u64,
    pub save_every: usize,
    pub eval_every: usize,
    pub out_dir: PathBuf,
    /// Start from these weights (unfolded, Darknet format).
    pub init_weights: Option<PathBuf>,
    /// Resume weights + momentum + `seen` (our own state format).
    pub resume: Option<PathBuf>,
    /// Evaluate the validation split (mAP) as the run goes.
    pub eval: bool,
}

impl Default for TrainOptions {
    fn default() -> Self {
        TrainOptions {
            iters: 100,
            batch: None,
            seed: 1,
            save_every: 500,
            eval_every: 1000,
            out_dir: PathBuf::from("."),
            init_weights: None,
            resume: None,
            eval: false,
        }
    }
}

/// What a finished (or partial) run produced.
#[derive(Debug, Clone, Default)]
pub struct TrainSummary {
    pub iterations: usize,
    pub seen: u64,
    pub first_loss: f32,
    pub last_loss: f32,
    pub avg_loss: f32,
    pub seconds: f64,
    pub final_weights: PathBuf,
    pub map_50: f32,
    pub map_50_95: f32,
}

/// Per-iteration numbers, matching the reference csv columns plus a per-image column.
#[derive(Debug, Clone, Copy, Default)]
pub struct LossRow {
    pub iteration: usize,
    pub seen: u64,
    pub loss: f32,
    pub avg_loss: f32,
    pub loss_per_image: f32,
    pub learning_rate: f32,
    pub ms: f32,
}

fn augment_for(cfg: &Cfg) -> Augment {
    // Jitter/resize live on the `[yolo]` heads; the colour keys live on `[net]`. They must agree.
    let mut jitter: Option<f32> = None;
    let mut resize: Option<f32> = None;
    for l in &cfg.layers {
        if let crate::cfg::LayerCfg::Yolo(y) = l {
            match jitter {
                None => jitter = Some(y.train.jitter),
                Some(j) if (j - y.train.jitter).abs() > 1e-6 => {
                    eprintln!("warning: [yolo] heads disagree on `jitter`; using {j}")
                }
                _ => {}
            }
            match resize {
                None => resize = Some(y.train.resize),
                Some(r) if (r - y.train.resize).abs() > 1e-6 => {
                    eprintln!("warning: [yolo] heads disagree on `resize`; using {r}")
                }
                _ => {}
            }
        }
    }
    let t = &cfg.train;
    Augment {
        jitter: jitter.unwrap_or(0.0),
        resize: resize.unwrap_or(1.0),
        flip: t.flip,
        hue: t.hue,
        saturation: t.saturation,
        exposure: t.exposure,
    }
}

fn next_batch(rng: &mut Rng, order: &mut [usize], cursor: &mut usize, batch: usize, n: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(batch);
    while out.len() < batch {
        if *cursor >= order.len() {
            // Fisher-Yates with our own PRNG: a fresh epoch order.
            for i in (1..n).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            *cursor = 0;
        }
        out.push(order[*cursor]);
        *cursor += 1;
    }
    out
}

/// Write the `loss.csv` header the reference dataset uses, plus `loss_per_image`.
pub const LOSS_CSV_HEADER: &str = "iteration,seen,loss,avg_loss,loss_per_image,learning_rate,milliseconds";
/// `map.csv` header, matching the reference dataset's columns.
pub const MAP_CSV_HEADER: &str = "iteration,seen,iou,threshold,map,best_map";

/// Run the training loop on the CPU.
pub fn train(cfg: &Cfg, ds: &Dataset, opts: &TrainOptions) -> Result<TrainSummary> {
    let classes = ds.classes;
    let batch = opts.batch.unwrap_or(cfg.train.batch).max(1);
    let mut net = TrainNet::new(cfg, batch, opts.seed)?;
    let mut seen: u64 = 0;
    if let Some(p) = &opts.resume {
        seen = checkpoint::load_state(p, &mut net)?;
    } else if let Some(p) = &opts.init_weights {
        seen = checkpoint::load_weights(p, &mut net)?;
    }
    let aug = augment_for(cfg);
    let (net_w, net_h) = (net.net_w(), net.net_h());
    let mut rng = Rng::new(opts.seed);
    let mut loss_csv = BufCsv::create(&opts.out_dir.join("loss.csv"), LOSS_CSV_HEADER)?;
    let mut map_csv = BufCsv::create(&opts.out_dir.join("map.csv"), MAP_CSV_HEADER)?;
    let mut order: Vec<usize> = (0..ds.train.len()).collect();
    let mut cursor = 0usize;
    let mut ema = -1.0f32;
    let mut best_map = 0.0f32;
    let start = Instant::now();
    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    let mut final_weights = opts.out_dir.join("last.weights");
    let mut map_50 = 0.0f32;
    let mut map_50_95 = 0.0f32;
    let mut eval_model: Option<crate::model::Model> = None;

    for iteration in 1..=opts.iters {
        let t0 = Instant::now();
        let picks = next_batch(&mut rng, &mut order, &mut cursor, batch, ds.train.len());
        let mut input = Vec::with_capacity(batch * 3 * net_h * net_w);
        let mut truths: Vec<Vec<crate::data::BoxLabel>> = Vec::with_capacity(batch);
        for &idx in &picks {
            let sample: &Sample = &ds.train[idx];
            let img = ds.image(sample)?;
            let prep = crate::augment::prepare(&mut rng, &img, &sample.labels, net_w as u32, net_h as u32, &aug);
            if prep.data.len() != 3 * net_h * net_w {
                return Err(Error::Model(format!(
                    "{}: prepared image has {} floats, expected {} (cfg channels must be 3)",
                    sample.image.display(),
                    prep.data.len(),
                    3 * net_h * net_w
                )));
            }
            input.extend_from_slice(&prep.data);
            truths.push(prep.labels);
        }
        net.set_input(&input)?;
        net.zero_grads();
        net.forward()?;
        let loss = net.loss_and_deltas(&truths)?;
        net.backward();
        let lr = learning_rate(&cfg.train, iteration.saturating_sub(1));
        net.update(lr);

        seen += batch as u64;
        if iteration == 1 {
            ema = loss;
            first_loss = loss;
        } else {
            ema = ema * 0.9 + loss * 0.1;
        }
        last_loss = loss;
        let row = LossRow {
            iteration,
            seen,
            loss,
            avg_loss: ema,
            loss_per_image: loss / batch as f32,
            learning_rate: lr,
            ms: t0.elapsed().as_secs_f32() * 1e3,
        };
        loss_csv.row(row)?;

        let should_save = opts.save_every > 0 && (iteration % opts.save_every == 0 || iteration == opts.iters);
        if should_save {
            std::fs::create_dir_all(&opts.out_dir)
                .map_err(|e| Error::Io { path: opts.out_dir.display().to_string(), source: e })?;
            let p = opts.out_dir.join(format!("iter_{iteration:06}.weights"));
            checkpoint::save_weights(&p, &net, seen)?;
            checkpoint::save_state(opts.out_dir.join("last.state"), &net, seen)?;
            final_weights = p;
        }
        if opts.eval && opts.eval_every > 0 && iteration % opts.eval_every == 0 && !ds.valid.is_empty() {
            let model = net.to_model()?;
            let res = crate::metrics::evaluate(&model, classes, &ds.valid, |s| ds.image(s), &DetectOptions::default())?;
            map_50 = res.map_50;
            map_50_95 = res.map_50_95;
            best_map = best_map.max(res.map_50);
            map_csv.row_raw(&format!(
                "{iteration},{seen},0.5,0.005,{:.8},{:.8}",
                res.map_50, best_map
            ))?;
            map_csv.row_raw(&format!("{iteration},{seen},0.5:0.95,0.05,{:.8},{:.8}", res.map_50_95, best_map))?;
            eprintln!(
                "iter {iteration} loss {loss:.4} (avg {ema:.4}) lr {lr:.8} mAP@0.5 {:.4} mAP@0.5:0.95 {:.4}",
                res.map_50, res.map_50_95
            );
            eval_model = Some(model);
        } else if iteration == 1 || iteration % 50 == 0 {
            eprintln!("iter {iteration} loss {loss:.4} (avg {ema:.4}) lr {lr:.8} {:.0} ms", row.ms);
        }
    }
    loss_csv.flush()?;
    map_csv.flush()?;
    let _ = eval_model;
    Ok(TrainSummary {
        iterations: opts.iters,
        seen,
        first_loss,
        last_loss,
        avg_loss: ema.max(0.0),
        seconds: start.elapsed().as_secs_f64(),
        final_weights,
        map_50,
        map_50_95,
    })
}

/// A tiny append-only CSV writer, flushed per row so a long run always leaves usable numbers.
struct BufCsv {
    w: std::io::BufWriter<std::fs::File>,
    path: PathBuf,
}

impl BufCsv {
    fn create(path: &Path, header: &str) -> Result<BufCsv> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| Error::Io { path: dir.display().to_string(), source: e })?;
        }
        let f = std::fs::File::create(path).map_err(|e| Error::Io { path: path.display().to_string(), source: e })?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(w, "{header}").map_err(|e| Error::Io { path: path.display().to_string(), source: e })?;
        Ok(BufCsv { w, path: path.to_path_buf() })
    }
    fn row(&mut self, r: LossRow) -> Result<()> {
        let line = format!(
            "{},{},{:.6},{:.6},{:.6},{:.8},{:.3}",
            r.iteration, r.seen, r.loss, r.avg_loss, r.loss_per_image, r.learning_rate, r.ms
        );
        self.row_raw(&line)
    }
    fn row_raw(&mut self, line: &str) -> Result<()> {
        writeln!(self.w, "{line}").map_err(|e| Error::Io { path: self.path.display().to_string(), source: e })?;
        self.w.flush().map_err(|e| Error::Io { path: self.path.display().to_string(), source: e })
    }
    fn flush(&mut self) -> Result<()> {
        self.w.flush().map_err(|e| Error::Io { path: self.path.display().to_string(), source: e })
    }
}
