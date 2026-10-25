//! `hip_darknet` CLI: run a Darknet YOLO model on images (CPU path) and print Darknet-style lines.

use hip_darknet::{Detection, DetectOptions, Model, detect_cpu, image::Image, load_names};
use std::path::Path;
use std::time::Instant;

const USAGE: &str =
    "usage: hip_darknet --cfg C --weights W --names N [--gpu] [--thresh 0.25] [--nms 0.45] [--out PATH] [--bench N] image...
  --gpu       run the forward pass on the AMD GPU (needs the default `gpu` feature)
  --out PATH  write annotated image(s); a directory when several images are given
  --bench N   time N forwards on the first image (CPU and, with --gpu, GPU)";

struct Args {
    cfg: String,
    weights: String,
    names: String,
    opts: DetectOptions,
    images: Vec<String>,
    gpu: bool,
    out: Option<String>,
    bench: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let (mut cfg, mut weights, mut names) = (None, None, None);
    let mut opts = DetectOptions::default();
    let mut images = vec![];
    let (mut gpu, mut out, mut bench) = (false, None, None);
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        let mut val = |what: &str| {
            i += 1;
            argv.get(i).cloned().ok_or(format!("{what} needs a value"))
        };
        match a {
            "--cfg" => cfg = Some(val("--cfg")?),
            "--weights" => weights = Some(val("--weights")?),
            "--names" => names = Some(val("--names")?),
            "--thresh" => opts.thresh = val("--thresh")?.parse().map_err(|e| format!("--thresh: {e}"))?,
            "--nms" => opts.nms_thresh = val("--nms")?.parse().map_err(|e| format!("--nms: {e}"))?,
            "--gpu" => gpu = true,
            "--out" => out = Some(val("--out")?),
            "--bench" => bench = Some(val("--bench")?.parse().map_err(|e| format!("--bench: {e}"))?),
            "-h" | "--help" => return Err(USAGE.into()),
            s if s.starts_with("--") => return Err(format!("unknown option {s}\n{USAGE}")),
            _ => images.push(a.to_string()),
        }
        i += 1;
    }
    let need = |v: Option<String>, n: &str| v.ok_or(format!("missing {n}\n{USAGE}"));
    Ok(Args { cfg: need(cfg, "--cfg")?, weights: need(weights, "--weights")?, names: need(names, "--names")?, opts, images, gpu, out, bench })
}

/// One detector: CPU, or GPU with the weights resident.
enum Backend<'m> {
    Cpu(&'m Model),
    #[cfg(feature = "gpu")]
    Gpu(&'m Model, hip_darknet::gpu::GpuModel<'m>),
}

impl Backend<'_> {
    fn detect(&self, img: &Image, opts: &DetectOptions) -> hip_darknet::Result<Vec<Detection>> {
        match self {
            Backend::Cpu(m) => detect_cpu(m, img, opts),
            #[cfg(feature = "gpu")]
            Backend::Gpu(m, g) => hip_darknet::detect_gpu(m, g, img, opts),
        }
    }
}

#[cfg(feature = "gpu")]
fn make_gpu(model: &Model) -> Result<Backend<'_>, Box<dyn std::error::Error>> {
    Ok(Backend::Gpu(model, hip_darknet::gpu::GpuModel::new(model)?))
}

#[cfg(not(feature = "gpu"))]
fn make_gpu(_: &Model) -> Result<Backend<'_>, Box<dyn std::error::Error>> {
    Err("--gpu: built without the `gpu` feature (rebuild with default features)".into())
}

/// Best-of-rounds mean ms per call (the iGPU clock varies with the shared power budget).
fn best_ms(rounds: usize, n: usize, mut f: impl FnMut()) -> f64 {
    (0..rounds)
        .map(|_| {
            let t = Instant::now();
            (0..n).for_each(|_| f());
            t.elapsed().as_secs_f64() * 1e3 / n as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn bench(model: &Model, gpu: bool, img: &Image, opts: &DetectOptions, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let sized = img.resize(model.net.width, model.net.height);
    let cpu_n = n.clamp(1, 20);
    detect_cpu(model, img, opts)?;
    let cpu = best_ms(3, cpu_n, || {
        hip_darknet::reference::forward(model, &sized.data).unwrap();
    });
    let cpu_e2e = best_ms(3, cpu_n, || {
        detect_cpu(model, img, opts).unwrap();
    });
    println!("cpu: forward {cpu:.2} ms/frame, end-to-end {cpu_e2e:.2} ms/frame (best of 3 x {cpu_n})");
    if gpu {
        #[cfg(feature = "gpu")]
        {
            let g = hip_darknet::gpu::GpuModel::new(model)?;
            let be = Backend::Gpu(model, g);
            let Backend::Gpu(_, g) = &be else { unreachable!() };
            for _ in 0..10 {
                g.run(&sized.data)?;
                be.detect(img, opts)?;
            }
            let fwd = best_ms(5, n, || g.run(&sized.data).unwrap());
            let e2e = best_ms(5, n, || {
                be.detect(img, opts).unwrap();
            });
            println!(
                "gpu: forward {fwd:.2} ms/frame (upload + kernels + sync), end-to-end {e2e:.2} ms/frame (resize, upload, download, decode; best of 5 x {n}); speedup {:.1}x fwd, {:.1}x e2e",
                cpu / fwd, cpu_e2e / e2e
            );
        }
    }
    Ok(())
}

fn out_path(out: &str, many: bool, img: &str) -> std::path::PathBuf {
    if many {
        let stem = Path::new(img).file_stem().map_or("out".into(), |s| s.to_string_lossy().into_owned());
        Path::new(out).join(format!("{stem}.png"))
    } else {
        out.into()
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a = parse_args()?;
    let model = Model::load(&a.cfg, &a.weights)?;
    let names = load_names(&a.names)?;
    if let Some(n) = a.bench {
        let first = a.images.first().ok_or("--bench needs an image")?;
        return bench(&model, a.gpu, &Image::load(first)?, &a.opts, n.max(1));
    }
    let backend = if a.gpu { make_gpu(&model)? } else { Backend::Cpu(&model) };
    let many = a.images.len() > 1;
    if let (Some(out), true) = (&a.out, many) {
        std::fs::create_dir_all(out)?;
    }
    for path in &a.images {
        let img = Image::load(path)?;
        let t = Instant::now();
        let dets = backend.detect(&img, &a.opts)?;
        println!("{path}: predicted in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        for d in &dets {
            let (l, t, w, h) = d.pixel_rect(img.w, img.h);
            let name = names.get(d.class).map_or("?", |s| s.as_str());
            println!(
                "{name}: {:.0}% (left_x: {l:4.0} top_y: {t:4.0} width: {w:4.0} height: {h:4.0})",
                d.prob * 100.0
            );
        }
        if let Some(out) = &a.out {
            let p = out_path(out, many, path);
            hip_darknet::draw::save_annotated(&img, &dets, &p)?;
            println!("wrote {}", p.display());
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
