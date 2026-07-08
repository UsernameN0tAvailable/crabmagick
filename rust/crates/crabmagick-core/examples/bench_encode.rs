use crabmagick_core::pipeline::{encode, DecodedImage};
use crabmagick_core::processor::OutputFormat;
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Parse binary P6 PPM: scan raw bytes for 3rd newline, parse header as ASCII.
fn load_ppm(path: &str) -> Option<DecodedImage> {
    let data = std::fs::read(path).ok()?;
    let mut nl = 0usize;
    let hdr_end = data.iter().position(|&b| {
        if b == b'\n' { nl += 1; nl == 3 } else { false }
    })? + 1;
    let hdr = std::str::from_utf8(&data[..hdr_end]).ok()?;
    let mut parts = hdr.split_ascii_whitespace();
    let magic = parts.next()?;
    if magic != "P6" { return None; }
    let w: u32 = parts.next()?.parse().ok()?;
    let h: u32 = parts.next()?.parse().ok()?;
    let _maxval: u32 = parts.next()?.parse().ok()?;
    Some(DecodedImage { pixels: data[hdr_end..].to_vec(), width: w, height: h })
}

fn bench(label: &str, mut f: impl FnMut() -> Vec<u8>) {
    let out = f(); // warmup
    let size_kb = out.len() / 1024;
    let mut times = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        let _ = f();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let med = median(times.clone());
    let runs: Vec<String> = times.iter().map(|t| format!("{:.1}", t)).collect();
    eprintln!("{label}: median={med:.1}ms  size={size_kb}KB  runs={runs:?}");
}

fn main() {
    let Some(img) = load_ppm("/tmp/test_bench.ppm") else {
        eprintln!("No /tmp/test_bench.ppm — run bench_formats first to create test images");
        return;
    };
    let (w, h) = (img.width, img.height);
    eprintln!("Encoding {w}x{h} ({} KB raw RGB)", img.pixels.len() / 1024);

    let mk = |p: &[u8]| DecodedImage { pixels: p.to_vec(), width: w, height: h };
    let px = img.pixels.clone();

    bench("JPEG Q90 encode",  || encode(mk(&px), OutputFormat::Jpeg, 90).unwrap());
    bench("WebP Q90 encode",  || encode(mk(&px), OutputFormat::Webp, 90).unwrap());
    bench("PNG encode",       || encode(mk(&px), OutputFormat::Png, 90).unwrap());
    bench("JXL d1.0 encode",  || encode(mk(&px), OutputFormat::Jxl, 90).unwrap());
    bench("TIFF LZW encode",  || encode(mk(&px), OutputFormat::Tiff, 90).unwrap());
}
