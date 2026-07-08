use crabmagick_core::pipeline::{encode, encode_jxl_rgb, DecodedImage, JxlEncodeOptions};
use crabmagick_core::processor::{EncodeOptions, OutputFormat};
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
        if b == b'\n' {
            nl += 1;
            nl == 3
        } else {
            false
        }
    })? + 1;
    let hdr = std::str::from_utf8(&data[..hdr_end]).ok()?;
    let mut parts = hdr.split_ascii_whitespace();
    let magic = parts.next()?;
    if magic != "P6" {
        return None;
    }
    let w: u32 = parts.next()?.parse().ok()?;
    let h: u32 = parts.next()?.parse().ok()?;
    let _maxval: u32 = parts.next()?.parse().ok()?;
    Some(DecodedImage {
            pixels: data[hdr_end..].to_vec(),
            alpha: None,
            icc: None,
            exif: None,
            width: w,
            height: h,
        })
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
    let ppm_path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("CRABMAGICK_BENCH_PPM").ok())
        .unwrap_or_else(|| "/tmp/test_bench.ppm".to_string());
    let Some(img) = load_ppm(&ppm_path) else {
        eprintln!(
            "No benchmark PPM at {ppm_path} — pass a path arg, set CRABMAGICK_BENCH_PPM, or create /tmp/test_bench.ppm"
        );
        return;
    };
    let (w, h) = (img.width, img.height);
    eprintln!("Encoding {w}x{h} ({} KB raw RGB)", img.pixels.len() / 1024);

    let mk = |p: &[u8]| DecodedImage {
            pixels: p.to_vec(),
            alpha: None,
            icc: None,
            exif: None,
            width: w,
            height: h,
        };
    let px = img.pixels.clone();

    bench("JPEG Q90 encode", || {
        encode(
            mk(&px),
            &EncodeOptions::with_quality(OutputFormat::Jpeg, 90),
        )
        .unwrap()
    });
    bench("WebP Q90 encode", || {
        encode(
            mk(&px),
            &EncodeOptions::with_quality(OutputFormat::Webp, 90),
        )
        .unwrap()
    });
    bench("WebP lossless encode", || {
        encode(
            mk(&px),
            &EncodeOptions::with_quality(OutputFormat::WebpLossless, 90),
        )
        .unwrap()
    });
    bench("PNG encode", || {
        encode(mk(&px), &EncodeOptions::with_quality(OutputFormat::Png, 90)).unwrap()
    });
    bench("JXL d1.0 encode", || {
        encode(mk(&px), &EncodeOptions::with_quality(OutputFormat::Jxl, 90)).unwrap()
    });
    for effort in [1u8, 3, 5, 7, 9] {
        bench(&format!("JXL lossy d1.0 effort={effort}"), || {
            encode_jxl_rgb(
                &px,
                w,
                h,
                &JxlEncodeOptions {
                    lossless: false,
                    effort,
                    distance: Some(1.0),
                    ..JxlEncodeOptions::default()
                },
            )
            .unwrap()
        });
    }
    for effort in [1u8, 3, 5, 7, 9] {
        bench(&format!("JXL lossless effort={effort}"), || {
            encode_jxl_rgb(
                &px,
                w,
                h,
                &JxlEncodeOptions {
                    lossless: true,
                    effort,
                    ..JxlEncodeOptions::default()
                },
            )
            .unwrap()
        });
    }
    bench("TIFF LZW encode", || {
        encode(
            mk(&px),
            &EncodeOptions::with_quality(OutputFormat::Tiff, 90),
        )
        .unwrap()
    });
}
