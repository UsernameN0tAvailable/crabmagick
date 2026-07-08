//! Reference correctness: compare CrabMagick's decode against libvips (libjpeg-turbo / libjxl).
//!
//! Usage: cargo run -p crabmagick-core --example verify_ref --profile bench -- <image-path>
//!
//! Decodes the image with the CrabMagick public pipeline (multithreaded for JXL) and with
//! libvips, converts both to packed sRGB RGB, and reports max/mean pixel differences plus decode
//! timings. A broken decode path shows up as a large pixel divergence or a dimension mismatch.

use std::time::Instant;

use crabmagick_core::{init, pipeline};
use libvips::{VipsApp, VipsImage, ops};

fn vips_rgb(path: &str) -> (u32, u32, Vec<u8>) {
    let img = VipsImage::new_from_file(path).expect("vips load");
    let img = ops::colourspace(&img, ops::Interpretation::Srgb).expect("to srgb");
    let img = if img.get_bands() == 4 {
        ops::flatten(&img).expect("flatten alpha")
    } else {
        img
    };
    let w = img.get_width() as u32;
    let h = img.get_height() as u32;
    let bands = img.get_bands() as usize;
    let mem = img.image_write_to_memory();
    // Pack down to RGB if libvips produced >3 bands.
    let pixels = if bands == 3 {
        mem
    } else {
        let mut out = Vec::with_capacity((w as usize) * (h as usize) * 3);
        for px in mem.chunks_exact(bands) {
            out.extend_from_slice(&px[..3]);
        }
        out
    };
    (w, h, pixels)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: verify_ref <image-path>");
    init(0, 0);
    let _app = VipsApp::new("crabmagick-verify-ref", false).expect("vips init");

    let t0 = Instant::now();
    let ours = pipeline::decode_any(&path, None, false).expect("crabmagick decode");
    let ours_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let (vw, vh, vpix) = vips_rgb(&path);
    let vips_ms = t1.elapsed().as_secs_f64() * 1e3;

    assert_eq!(
        (ours.width, ours.height),
        (vw, vh),
        "dimension mismatch vs libvips"
    );
    assert_eq!(
        ours.pixels.len(),
        vpix.len(),
        "pixel buffer length mismatch vs libvips"
    );

    let mut max_diff = 0u8;
    let mut sum: u64 = 0;
    let mut within2: u64 = 0;
    for (a, b) in ours.pixels.iter().zip(vpix.iter()) {
        let d = a.abs_diff(*b);
        max_diff = max_diff.max(d);
        sum += d as u64;
        if d <= 2 {
            within2 += 1;
        }
    }
    let n = ours.pixels.len() as f64;
    let mean = sum as f64 / n;
    let pct_within2 = 100.0 * within2 as f64 / n;
    println!(
        "{path}\n  dims {vw}x{vh}  crabmagick={ours_ms:.1}ms  libvips={vips_ms:.1}ms\n  max_diff={max_diff} mean_diff={mean:.4} within±2={pct_within2:.3}%"
    );
    assert!(mean < 3.0, "mean pixel diff vs libvips too high: {mean}");
    assert!(
        pct_within2 > 95.0,
        "too many pixels diverge from libvips: {pct_within2:.2}% within ±2"
    );
    println!("  OK");
}
