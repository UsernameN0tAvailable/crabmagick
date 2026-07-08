//! Standalone correctness check for the vendored decode pipeline.
//! Usage: cargo run -p crabmagick-core --example verify_decode -- <image-path>

use crabmagick_core::{OutputFormat, ProcessRequest, process_image};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: verify_decode <image-path>");
    let orig = std::fs::read(&path).expect("read input");

    let request = ProcessRequest::with_quality(OutputFormat::Png, 100);

    let png = process_image(&path, request).expect("crabmagick pipeline decode->png");
    let ours = image::load_from_memory(&png)
        .expect("decode our png")
        .to_rgb8();
    let reference = image::load_from_memory(&orig)
        .expect("reference decode")
        .to_rgb8();

    assert_eq!(
        ours.dimensions(),
        reference.dimensions(),
        "dimension mismatch"
    );
    let mut max_diff = 0u8;
    let mut sum: u64 = 0;
    for (a, b) in ours.as_raw().iter().zip(reference.as_raw().iter()) {
        let d = a.abs_diff(*b);
        max_diff = max_diff.max(d);
        sum += d as u64;
    }
    let mean = sum as f64 / ours.as_raw().len() as f64;
    let (w, h) = ours.dimensions();
    println!("OK {w}x{h} max_diff={max_diff} mean_diff={mean:.4}");
    assert!(mean < 2.0, "mean pixel diff too high: {mean}");
}
