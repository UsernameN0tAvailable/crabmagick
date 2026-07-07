//! Codec correctness smoke test using only the public API.
//!
//! Encodes a synthetic RGB image to **lossless** JPEG XL and decodes it back through the
//! full public decode pipeline, asserting byte-for-byte reconstruction. Lossless mode makes
//! the round-trip exact, so any regression in the entropy decoder (the ANS hot path in
//! `jxl_coding::ans`), the modular decoder, or the render pipeline shows up as either a decode
//! error, a dimension mismatch, or a non-zero pixel difference.

use crabmagick_core::pipeline::{decode_jxl_from_bytes, encode_jxl_rgb};
use crabmagick_core::JxlEncodeOptions;

/// Builds a deterministic `width * height` RGB image with per-channel gradients plus a small
/// high-frequency component so the entropy coder has non-trivial content to round-trip.
fn synthetic_rgb(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width.max(1)) as u8;
            let g = (y * 255 / height.max(1)) as u8;
            let b = ((x ^ y) & 0xff) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    pixels
}

#[test]
fn jxl_lossless_roundtrip_is_exact() {
    let (width, height) = (64u32, 48u32);
    let original = synthetic_rgb(width, height);

    let options = JxlEncodeOptions {
        lossless: true,
        threads: 1,
        ..JxlEncodeOptions::default()
    };

    let encoded = encode_jxl_rgb(&original, width, height, &options)
        .expect("lossless JPEG XL encode should succeed");
    assert!(!encoded.is_empty(), "encoder produced empty JXL stream");

    let decoded =
        decode_jxl_from_bytes(&encoded).expect("JPEG XL decode of our own stream should succeed");

    assert_eq!(
        (decoded.width, decoded.height),
        (width, height),
        "decoded dimensions must match the source image"
    );
    assert_eq!(
        decoded.pixels.len(),
        original.len(),
        "decoded pixel buffer length must match the source image"
    );
    assert!(
        decoded.pixels == original,
        "lossless JPEG XL round-trip must reconstruct pixels exactly"
    );
}

#[test]
fn jxl_lossy_roundtrip_is_close() {
    let (width, height) = (64u32, 64u32);
    let original = synthetic_rgb(width, height);

    let options = JxlEncodeOptions {
        lossless: false,
        distance: Some(1.0),
        threads: 1,
        ..JxlEncodeOptions::default()
    };

    let encoded = encode_jxl_rgb(&original, width, height, &options)
        .expect("lossy JPEG XL encode should succeed");
    let decoded =
        decode_jxl_from_bytes(&encoded).expect("JPEG XL decode of our own stream should succeed");

    assert_eq!((decoded.width, decoded.height), (width, height));
    assert_eq!(decoded.pixels.len(), original.len());

    // Lossy decode exercises the VarDCT / color-convert path; reconstruction is approximate but
    // must stay close for a low distance target. A broken DCT or ANS path blows this up.
    let sum: u64 = decoded
        .pixels
        .iter()
        .zip(original.iter())
        .map(|(a, b)| a.abs_diff(*b) as u64)
        .sum();
    let mean = sum as f64 / original.len() as f64;
    assert!(
        mean < 8.0,
        "lossy JPEG XL round-trip mean pixel diff too large: {mean}"
    );
}
