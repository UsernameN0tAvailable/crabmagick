//! Comprehensive codec roundtrip coverage using only the public API.

use std::io::Write;
use std::path::Path;

use crabmagick_core::pipeline::{
    DecodedImage, JxlEncodeOptions as RawJxlEncodeOptions, decode_any_with_options,
    decode_jxl_from_bytes, encode, encode_jxl_rgb,
};
#[cfg(feature = "avif")]
use crabmagick_core::processor::AvifEncodeOptions;
use crabmagick_core::processor::{
    ChromaSubsampling, EncodeOptions, GifEncodeOptions, JpegEncodeOptions,
    JxlEncodeOptions as ProcJxlEncodeOptions, PngEncodeOptions, TiffCompression, TiffEncodeOptions,
    WebpEncodeOptions,
};

/// Compute PSNR (dB) between two u8 RGB buffers. Returns f64::INFINITY for identical inputs.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    20.0 * f64::log10(255.0) - 10.0 * f64::log10(mse)
}

/// Synthetic photo-like test image (W×H RGB): perlin-ish using bit patterns.
fn synthetic_photo(w: u32, h: u32) -> Vec<u8> {
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x.wrapping_mul(3) ^ y.wrapping_mul(5)) & 0xff) as u8;
            let g = ((x.wrapping_mul(7) ^ y.wrapping_mul(11)) & 0xff) as u8;
            let b = ((x.wrapping_mul(13) ^ y.wrapping_mul(17)) & 0xff) as u8;
            px.extend_from_slice(&[r, g, b]);
        }
    }
    px
}

/// Decode any encoded buffer back to RGB pixels using the pipeline.
fn decode_to_rgb(encoded: &[u8]) -> (u32, u32, Vec<u8>) {
    let mut f = tempfile::NamedTempFile::new_in(Path::new(env!("CARGO_MANIFEST_DIR"))).unwrap();
    f.write_all(encoded).unwrap();
    let path = f.path().to_str().unwrap().to_string();
    let result = decode_any_with_options(&path, None, false, 0, None)
        .unwrap_or_else(|e| panic!("decode failed: {e}"));
    (result.width, result.height, result.pixels)
}

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

fn smooth_gradient(w: u32, h: u32) -> Vec<u8> {
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = (x * 255 / w.max(1)) as u8;
            let g = (y * 255 / h.max(1)) as u8;
            let b = (((x + y) * 255) / (w + h).max(1)) as u8;
            px.extend_from_slice(&[r, g, b]);
        }
    }
    px
}

fn lossy_test_image(w: u32, h: u32) -> Vec<u8> {
    smooth_gradient(w, h)
}

fn webp_test_image(w: u32, h: u32) -> Vec<u8> {
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h) {
        px.extend_from_slice(&[160, 160, 160]);
    }
    px
}

fn document_pattern(w: u32, h: u32) -> Vec<u8> {
    let mut px = vec![255u8; (w * h * 3) as usize];
    let stride = (w * 3) as usize;
    for y in 0..h {
        for x in 0..w {
            let text_band = (y / 6) % 2 == 0;
            let glyph = ((x / 5) + (y / 9)) % 7 < 3;
            let margin = x > 6 && x + 6 < w && y > 6 && y + 6 < h;
            if text_band && glyph && margin {
                let idx = y as usize * stride + x as usize * 3;
                px[idx..idx + 3].copy_from_slice(&[0, 0, 0]);
            }
        }
    }
    px
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x.abs_diff(y) as u64)
        .sum::<u64>() as f64
        / a.len() as f64
}

fn encode_roundtrip(
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    opts: &EncodeOptions,
) -> (Vec<u8>, Vec<u8>) {
    let encoded = encode(
        DecodedImage {
            pixels: pixels.clone(),
            alpha: None,
            icc: None,
            exif: None,
            width,
            height,
        },
        opts,
    )
    .expect("encode failed");
    let (decoded_w, decoded_h, decoded) = decode_to_rgb(&encoded);
    assert_eq!((decoded_w, decoded_h), (width, height));
    (encoded, decoded)
}

fn assert_magic(encoded: &[u8], magic: &[u8]) {
    assert!(
        encoded.starts_with(magic),
        "magic mismatch: expected {:x?}, got prefix {:x?}",
        magic,
        &encoded[..encoded.len().min(magic.len())]
    );
}

#[test]
fn jxl_lossless_roundtrip_is_exact() {
    let (width, height) = (64u32, 48u32);
    let original = synthetic_rgb(width, height);

    let options = RawJxlEncodeOptions {
        lossless: true,
        threads: 1,
        ..RawJxlEncodeOptions::default()
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

    let options = RawJxlEncodeOptions {
        lossless: false,
        distance: Some(1.0),
        threads: 1,
        ..RawJxlEncodeOptions::default()
    };

    let encoded = encode_jxl_rgb(&original, width, height, &options)
        .expect("lossy JPEG XL encode should succeed");
    let decoded =
        decode_jxl_from_bytes(&encoded).expect("JPEG XL decode of our own stream should succeed");

    assert_eq!((decoded.width, decoded.height), (width, height));
    assert_eq!(decoded.pixels.len(), original.len());

    let mean = mean_abs_diff(&decoded.pixels, &original);
    assert!(
        mean < 8.0,
        "lossy JPEG XL round-trip mean pixel diff too large: {mean}"
    );
}

#[test]
fn jpeg_q85_roundtrip() {
    let (w, h) = (64, 64);
    let original = lossy_test_image(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Jpeg(JpegEncodeOptions {
            quality: 85,
            ..Default::default()
        }),
    );
    assert!(psnr(&original, &decoded) > 35.0);
}

#[test]
fn jpeg_q95_roundtrip() {
    let (w, h) = (64, 64);
    let original = lossy_test_image(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Jpeg(JpegEncodeOptions {
            quality: 95,
            ..Default::default()
        }),
    );
    assert!(psnr(&original, &decoded) > 42.0);
}

#[test]
fn jpeg_progressive_roundtrip() {
    let (w, h) = (64, 64);
    let original = lossy_test_image(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Jpeg(JpegEncodeOptions {
            quality: 85,
            progressive: true,
            ..Default::default()
        }),
    );
    assert!(psnr(&original, &decoded) > 35.0);
}

#[test]
fn jpeg_baseline_roundtrip_wide_mcu_rows() {
    let original_444 = synthetic_photo(80, 80);
    let (_, decoded_444) = encode_roundtrip(
        original_444.clone(),
        80,
        80,
        &EncodeOptions::Jpeg(JpegEncodeOptions {
            quality: 85,
            chroma_subsampling: ChromaSubsampling::Cs444,
            ..Default::default()
        }),
    );
    assert!(psnr(&original_444, &decoded_444) > 22.0);

    let original_420 = synthetic_photo(144, 144);
    let (_, decoded_420) = encode_roundtrip(
        original_420.clone(),
        144,
        144,
        &EncodeOptions::Jpeg(JpegEncodeOptions {
            quality: 85,
            chroma_subsampling: ChromaSubsampling::Auto,
            ..Default::default()
        }),
    );
    assert!(psnr(&original_420, &decoded_420) > 18.0);
}

#[test]
fn jpeg_444_roundtrip() {
    let (w, h) = (64, 64);
    let original = lossy_test_image(w, h);
    let (encoded, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Jpeg(JpegEncodeOptions {
            quality: 90,
            chroma_subsampling: ChromaSubsampling::Cs444,
            ..Default::default()
        }),
    );
    assert_magic(&encoded, &[0xff, 0xd8]);
    assert!(psnr(&original, &decoded) > 40.0);
}

#[test]
fn jpeg_non_aligned_roundtrip() {
    let (w, h) = (100, 99);
    let original = lossy_test_image(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Jpeg(JpegEncodeOptions {
            quality: 85,
            ..Default::default()
        }),
    );
    assert!(psnr(&original, &decoded) > 30.0);
}

#[test]
fn jpeg_tiny_roundtrip() {
    let (w, h) = (8, 8);
    let original = lossy_test_image(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Jpeg(JpegEncodeOptions {
            quality: 85,
            ..Default::default()
        }),
    );
    assert!(psnr(&original, &decoded) > 20.0);
}

#[test]
fn webp_q80_roundtrip() {
    let (w, h) = (64, 64);
    let original = webp_test_image(w, h);
    let (encoded, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Webp(WebpEncodeOptions {
            quality: 80,
            ..Default::default()
        }),
    );
    assert_magic(&encoded, b"RIFF");
    assert!(psnr(&original, &decoded) > 30.0);
}

#[test]
fn webp_q90_roundtrip() {
    let (w, h) = (64, 64);
    let original = webp_test_image(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Webp(WebpEncodeOptions {
            quality: 90,
            ..Default::default()
        }),
    );
    assert!(psnr(&original, &decoded) > 35.0);
}

#[test]
fn webp_effort0_roundtrip() {
    let (w, h) = (64, 64);
    let original = webp_test_image(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Webp(WebpEncodeOptions {
            quality: 80,
            effort: 0,
            ..Default::default()
        }),
    );
    assert!(psnr(&original, &decoded) > 28.0);
}

#[test]
fn webp_effort6_roundtrip() {
    let (w, h) = (64, 64);
    let original = webp_test_image(w, h);
    let fast = encode(
        DecodedImage {
            pixels: original.clone(),
            alpha: None,
            icc: None,
            exif: None,
            width: w,
            height: h,
        },
        &EncodeOptions::Webp(WebpEncodeOptions {
            quality: 80,
            effort: 0,
            ..Default::default()
        }),
    )
    .expect("encode failed");
    let (encoded, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Webp(WebpEncodeOptions {
            quality: 80,
            effort: 6,
            ..Default::default()
        }),
    );
    assert!(psnr(&original, &decoded) > 30.0);
    assert!(encoded.len() < fast.len());
}

#[test]
fn webp_lossless_exact_roundtrip() {
    let (w, h) = (64, 64);
    let original = smooth_gradient(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Webp(WebpEncodeOptions {
            lossless: true,
            ..Default::default()
        }),
    );
    assert_eq!(decoded, original);
}

#[test]
fn png_lossless_exact_roundtrip() {
    let (w, h) = (64, 64);
    let original = smooth_gradient(w, h);
    let (encoded, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Png(PngEncodeOptions::default()),
    );
    assert_magic(&encoded, &[0x89, 0x50, 0x4e, 0x47]);
    assert_eq!(decoded, original);
}

#[test]
fn png_compress1_roundtrip() {
    let (w, h) = (64, 64);
    let original = document_pattern(w, h);
    let (encoded1, decoded1) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Png(PngEncodeOptions {
            compression: 1,
            ..Default::default()
        }),
    );
    let encoded9 = encode(
        DecodedImage {
            pixels: original.clone(),
            alpha: None,
            icc: None,
            exif: None,
            width: w,
            height: h,
        },
        &EncodeOptions::Png(PngEncodeOptions {
            compression: 9,
            ..Default::default()
        }),
    )
    .expect("encode failed");
    assert_eq!(decoded1, original);
    assert!(encoded1.len() > encoded9.len());
}

#[test]
fn png_compress9_roundtrip() {
    let (w, h) = (64, 64);
    let original = document_pattern(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Png(PngEncodeOptions {
            compression: 9,
            ..Default::default()
        }),
    );
    assert_eq!(decoded, original);
}

#[test]
fn png_16bit_roundtrip() {
    let (w, h) = (64, 64);
    let original = smooth_gradient(w, h);
    let (_, decoded) = encode_roundtrip(
        original,
        w,
        h,
        &EncodeOptions::Png(PngEncodeOptions {
            bitdepth: 16,
            ..Default::default()
        }),
    );
    assert_eq!(decoded.len(), (w * h * 3) as usize);
}

#[test]
fn jxl_d05_roundtrip() {
    let (w, h) = (64, 64);
    let original = lossy_test_image(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Jxl(ProcJxlEncodeOptions {
            distance: Some(0.5),
            ..Default::default()
        }),
    );
    assert!(psnr(&original, &decoded) > 45.0);
}

#[test]
fn jxl_d20_roundtrip() {
    let (w, h) = (64, 64);
    let original = lossy_test_image(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Jxl(ProcJxlEncodeOptions {
            distance: Some(2.0),
            ..Default::default()
        }),
    );
    assert!(psnr(&original, &decoded) > 30.0);
}

#[test]
fn jxl_effort3_roundtrip() {
    let (w, h) = (64, 64);
    let original = lossy_test_image(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Jxl(ProcJxlEncodeOptions {
            effort: 3,
            distance: Some(1.0),
            ..Default::default()
        }),
    );
    assert!(psnr(&original, &decoded) > 35.0);
}

#[test]
fn jxl_effort7_roundtrip() {
    let (w, h) = (64, 64);
    let original = lossy_test_image(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Jxl(ProcJxlEncodeOptions {
            effort: 7,
            distance: Some(1.0),
            ..Default::default()
        }),
    );
    assert!(psnr(&original, &decoded) > 38.0);
}

#[test]
fn tiff_lzw_exact_roundtrip() {
    let (w, h) = (64, 64);
    let original = document_pattern(w, h);
    let (encoded, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Tiff(TiffEncodeOptions {
            compression: TiffCompression::Lzw,
            ..Default::default()
        }),
    );
    assert!(encoded.starts_with(b"II") || encoded.starts_with(b"MM"));
    assert_eq!(decoded, original);
}

#[test]
fn tiff_deflate_exact_roundtrip() {
    let (w, h) = (64, 64);
    let original = document_pattern(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Tiff(TiffEncodeOptions {
            compression: TiffCompression::Deflate,
            ..Default::default()
        }),
    );
    assert_eq!(decoded, original);
}

#[test]
fn tiff_none_exact_roundtrip() {
    let (w, h) = (64, 64);
    let original = document_pattern(w, h);
    let lzw = encode(
        DecodedImage {
            pixels: original.clone(),
            alpha: None,
            icc: None,
            exif: None,
            width: w,
            height: h,
        },
        &EncodeOptions::Tiff(TiffEncodeOptions {
            compression: TiffCompression::Lzw,
            ..Default::default()
        }),
    )
    .expect("encode failed");
    let (encoded, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Tiff(TiffEncodeOptions {
            compression: TiffCompression::None,
            ..Default::default()
        }),
    );
    assert_eq!(decoded, original);
    assert!(encoded.len() > lzw.len());
}

#[test]
fn gif_roundtrip_near_lossless() {
    let (w, h) = (64, 64);
    let original = smooth_gradient(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Gif(GifEncodeOptions::default()),
    );
    assert!(mean_abs_diff(&original, &decoded) < 30.0);
}

#[test]
fn bmp_exact_roundtrip() {
    let (w, h) = (64, 64);
    let original = synthetic_photo(w, h);
    let (encoded, decoded) = encode_roundtrip(original.clone(), w, h, &EncodeOptions::Bmp);
    assert_magic(&encoded, b"BM");
    assert_eq!(decoded, original);
}

#[cfg(feature = "avif")]
#[test]
fn avif_q80_roundtrip() {
    let (w, h) = (64, 64);
    let original = lossy_test_image(w, h);
    let (_, decoded) = encode_roundtrip(
        original.clone(),
        w,
        h,
        &EncodeOptions::Avif(AvifEncodeOptions {
            quality: 80,
            ..Default::default()
        }),
    );
    assert!(psnr(&original, &decoded) > 30.0);
}

#[test]
fn jpeg_quality_psnr_ordering() {
    // PSNR must increase with quality for same image
    let (w, h) = (800u32, 600u32);
    let mut px = vec![255u8; (w*h*3) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let tb = (y / 7) % 2 == 0;
            let gl = ((x / 5) + (y / 11)) % 9 < 4;
            let mg = x > 18 && x + 18 < w as usize && y > 18 && y + 18 < h as usize;
            if tb && gl && mg { let i = (y*w as usize+x)*3; px[i]=0; px[i+1]=0; px[i+2]=0; }
        }
    }
    let opts_75 = EncodeOptions::Jpeg(JpegEncodeOptions { quality: 75, ..Default::default() });
    let opts_85 = EncodeOptions::Jpeg(JpegEncodeOptions { quality: 85, ..Default::default() });
    let opts_95 = EncodeOptions::Jpeg(JpegEncodeOptions { quality: 95, ..Default::default() });
    let (_, dec75) = encode_roundtrip(px.clone(), w, h, &opts_75);
    let (_, dec85) = encode_roundtrip(px.clone(), w, h, &opts_85);
    let (_, dec95) = encode_roundtrip(px.clone(), w, h, &opts_95);
    let p75 = psnr(&px, &dec75);
    let p85 = psnr(&px, &dec85);
    let p95 = psnr(&px, &dec95);
    eprintln!("PSNR: Q75={p75:.1}  Q85={p85:.1}  Q95={p95:.1}");
    assert!(p75 > 20.0, "Q75 PSNR {p75:.1} < 20 dB");
    assert!(p85 > 20.0, "Q85 PSNR {p85:.1} < 20 dB");
    assert!(p95 > 30.0, "Q95 PSNR {p95:.1} < 30 dB");
    // Quality ordering
    assert!(p95 >= p85, "Q95 ({p95:.1}) should be >= Q85 ({p85:.1})");
    assert!(p85 >= p75, "Q85 ({p85:.1}) should be >= Q75 ({p75:.1})");
}
