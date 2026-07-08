use crabmagick_core::pipeline::{encode, DecodedImage};
use crabmagick_core::processor::OutputFormat;

fn load_ppm(path: &str) -> Option<DecodedImage> {
    let data = std::fs::read(path).ok()?;
    let mut nl = 0usize;
    let hdr_end = data.iter().position(|&b| {
        if b == b'\n' { nl += 1; nl == 3 } else { false }
    })? + 1;
    let hdr = std::str::from_utf8(&data[..hdr_end]).ok()?;
    let mut parts = hdr.split_ascii_whitespace();
    if parts.next()? != "P6" { return None; }
    let w: u32 = parts.next()?.parse().ok()?;
    let h: u32 = parts.next()?.parse().ok()?;
    let _m: u32 = parts.next()?.parse().ok()?;
    Some(DecodedImage { pixels: data[hdr_end..].to_vec(), width: w, height: h })
}

fn main() {
    let img = load_ppm("/tmp/test_bench.ppm").expect("no ppm");
    let (w, h) = (img.width, img.height);
    let orig = img.pixels.clone();
    let jpeg = encode(img, OutputFormat::Jpeg, 90).expect("encode");
    eprintln!("JPEG size: {} KB", jpeg.len() / 1024);

    // Decode with the `image` crate to confirm the bitstream is valid & correct.
    let decoded = image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg)
        .expect("decode failed");
    let rgb = decoded.to_rgb8();
    assert_eq!(rgb.width(), w);
    assert_eq!(rgb.height(), h);

    // Mean absolute error vs original (sanity: should be small for Q90).
    let dec = rgb.as_raw();
    let mut sum = 0u64;
    for (a, b) in orig.iter().zip(dec.iter()) {
        sum += (*a as i32 - *b as i32).unsigned_abs() as u64;
    }
    let mae = sum as f64 / orig.len() as f64;
    eprintln!("Decoded {}x{} OK, MAE={:.3}", rgb.width(), rgb.height(), mae);
    assert!(mae < 5.0, "MAE too high: {mae}");
    eprintln!("JPEG ROUNDTRIP OK");

    // WebP roundtrip
    let webp = encode(DecodedImage { pixels: orig.clone(), width: w, height: h }, OutputFormat::Webp, 90).expect("webp encode");
    eprintln!("WebP size: {} KB", webp.len() / 1024);
    let wdec = image::load_from_memory_with_format(&webp, image::ImageFormat::WebP)
        .expect("webp decode failed");
    let wrgb = wdec.to_rgb8();
    assert_eq!(wrgb.width(), w);
    assert_eq!(wrgb.height(), h);
    let wdecr = wrgb.as_raw();
    let mut wsum = 0u64;
    for (a, b) in orig.iter().zip(wdecr.iter()) {
        wsum += (*a as i32 - *b as i32).unsigned_abs() as u64;
    }
    let wmae = wsum as f64 / orig.len() as f64;
    eprintln!("WebP decoded {}x{} OK, MAE={:.3}", wrgb.width(), wrgb.height(), wmae);
    assert!(wmae < 8.0, "WebP MAE too high: {wmae}");
    eprintln!("WEBP ROUNDTRIP OK");
}
