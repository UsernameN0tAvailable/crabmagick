// Run with: cd rust && cargo run --example diag_jpeg --profile bench
use crabmagick_core::pipeline::{encode, decode_any_with_options, DecodedImage};
use crabmagick_core::processor::{EncodeOptions, JpegEncodeOptions, ChromaSubsampling};
use std::io::Write;

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse: f64 = a.iter().zip(b).map(|(&x, &y)| { let d = x as f64 - y as f64; d*d }).sum::<f64>() / a.len() as f64;
    if mse == 0.0 { return f64::INFINITY; }
    20.0 * 255f64.log10() - 10.0 * mse.log10()
}

fn main() {
    // Gradient: easy image for JPEG
    let (w, h) = (64u32, 64u32);
    let pixels: Vec<u8> = (0..w*h).flat_map(|i| {
        let x = i % w; let y = i / w;
        let v = ((x + y) * 255 / (w + h)) as u8;
        [v, v, v]
    }).collect();

    for (label, opts) in [
        ("JPEG Q85 baseline", EncodeOptions::Jpeg(JpegEncodeOptions { quality: 85, ..Default::default() })),
        ("JPEG Q85 progressive", EncodeOptions::Jpeg(JpegEncodeOptions { quality: 85, progressive: true, ..Default::default() })),
        ("JPEG Q85 4:4:4", EncodeOptions::Jpeg(JpegEncodeOptions { quality: 85, chroma_subsampling: ChromaSubsampling::Cs444, ..Default::default() })),
    ] {
        let img = DecodedImage {
            pixels: pixels.clone(),
            alpha: None,
            icc: None,
            exif: None,
            width: w,
            height: h,
        };
        let encoded = encode(img, &opts).expect("encode");
        
        // Write to tempfile and decode back
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&encoded).unwrap();
        let path = f.path().to_str().unwrap().to_string();
        
        match decode_any_with_options(&path, None, false, 0, None) {
            Ok(decoded) => {
                let p = psnr(&pixels, &decoded.pixels);
                let first_orig: Vec<u8> = pixels[..9].to_vec();
                let first_dec: Vec<u8> = decoded.pixels[..9].to_vec();
                println!("{label}: PSNR={p:.1}dB dims={}x{} size={}B orig[0..9]={first_orig:?} dec[0..9]={first_dec:?}", decoded.width, decoded.height, encoded.len());
            }
            Err(e) => println!("{label}: DECODE ERROR: {e}")
        }
    }
}
