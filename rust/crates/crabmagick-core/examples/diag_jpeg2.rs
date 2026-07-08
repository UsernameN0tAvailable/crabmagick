use crabmagick_core::pipeline::{encode, decode_any_with_options, DecodedImage};
use crabmagick_core::processor::{EncodeOptions, JpegEncodeOptions};
use std::io::Write;

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 { return 0.0; }
    let mse: f64 = a[..n].iter().zip(&b[..n]).map(|(&x,&y)| { let d = x as f64-y as f64; d*d }).sum::<f64>() / n as f64;
    if mse == 0.0 { return f64::INFINITY; }
    20.0*255f64.log10() - 10.0*mse.log10()
}

fn test(label: &str, pixels: &[u8], w: u32, h: u32, opts: EncodeOptions) {
    let img = DecodedImage {
            pixels: pixels.to_vec(),
            alpha: None,
            icc: None,
            exif: None,
            width: w,
            height: h,
        };
    let encoded = encode(img, &opts).expect("encode");
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(&encoded).unwrap();
    let path = f.path().to_str().unwrap().to_string();
    match decode_any_with_options(&path, None, false, 0, None) {
        Ok(dec) => {
            let p = psnr(pixels, &dec.pixels);
            let all_zero = dec.pixels.iter().all(|&x| x == 0);
            println!("{label}: PSNR={p:.1}dB dims={}x{} enc_size={}B decoded_all_zero={all_zero}", dec.width, dec.height, encoded.len());
        }
        Err(e) => println!("{label}: DECODE ERROR: {e}")
    }
}

fn main() {
    crabmagick_core::init(0, 0);
    let (w, h) = (800u32, 600u32);
    
    // XOR pattern (what bench_all synthetic_photo generates)
    let xor: Vec<u8> = (0..w*h).flat_map(|i| {
        let x = i % w; let y = i / w;
        [(x*3^y*5) as u8, (x*7^y*11) as u8, (x*13^y*17) as u8]
    }).collect();
    
    // Smooth gradient
    let grad: Vec<u8> = (0..w*h).flat_map(|i| {
        let x = i % w; let y = i / w;
        let v = ((x+y)*255/(w+h)) as u8;
        [v, v, v]
    }).collect();
    
    let q85 = EncodeOptions::Jpeg(JpegEncodeOptions { quality: 85, ..Default::default() });
    let q85p = EncodeOptions::Jpeg(JpegEncodeOptions { quality: 85, progressive: true, ..Default::default() });
    
    test("XOR photo Q85 baseline  ", &xor, w, h, q85.clone());
    test("XOR photo Q85 progressive", &xor, w, h, q85p.clone());
    test("Gradient  Q85 baseline  ", &grad, w, h, q85);
    test("Gradient  Q85 progressive", &grad, w, h, q85p);
}
