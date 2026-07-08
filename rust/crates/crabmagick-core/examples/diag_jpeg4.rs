use crabmagick_core::{init, pipeline::{encode, decode_any_with_options, DecodedImage}};
use crabmagick_core::processor::{EncodeOptions, JpegEncodeOptions, ChromaSubsampling};
use std::io::Write;

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    let mse: f64 = a[..n].iter().zip(&b[..n]).map(|(&x,&y)| { let d=x as f64-y as f64; d*d }).sum::<f64>()/n as f64;
    if mse==0.0 { f64::INFINITY } else { 20.0*255f64.log10()-10.0*mse.log10() }
}
fn roundtrip(w: u32, h: u32, opts: EncodeOptions) -> f64 {
    let px: Vec<u8> = (0..w*h).flat_map(|i| { let x=i%w; let y=i/w; [(x*3^y*5) as u8,(x*7^y*11) as u8,(x*13^y*17) as u8] }).collect();
    let enc = encode(DecodedImage {
            pixels: px.clone(),
            alpha: None,
            icc: None,
            exif: None,
            width: w,
            height: h,
        }, &opts).unwrap();
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(&enc).unwrap();
    match decode_any_with_options(f.path().to_str().unwrap(), None, false, 0, None) {
        Ok(d) => psnr(&px, &d.pixels),
        Err(e) => { eprintln!("  decode error: {e}"); -1.0 }
    }
}
fn main() {
    init(0, 0);
    let b = |prog, cs: ChromaSubsampling| EncodeOptions::Jpeg(JpegEncodeOptions { quality:85, progressive:prog, chroma_subsampling:cs, ..Default::default() });
    
    println!("Size     Baseline-420  Prog-420  Baseline-444  Prog-444");
    for &(w, h) in &[(8u32,8),(16,16),(32,32),(64,64),(128,128),(256,256),(320,240),(400,300),(512,512),(800,600)] {
        let b420 = roundtrip(w, h, b(false, ChromaSubsampling::Auto));
        let p420 = roundtrip(w, h, b(true,  ChromaSubsampling::Auto));
        let b444 = roundtrip(w, h, b(false, ChromaSubsampling::Cs444));
        let p444 = roundtrip(w, h, b(true,  ChromaSubsampling::Cs444));
        println!("{w:3}x{h:<3}  {b420:>8.1}    {p420:>6.1}    {b444:>8.1}      {p444:>6.1}");
    }
}
