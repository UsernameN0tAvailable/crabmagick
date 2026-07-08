use crabmagick_core::pipeline::{encode, DecodedImage};
use crabmagick_core::processor::{EncodeOptions, JpegEncodeOptions};
use std::process::Command;

fn main() {
    crabmagick_core::init(0, 0);
    let (w, h) = (800u32, 600u32);
    let xor: Vec<u8> = (0..w*h).flat_map(|i| {
        let x = i%w; let y = i/w;
        [(x*3^y*5) as u8, (x*7^y*11) as u8, (x*13^y*17) as u8]
    }).collect();
    
    let img = DecodedImage {
            pixels: xor.clone(),
            alpha: None,
            icc: None,
            exif: None,
            width: w,
            height: h,
        };
    let opts = EncodeOptions::Jpeg(JpegEncodeOptions { quality: 85, ..Default::default() });
    let encoded = encode(img, &opts).expect("encode");
    
    // Write to a .jpg file
    std::fs::write("/tmp/diag_out.jpg", &encoded).unwrap();
    
    // Ask pyvips to decode it and compare
    let out = Command::new("python3").arg("-c").arg(r#"
import pyvips, math
orig_ppm = pyvips.Image.new_from_file('/tmp/xor_photo.ppm')
decoded = pyvips.Image.new_from_file('/tmp/diag_out.jpg')
diff = (orig_ppm.cast(pyvips.BandFormat.FLOAT) - decoded.cast(pyvips.BandFormat.FLOAT)).pow(2)
mse = diff.avg()
psnr = 10*math.log10(255**2/mse) if mse>0 else float('inf')
print(f'pyvips decodes OUR encoder output: PSNR={psnr:.1f}dB size_encoded={decoded.width}x{decoded.height}')
"#).output().unwrap();
    println!("{}", String::from_utf8_lossy(&out.stdout));
    println!("{}", String::from_utf8_lossy(&out.stderr));
    
    // Also check: does our decoder decode a VIPS-encoded JPEG correctly?
    // First create vips JPEG
    let _ = Command::new("python3").arg("-c").arg(
        "import pyvips; pyvips.Image.new_from_file('/tmp/xor_photo.ppm').jpegsave('/tmp/vips_xor.jpg', Q=85)"
    ).status();
    
    // Decode vips JPEG with our decoder
    match crabmagick_core::pipeline::decode_any_with_options("/tmp/vips_xor.jpg", None, false, 0, None) {
        Ok(dec) => {
            let n = xor.len().min(dec.pixels.len());
            let mse: f64 = xor[..n].iter().zip(&dec.pixels[..n]).map(|(&a,&b)| { let d=a as f64-b as f64; d*d }).sum::<f64>() / n as f64;
            let psnr = if mse>0.0 { 10.0*(255f64*255.0/mse).log10() } else { f64::INFINITY };
            println!(
                "our decoder on VIPS-encoded JPEG: PSNR={psnr:.1}dB  dims={}x{}",
                dec.width, dec.height
            );
        }
        Err(e) => println!("our decoder on VIPS JPEG: ERROR: {e}")
    }
}
