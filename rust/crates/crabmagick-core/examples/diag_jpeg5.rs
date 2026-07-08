use crabmagick_core::{init, pipeline::{encode, decode_any_with_options, DecodedImage}};
use crabmagick_core::processor::{EncodeOptions, JpegEncodeOptions, ChromaSubsampling};
use std::io::Write;

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    let mse: f64 = a[..n].iter().zip(&b[..n]).map(|(&x,&y)|{let d=x as f64-y as f64;d*d}).sum::<f64>()/n as f64;
    if mse==0.0{f64::INFINITY}else{20.0*255f64.log10()-10.0*mse.log10()}
}
fn rt(w:u32,h:u32,cs:ChromaSubsampling)->f64{
    let px:Vec<u8>=(0..w*h).flat_map(|i|{let x=i%w;let y=i/w;[(x*3^y*5)as u8,(x*7^y*11)as u8,(x*13^y*17)as u8]}).collect();
    let enc=encode(DecodedImage {
            pixels: px.clone(),
            alpha: None,
            icc: None,
            exif: None,
            width: w,
            height: h,
        },
        &EncodeOptions::Jpeg(JpegEncodeOptions{quality:85,chroma_subsampling:cs,..Default::default()})).unwrap();
    let mut f=tempfile::NamedTempFile::new().unwrap();f.write_all(&enc).unwrap();
    match decode_any_with_options(f.path().to_str().unwrap(),None,false,0,None){Ok(d)=>psnr(&px,&d.pixels),Err(_)=>-1.0}
}
fn main(){
    init(0,0);
    // Test: what is the exact MCU-per-row threshold?
    // 4:2:0: MCU = 16px wide. 4:4:4: MCU = 8px wide
    println!("Width  MCU_per_row(420)  PSNR_420  MCU_per_row(444)  PSNR_444");
    for &w in &[64u32,80,96,112,128,144,160,176,192,208,224,240,256] {
        let h=w;
        let mcu420=(w+15)/16; let mcu444=(w+7)/8;
        let p420=rt(w,h,ChromaSubsampling::Auto);
        let p444=rt(w,h,ChromaSubsampling::Cs444);
        println!("{w:3}  {mcu420:>3}({:>4}px)  {p420:>8.1}  {mcu444:>3}({:>4}px)  {p444:>8.1}", mcu420*16, mcu444*8);
    }
}
