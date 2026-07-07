use crabmagick_core::pipeline::decode_jxl_from_bytes;

fn main() {
    let path = "/home/mattia/Work/IIIF_Server/var/storage/f7f3/401b/7c27/455b/907c/b30e/8d8a/eb9f/50.jxl";
    let bytes = std::fs::read(path).unwrap();
    // Warmup
    let _ = decode_jxl_from_bytes(&bytes).unwrap();
    // Timed runs
    for _ in 0..5 {
        let t = std::time::Instant::now();
        let _ = decode_jxl_from_bytes(&bytes).unwrap();
        eprintln!("decode done in {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
    }
}
