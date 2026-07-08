use crabmagick_core::pipeline::decode_jxl_from_bytes;

/// Microbenchmark for the JXL fast-path decoder.
///
/// Pass the image path as the first CLI argument, or set the `PROBE_JXL`
/// environment variable. Reports the median wall-clock decode time over several
/// timed runs after a warmup pass.
fn main() {
    let path = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("PROBE_JXL").ok())
        .unwrap_or_else(|| {
            "/home/mattia/Work/IIIF_Server/var/storage/f7f3/401b/7c27/455b/907c/b30e/8d8a/eb9f/50.jxl"
                .to_string()
        });
    let bytes = std::fs::read(&path).unwrap();

    // Warmup our decoder (populates any lazily-initialized caches).
    let _ = decode_jxl_from_bytes(&bytes).unwrap();

    // Timed runs.
    let mut times = Vec::new();
    for _ in 0..7 {
        let t = std::time::Instant::now();
        let _ = decode_jxl_from_bytes(&bytes).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        times.push(ms);
        eprintln!("  crabmagick: {ms:.1}ms");
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("crabmagick median: {:.1}ms", times[times.len() / 2]);
}
