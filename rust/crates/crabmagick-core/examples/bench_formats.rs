use crabmagick_core::pipeline::decode_any_with_options;
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn bench(label: &str, path: &str) {
    let path_owned = path.to_string();
    // warmup
    let _ = decode_any_with_options(&path_owned, None, false, 0, None);
    let mut times = Vec::new();
    for _ in 0..7 {
        let t = Instant::now();
        let _ = decode_any_with_options(&path_owned, None, false, 0, None);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let med = median(times.clone());
    let runs: Vec<String> = times.iter().map(|t| format!("{:.1}", t)).collect();
    eprintln!("{label}: median={med:.1}ms  runs={runs:?}");
}

fn main() {
    let images = [
        ("/tmp/test_bench.jpg", "JPEG  (1680x2446 Q90)"),
        ("/tmp/test_bench.webp", "WebP  (1680x2446 Q90)"),
        ("/tmp/test_bench.png", "PNG   (1680x2446 lossless)"),
    ];
    for (path, label) in &images {
        if std::path::Path::new(path).exists() {
            bench(label, path);
        } else {
            eprintln!("{label}: skipped (no file)");
        }
    }
}
