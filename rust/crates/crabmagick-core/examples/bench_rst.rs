use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a,b| a.partial_cmp(b).unwrap());
    v[v.len()/2]
}

fn bench(label: &str, path: &str) {
    let _ = crabmagick_core::pipeline::decode_any_with_options(path, None, false, 0, None);
    let mut times = Vec::new();
    for _ in 0..7 {
        let t = Instant::now();
        let _ = crabmagick_core::pipeline::decode_any_with_options(path, None, false, 0, None);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let med = median(times.clone());
    eprintln!("{label}: median={med:.1}ms  {:?}", times.iter().map(|t| format!("{:.1}",t)).collect::<Vec<_>>());
}

fn main() {
    bench("JPEG no-RST", "/tmp/test_bench.jpg");
    bench("JPEG RST-1row", "/tmp/test_bench_rst.jpg");
}
