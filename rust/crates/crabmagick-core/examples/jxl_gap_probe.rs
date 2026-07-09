use crabmagick_core::jxl_encode::{
    LosslessConfig, LossyConfig, PixelLayout,
    effort::{LosslessInternalParams, LossyInternalParams},
};
use crabmagick_core::pipeline::decode_any_with_options;
use fast_ssim2::compute_ssimulacra2;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

#[derive(Clone)]
struct RgbImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

struct VipsResult {
    ms: f64,
    bytes: usize,
}

struct QualityMetrics {
    psnr: f64,
    ssim2: f64,
}

enum EncodeKind {
    Lossy(LossyConfig, String),
    Lossless(LosslessConfig, String),
}

impl EncodeKind {
    fn label(&self) -> &str {
        match self {
            Self::Lossy(_, label) | Self::Lossless(_, label) => label,
        }
    }

    fn encode(&self, image: &RgbImage) -> Vec<u8> {
        match self {
            Self::Lossy(cfg, _) => cfg
                .encode(&image.pixels, image.width, image.height, PixelLayout::Rgb8)
                .expect("lossy encode failed"),
            Self::Lossless(cfg, _) => cfg
                .encode(&image.pixels, image.width, image.height, PixelLayout::Rgb8)
                .expect("lossless encode failed"),
        }
    }
}

fn median_ms(mut times: Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn load_rgb(path: &Path) -> RgbImage {
    let decoded =
        decode_any_with_options(path.to_str().expect("non-utf8 path"), None, false, 0, None)
            .expect("decode failed");
    RgbImage {
        width: decoded.width,
        height: decoded.height,
        pixels: decoded.pixels,
    }
}

fn pyvips_encode(path: &Path, suffix: &str, runs: usize) -> Option<VipsResult> {
    let script = r#"
import pyvips, time, statistics, sys
pyvips.cache_set_max(0)
img = pyvips.Image.new_from_file(sys.argv[1], access="sequential")
times = []
buf = b""
for _ in range(int(sys.argv[3])):
    t = time.perf_counter()
    buf = img.write_to_buffer(sys.argv[2])
    times.append((time.perf_counter() - t) * 1000.0)
print(f"ms={statistics.median(times):.3f} bytes={len(buf)}")
"#;

    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(path)
        .arg(suffix)
        .arg(runs.to_string())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let mut ms = None;
    let mut bytes = None;
    for part in stdout.split_whitespace() {
        if let Some(v) = part.strip_prefix("ms=") {
            ms = v.parse::<f64>().ok();
        } else if let Some(v) = part.strip_prefix("bytes=") {
            bytes = v.parse::<usize>().ok();
        }
    }
    Some(VipsResult {
        ms: ms?,
        bytes: bytes?,
    })
}

fn bench_ours(kind: &EncodeKind, image: &RgbImage, runs: usize) -> (f64, usize) {
    let first = kind.encode(image);
    let bytes = first.len();
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let _ = kind.encode(image);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    (median_ms(times), bytes)
}

fn decode_rgb(encoded: &[u8]) -> RgbImage {
    let mut file = tempfile::NamedTempFile::new().expect("tempfile create failed");
    std::io::Write::write_all(&mut file, encoded).expect("tempfile write failed");
    let decoded = decode_any_with_options(
        file.path().to_str().expect("non-utf8 temp path"),
        None,
        false,
        0,
        None,
    )
    .expect("roundtrip decode failed");
    RgbImage {
        width: decoded.width,
        height: decoded.height,
        pixels: decoded.pixels,
    }
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        20.0 * f64::log10(255.0) - 10.0 * f64::log10(mse)
    }
}

fn ssim2(original: &RgbImage, decoded: &RgbImage) -> f64 {
    let src: Vec<[f32; 3]> = original
        .pixels
        .chunks_exact(3)
        .map(|rgb| {
            [
                rgb[0] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                rgb[2] as f32 / 255.0,
            ]
        })
        .collect();
    let dst: Vec<[f32; 3]> = decoded
        .pixels
        .chunks_exact(3)
        .map(|rgb| {
            [
                rgb[0] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                rgb[2] as f32 / 255.0,
            ]
        })
        .collect();
    let width = NonZeroUsize::new(original.width as usize).expect("ssim2 width must be non-zero");
    let height =
        NonZeroUsize::new(original.height as usize).expect("ssim2 height must be non-zero");
    let source = Rgb::new(
        src,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("ssim2 source build failed");
    let distorted = Rgb::new(
        dst,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("ssim2 distorted build failed");
    compute_ssimulacra2(source, distorted).unwrap_or(f64::NAN)
}

fn compute_metrics(image: &RgbImage, encoded: &[u8]) -> QualityMetrics {
    let decoded = decode_rgb(encoded);
    QualityMetrics {
        psnr: psnr(&image.pixels, &decoded.pixels),
        ssim2: ssim2(image, &decoded),
    }
}

fn fmt_ratio(ours: usize, theirs: usize) -> String {
    if theirs == 0 {
        return "-".into();
    }
    let delta = (ours as f64 / theirs as f64 - 1.0) * 100.0;
    format!("{delta:+.1}%")
}

fn lossy_variants(threads: usize, distance: f32, effort: u8) -> Vec<(EncodeKind, String)> {
    let base = LossyConfig::new(distance)
        .with_effort(effort)
        .with_threads(threads);
    let vips_suffix = format!(".jxl[distance={distance},effort={effort}]");
    let mut k_ac_low = LossyInternalParams::default();
    k_ac_low.k_ac_quant = Some(0.70);
    let mut k_ac_high = LossyInternalParams::default();
    k_ac_high.k_ac_quant = Some(0.82);
    vec![
        (
            EncodeKind::Lossy(base.clone(), format!("lossy/base-e{effort}")),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone().with_adaptive_block_contexts(false),
                "lossy/no-block-ctx".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone().with_cfl_two_pass(false),
                "lossy/no-cfl2".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(base.clone().with_patches(false), "lossy/no-patches".into()),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(base.clone().with_epf(false), "lossy/no-epf".into()),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone().with_epf_dynamic_sharpness(false),
                "lossy/no-epf-dyn".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone().with_patches(false).with_epf(false),
                "lossy/no-patches+no-epf".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone()
                    .with_patches(false)
                    .with_epf_dynamic_sharpness(false),
                "lossy/no-patches+no-epf-dyn".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone().with_pixel_domain_loss(false),
                "lossy/no-pixel-loss".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                LossyConfig::new(1.02)
                    .with_effort(effort)
                    .with_threads(threads)
                    .with_pixel_domain_loss(false),
                "lossy/no-pixel-loss@d1.02".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                LossyConfig::new(1.04)
                    .with_effort(effort)
                    .with_threads(threads)
                    .with_pixel_domain_loss(false),
                "lossy/no-pixel-loss@d1.04".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                LossyConfig::new(1.06)
                    .with_effort(effort)
                    .with_threads(threads)
                    .with_pixel_domain_loss(false),
                "lossy/no-pixel-loss@d1.06".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone()
                    .with_pixel_domain_loss(false)
                    .with_cfl_two_pass(false),
                "lossy/no-pixel-loss+no-cfl2".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone()
                    .with_pixel_domain_loss(false)
                    .with_adaptive_block_contexts(false),
                "lossy/no-pixel-loss+no-block-ctx".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone()
                    .with_pixel_domain_loss(false)
                    .with_max_strategy_size(Some(32)),
                "lossy/no-pixel-loss+max-32".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone()
                    .with_pixel_domain_loss(false)
                    .with_adaptive_quant(false),
                "lossy/no-pixel-loss+flat-qf".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone().with_gaborish(false),
                "lossy/no-gaborish".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone().with_max_strategy_size(Some(32)),
                "lossy/max-32".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone().with_adaptive_quant(false),
                "lossy/flat-qf".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone()
                    .with_adaptive_quant(false)
                    .with_butteraugli_iters(2),
                "lossy/flat-qf+bfly2".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone().with_internal_params(k_ac_low.clone()),
                "lossy/k_ac_quant=0.70".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone()
                    .with_pixel_domain_loss(false)
                    .with_internal_params(k_ac_low),
                "lossy/no-pixel-loss+k_ac_quant=0.70".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.clone().with_internal_params(k_ac_high.clone()),
                "lossy/k_ac_quant=0.82".into(),
            ),
            vips_suffix.clone(),
        ),
        (
            EncodeKind::Lossy(
                base.with_pixel_domain_loss(false)
                    .with_internal_params(k_ac_high),
                "lossy/no-pixel-loss+k_ac_quant=0.82".into(),
            ),
            vips_suffix,
        ),
    ]
}

fn lossless_variants(threads: usize) -> Vec<(EncodeKind, String)> {
    let base = LosslessConfig::new().with_effort(7).with_threads(threads);
    let eff5 = LosslessConfig::new().with_effort(5).with_threads(threads);
    let eff9 = LosslessConfig::new().with_effort(9).with_threads(threads);
    let mut sample_075 = LosslessInternalParams::default();
    sample_075.tree_sample_fraction = Some(0.75);
    let mut e8_tree = LosslessInternalParams::default();
    e8_tree.tree_num_properties = Some(10);
    e8_tree.tree_max_buckets = Some(128);
    e8_tree.tree_sample_fraction = Some(0.75);
    let mut e9_tree_lite = LosslessInternalParams::default();
    e9_tree_lite.tree_num_properties = Some(16);
    e9_tree_lite.tree_max_buckets = Some(256);
    e9_tree_lite.tree_threshold_base = Some(89.0);
    e9_tree_lite.tree_sample_fraction = Some(0.85);
    e9_tree_lite.wp_num_param_sets = Some(2);
    let mut e9_tree_full = LosslessInternalParams::default();
    e9_tree_full.tree_num_properties = Some(16);
    e9_tree_full.tree_max_buckets = Some(256);
    e9_tree_full.tree_threshold_base = Some(75.0);
    e9_tree_full.tree_sample_fraction = Some(0.95);
    e9_tree_full.wp_num_param_sets = Some(5);
    e9_tree_full.nb_rcts_to_try = Some(19);
    vec![
        (
            EncodeKind::Lossless(eff5, "lossless/base-e5".into()),
            ".jxl[lossless=1,effort=5]".into(),
        ),
        (
            EncodeKind::Lossless(base.clone(), "lossless/base-e7".into()),
            ".jxl[lossless=1,effort=7]".into(),
        ),
        (
            EncodeKind::Lossless(
                base.clone().with_squeeze(false),
                "lossless/e7-no-squeeze".into(),
            ),
            ".jxl[lossless=1,effort=7]".into(),
        ),
        (
            EncodeKind::Lossless(
                base.clone().with_tree_learning(false),
                "lossless/e7-no-tree".into(),
            ),
            ".jxl[lossless=1,effort=7]".into(),
        ),
        (
            EncodeKind::Lossless(
                base.clone().with_squeeze(false).with_tree_learning(true),
                "lossless/e7-tree-no-squeeze".into(),
            ),
            ".jxl[lossless=1,effort=7]".into(),
        ),
        (
            EncodeKind::Lossless(
                base.clone().with_internal_params(sample_075),
                "lossless/sample=0.75".into(),
            ),
            ".jxl[lossless=1,effort=7]".into(),
        ),
        (
            EncodeKind::Lossless(
                base.clone().with_internal_params(e8_tree),
                "lossless/e8-tree@e7".into(),
            ),
            ".jxl[lossless=1,effort=7]".into(),
        ),
        (
            EncodeKind::Lossless(
                base.clone().with_internal_params(e9_tree_lite),
                "lossless/e9-tree-lite".into(),
            ),
            ".jxl[lossless=1,effort=7]".into(),
        ),
        (
            EncodeKind::Lossless(
                base.with_internal_params(e9_tree_full),
                "lossless/e9-tree-full".into(),
            ),
            ".jxl[lossless=1,effort=7]".into(),
        ),
        (
            EncodeKind::Lossless(eff9, "lossless/base-e9".into()),
            ".jxl[lossless=1,effort=9]".into(),
        ),
    ]
}

fn run_group(
    title: &str,
    path: &Path,
    image: &RgbImage,
    variants: Vec<(EncodeKind, String)>,
    filter: Option<&str>,
) {
    println!("## {title}");
    println!("image: {}", path.display());
    println!("dims: {}x{}", image.width, image.height);
    println!("| variant | ours ms | ours KB | PSNR | SSIM2 | vips ms | vips KB | vs vips |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for (kind, suffix) in variants {
        if let Some(filter) = filter {
            if kind.label() != filter {
                continue;
            }
        }
        let (ours_ms, ours_bytes) = bench_ours(&kind, image, 1);
        let ours_encoded = kind.encode(image);
        let metrics = compute_metrics(image, &ours_encoded);
        let vips = pyvips_encode(path, &suffix, 1).expect("pyvips encode failed");
        println!(
            "| {} | {:.2} | {} | {:.2} | {:.2} | {:.2} | {} | {} |",
            kind.label(),
            ours_ms,
            ours_bytes / 1024,
            metrics.psnr,
            metrics.ssim2,
            vips.ms,
            vips.bytes / 1024,
            fmt_ratio(ours_bytes, vips.bytes),
        );
    }
    println!();
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut group = String::from("both");
    let mut filter: Option<String> = None;
    let mut threads = 0usize;
    let mut distance = 1.0f32;
    let mut effort = 5u8;
    let mut path_arg: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        if arg == "--group" {
            group = args.next().expect("missing value after --group");
        } else if arg == "--filter" {
            filter = Some(args.next().expect("missing value after --filter"));
        } else if arg == "--threads" {
            threads = args
                .next()
                .expect("missing value after --threads")
                .parse()
                .expect("invalid usize after --threads");
        } else if arg == "--distance" {
            distance = args
                .next()
                .expect("missing value after --distance")
                .parse()
                .expect("invalid f32 after --distance");
        } else if arg == "--effort" {
            effort = args
                .next()
                .expect("missing value after --effort")
                .parse()
                .expect("invalid u8 after --effort");
        } else {
            path_arg = Some(PathBuf::from(arg));
        }
    }
    let path = path_arg
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/home/mattia/Work/IIIF_Server/var/storage/f7f3/401b/7c27/455b/907c/b30e/8d8a/eb9f/50.jxl",
            )
        });
    let image = load_rgb(&path);
    match group.as_str() {
        "lossy" => run_group(
            &format!("Lossy d={distance:.1} eff={effort}"),
            &path,
            &image,
            lossy_variants(threads, distance, effort),
            filter.as_deref(),
        ),
        "lossless" => run_group(
            "Lossless eff=7",
            &path,
            &image,
            lossless_variants(threads),
            filter.as_deref(),
        ),
        "both" => {
            run_group(
                &format!("Lossy d={distance:.1} eff={effort}"),
                &path,
                &image,
                lossy_variants(threads, distance, effort),
                filter.as_deref(),
            );
            run_group(
                "Lossless eff=7",
                &path,
                &image,
                lossless_variants(threads),
                filter.as_deref(),
            );
        }
        other => panic!("unknown --group value: {other}"),
    }
}
