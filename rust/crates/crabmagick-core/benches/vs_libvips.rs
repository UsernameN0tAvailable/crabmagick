//! Criterion benchmarks comparing CrabMagick and libvips on real IIIF storage files.
//!
//! Run these on the remote machine where libvips is installed:
//!   from the repo root: `./bench-remote.sh`
//!
//! Metrics to watch:
//! - mean / median time (ns/iter)
//! - throughput (Criterion "elem/s"), which here means input pixels per second
//!   and can be read as MP/s by dividing by 1_000_000.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use crabmagick_core::pipeline::{self, JxlEncodeOptions};
use crabmagick_core::processor::EncodeOptions;
use crabmagick_core::{CrabMagickError, ImageInfo, OutputFormat, get_info, init};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use libvips::{VipsImage, ops};

const STORAGE_ROOT: &str = "/home/mattia/Work/IIIF_Server/var/storage";
const REGION_SIDE: u32 = 256;
const RESIZE_INPUT_SIDE: u32 = 512;
const TILE_INPUT_SIDE: u32 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum InputFormat {
    Jxl,
    Jpeg,
    Webp,
    Tiff,
}

impl InputFormat {
    const ALL: [Self; 4] = [Self::Jxl, Self::Jpeg, Self::Webp, Self::Tiff];

    fn label(self) -> &'static str {
        match self {
            Self::Jxl => "jxl",
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
            Self::Tiff => "tiff",
        }
    }

    fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "jxl" => Some(Self::Jxl),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "webp" => Some(Self::Webp),
            "tif" | "tiff" => Some(Self::Tiff),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Operation {
    FullDecode,
    Region256,
    Resize512To256,
    RegionResizeTile,
}

impl Operation {
    const ALL: [Self; 4] = [
        Self::FullDecode,
        Self::Region256,
        Self::Resize512To256,
        Self::RegionResizeTile,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::FullDecode => "full_decode",
            Self::Region256 => "region_256",
            Self::Resize512To256 => "resize_512_to_256",
            Self::RegionResizeTile => "iiif_tile_1024_to_256",
        }
    }

    fn crop_side(self) -> Option<u32> {
        match self {
            Self::FullDecode => None,
            Self::Region256 => Some(REGION_SIDE),
            Self::Resize512To256 => Some(RESIZE_INPUT_SIDE),
            Self::RegionResizeTile => Some(TILE_INPUT_SIDE),
        }
    }

    fn resize_target(self) -> Option<(u32, u32)> {
        match self {
            Self::Resize512To256 | Self::RegionResizeTile => Some((REGION_SIDE, REGION_SIDE)),
            Self::FullDecode | Self::Region256 => None,
        }
    }

    fn throughput_pixels(self, info: ImageInfo) -> u64 {
        match self.crop_side() {
            Some(side) => {
                let cropped = side.min(info.width).min(info.height).max(1);
                u64::from(cropped) * u64::from(cropped)
            }
            None => u64::from(info.width) * u64::from(info.height),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct OutputSpec {
    label: &'static str,
    format: OutputFormat,
    quality: u8,
    vips_suffix: &'static str,
}

const OUTPUTS: [OutputSpec; 3] = [
    OutputSpec {
        label: "jpeg_q85",
        format: OutputFormat::Jpeg,
        quality: 85,
        vips_suffix: ".jpg[Q=85]",
    },
    OutputSpec {
        label: "webp_q80",
        format: OutputFormat::Webp,
        quality: 80,
        vips_suffix: ".webp[Q=80]",
    },
    OutputSpec {
        label: "jxl_d1_0",
        format: OutputFormat::Jxl,
        quality: 85,
        vips_suffix: ".jxl[distance=1.0]",
    },
];

#[derive(Debug, Clone)]
struct SampleImage {
    format: InputFormat,
    path: PathBuf,
    size_bytes: u64,
    info: ImageInfo,
}

impl SampleImage {
    fn path_str(&self) -> &str {
        self.path
            .to_str()
            .expect("benchmark sample path must be valid UTF-8")
    }
}

fn benchmark_suite(c: &mut Criterion) {
    init(0, 0);
    ensure_vips();

    let samples = discover_samples(Path::new(STORAGE_ROOT));
    if samples.is_empty() {
        panic!("no benchmark images found under {STORAGE_ROOT}");
    }

    for sample in samples.values() {
        eprintln!(
            "benchmark sample: format={} path={} size={} dims={}x{}",
            sample.format.label(),
            sample.path.display(),
            sample.size_bytes,
            sample.info.width,
            sample.info.height
        );
        bench_decode_ops(c, sample);
        bench_pipeline_ops(c, sample);
    }
}

fn bench_decode_ops(c: &mut Criterion, sample: &SampleImage) {
    let mut group = c.benchmark_group(format!("decode/{}", sample.format.label()));

    for op in Operation::ALL {
        if crabmagick_decode(sample, op).is_err() || vips_decode(sample, op).is_err() {
            eprintln!(
                "skipping decode bench: format={} op={} (probe failed)",
                sample.format.label(),
                op.label()
            );
            continue;
        }

        group.throughput(Throughput::Elements(op.throughput_pixels(sample.info)));

        group.bench_function(BenchmarkId::new("crabmagick", op.label()), |b| {
            b.iter(|| {
                let decoded =
                    crabmagick_decode(sample, op).expect("crabmagick decode bench failed");
                black_box((decoded.width, decoded.height, decoded.pixels.len()))
            });
        });

        group.bench_function(BenchmarkId::new("libvips", op.label()), |b| {
            b.iter(|| {
                let bytes = vips_decode(sample, op).expect("libvips decode bench failed");
                black_box(bytes)
            });
        });
    }

    group.finish();
}

fn bench_pipeline_ops(c: &mut Criterion, sample: &SampleImage) {
    let mut group = c.benchmark_group(format!("pipeline/{}", sample.format.label()));

    for op in Operation::ALL {
        group.throughput(Throughput::Elements(op.throughput_pixels(sample.info)));

        for output in OUTPUTS {
            if crabmagick_pipeline(sample, op, output).is_err()
                || vips_pipeline(sample, op, output).is_err()
            {
                eprintln!(
                    "skipping pipeline bench: input={} op={} output={} (probe failed)",
                    sample.format.label(),
                    op.label(),
                    output.label
                );
                continue;
            }

            let bench_id = format!("{}::{}", op.label(), output.label);

            group.bench_function(BenchmarkId::new("crabmagick", &bench_id), |b| {
                b.iter(|| {
                    let encoded = crabmagick_pipeline(sample, op, output)
                        .expect("crabmagick pipeline bench failed");
                    black_box(encoded.len())
                });
            });

            group.bench_function(BenchmarkId::new("libvips", &bench_id), |b| {
                b.iter(|| {
                    let encoded =
                        vips_pipeline(sample, op, output).expect("libvips pipeline bench failed");
                    black_box(encoded.len())
                });
            });
        }
    }

    group.finish();
}

fn discover_samples(root: &Path) -> BTreeMap<InputFormat, SampleImage> {
    let mut best: BTreeMap<InputFormat, SampleImage> = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };

            if file_type.is_dir() {
                stack.push(path);
                continue;
            }

            let Some(format) = InputFormat::from_path(&path) else {
                continue;
            };

            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.len() == 0 {
                continue;
            }

            let Ok(info) = get_info(path.to_str().unwrap_or_default()) else {
                continue;
            };

            let candidate = SampleImage {
                format,
                path: path.clone(),
                size_bytes: metadata.len(),
                info,
            };

            match best.get(&format) {
                Some(existing) if !candidate_is_better(existing, &candidate) => {}
                _ => {
                    best.insert(format, candidate);
                }
            }
        }
    }

    for format in InputFormat::ALL {
        if !best.contains_key(&format) {
            eprintln!("no sample found for {}", format.label());
        }
    }

    best
}

fn candidate_is_better(existing: &SampleImage, candidate: &SampleImage) -> bool {
    let existing_usable = sample_supports_all_ops(existing);
    let candidate_usable = sample_supports_all_ops(candidate);

    match (existing_usable, candidate_usable) {
        (false, true) => return true,
        (true, false) => return false,
        _ => {}
    }

    let existing_pixels = u64::from(existing.info.width) * u64::from(existing.info.height);
    let candidate_pixels = u64::from(candidate.info.width) * u64::from(candidate.info.height);

    if candidate_pixels != existing_pixels {
        return candidate_pixels < existing_pixels;
    }

    candidate.size_bytes < existing.size_bytes
}

fn sample_supports_all_ops(sample: &SampleImage) -> bool {
    sample.info.width >= TILE_INPUT_SIDE && sample.info.height >= TILE_INPUT_SIDE
}

fn crabmagick_decode(
    sample: &SampleImage,
    op: Operation,
) -> Result<pipeline::DecodedImage, CrabMagickError> {
    let region = op
        .crop_side()
        .map(|side| crop_at_ten_percent(sample.info, side));
    let mut decoded = pipeline::decode_any_with_options(sample.path_str(), region, false, 0, None)?;
    if let Some((out_w, out_h)) = op.resize_target() {
        decoded = pipeline::resize_rgb(decoded, out_w, out_h);
    }
    Ok(decoded)
}

fn crabmagick_pipeline(
    sample: &SampleImage,
    op: Operation,
    output: OutputSpec,
) -> Result<Vec<u8>, CrabMagickError> {
    let decoded = crabmagick_decode(sample, op)?;
    match output.format {
        OutputFormat::Jxl => pipeline::encode_jxl_rgb(
            &decoded.pixels,
            decoded.width,
            decoded.height,
            &JxlEncodeOptions {
                distance: Some(1.0),
                ..JxlEncodeOptions::default()
            },
        ),
        _ => pipeline::encode(
            decoded,
            &EncodeOptions::with_quality(output.format, output.quality),
        ),
    }
}

fn vips_decode(sample: &SampleImage, op: Operation) -> Result<usize, String> {
    let image = vips_transform(sample, op)?;
    Ok(image.image_write_to_memory().len())
}

fn vips_pipeline(
    sample: &SampleImage,
    op: Operation,
    output: OutputSpec,
) -> Result<Vec<u8>, String> {
    let image = vips_transform(sample, op)?;
    image
        .image_write_to_buffer(output.vips_suffix)
        .map_err(|e| e.to_string())
}

fn vips_transform(sample: &SampleImage, op: Operation) -> Result<VipsImage, String> {
    ensure_vips();
    let mut image = vips_load(sample).map_err(|e| e.to_string())?;

    if let Some(side) = op.crop_side() {
        let (x, y, w, h) = crop_at_ten_percent(sample.info, side);
        image = ops::extract_area(&image, x as i32, y as i32, w as i32, h as i32)
            .map_err(|e| e.to_string())?;
    }

    if let Some((out_w, _)) = op.resize_target() {
        let scale = f64::from(out_w) / f64::from(image.get_width());
        image = ops::resize(&image, scale).map_err(|e| e.to_string())?;
    }

    Ok(image)
}

fn vips_load(sample: &SampleImage) -> libvips::Result<VipsImage> {
    match sample.format {
        InputFormat::Jpeg => ops::jpegload(sample.path_str())
            .or_else(|_| VipsImage::new_from_file(sample.path_str())),
        InputFormat::Webp => ops::webpload(sample.path_str())
            .or_else(|_| VipsImage::new_from_file(sample.path_str())),
        InputFormat::Tiff => ops::tiffload(sample.path_str())
            .or_else(|_| VipsImage::new_from_file(sample.path_str())),
        InputFormat::Jxl => VipsImage::new_from_file(sample.path_str()),
    }
}

fn crop_at_ten_percent(info: ImageInfo, requested_side: u32) -> (u32, u32, u32, u32) {
    let side = requested_side.min(info.width).min(info.height).max(1);
    let max_x = info.width.saturating_sub(side);
    let max_y = info.height.saturating_sub(side);
    let x = ((info.width as f64) * 0.10).floor() as u32;
    let y = ((info.height as f64) * 0.10).floor() as u32;
    (x.min(max_x), y.min(max_y), side, side)
}

fn ensure_vips() {
    static VIPS_INIT: OnceLock<()> = OnceLock::new();

    VIPS_INIT.get_or_init(|| {
        let app = Box::new(
            libvips::VipsApp::new("crabmagick-vs-libvips-bench", false)
                .expect("failed to init libvips"),
        );
        let _ = Box::leak(app);
    });
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = benchmark_suite
}
criterion_main!(benches);
