use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Instant;

use crabmagick_core::pipeline::{DecodedImage, decode_any_with_options, encode};
use crabmagick_core::processor::{
    ChromaSubsampling, EncodeOptions, JpegEncodeOptions, JxlEncodeOptions, PngEncodeOptions,
    TiffCompression, TiffEncodeOptions, WebpEncodeOptions,
};

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
        return f64::INFINITY;
    }
    20.0 * f64::log10(255.0) - 10.0 * f64::log10(mse)
}

fn synthetic_photo(w: u32, h: u32) -> Vec<u8> {
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x.wrapping_mul(3) ^ y.wrapping_mul(5)) & 0xff) as u8;
            let g = ((x.wrapping_mul(7) ^ y.wrapping_mul(11)) & 0xff) as u8;
            let b = ((x.wrapping_mul(13) ^ y.wrapping_mul(17)) & 0xff) as u8;
            px.extend_from_slice(&[r, g, b]);
        }
    }
    px
}

fn document_pattern(w: u32, h: u32) -> Vec<u8> {
    let mut px = vec![255u8; (w * h * 3) as usize];
    let stride = (w * 3) as usize;
    for y in 0..h {
        for x in 0..w {
            let text_band = (y / 7) % 2 == 0;
            let glyph = ((x / 5) + (y / 11)) % 9 < 4;
            let margin = x > 18 && x + 18 < w && y > 18 && y + 18 < h;
            if text_band && glyph && margin {
                let idx = y as usize * stride + x as usize * 3;
                px[idx..idx + 3].copy_from_slice(&[0, 0, 0]);
            }
        }
    }
    px
}

fn gradient(w: u32, h: u32) -> Vec<u8> {
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = (x * 255 / w.max(1)) as u8;
            let g = (y * 255 / h.max(1)) as u8;
            let b = (((x + y) * 255) / (w + h).max(1)) as u8;
            px.extend_from_slice(&[r, g, b]);
        }
    }
    px
}

fn decode_to_rgb(encoded: &[u8]) -> (u32, u32, Vec<u8>) {
    let mut f = tempfile::NamedTempFile::new_in(Path::new(env!("CARGO_MANIFEST_DIR"))).unwrap();
    f.write_all(encoded).unwrap();
    let path = f.path().to_string_lossy().into_owned();
    let decoded = decode_any_with_options(&path, None, false, 0, None).expect("decode failed");
    (decoded.width, decoded.height, decoded.pixels)
}

fn compute_psnr_from_encoded(img: &DecodedImage, encoded: &[u8]) -> f64 {
    let (w, h, decoded) = decode_to_rgb(encoded);
    assert_eq!((w, h), (img.width, img.height));
    psnr(&img.pixels, &decoded)
}

fn bench_encode(
    _label: &str,
    img: &DecodedImage,
    opts: &EncodeOptions,
    runs: usize,
) -> (f64, usize, f64) {
    let out = encode(img.clone(), opts).unwrap();
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        let _ = encode(img.clone(), opts).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    let size_kb = out.len() / 1024;
    let psnr_val = compute_psnr_from_encoded(img, &out);
    (med, size_kb, psnr_val)
}

fn write_ppm(file: &mut tempfile::NamedTempFile, img: &DecodedImage) {
    write!(
        file,
        "P6
{} {}
255
",
        img.width, img.height
    )
    .unwrap();
    file.write_all(&img.pixels).unwrap();
    file.flush().unwrap();
}

fn pyvips_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Command::new("python3")
            .arg("-c")
            .arg("import pyvips")
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    })
}

fn vips_encode_bench(format_suffix: &str, img: &DecodedImage, runs: usize) -> Option<(f64, usize)> {
    if !pyvips_available() {
        return None;
    }

    let mut ppm = tempfile::NamedTempFile::new_in(Path::new(env!("CARGO_MANIFEST_DIR"))).ok()?;
    write_ppm(&mut ppm, img);

    let script = r#"
import pyvips, time, statistics, sys
pyvips.cache_set_max(0)
img = pyvips.Image.new_from_file(sys.argv[1])
times = []
for _ in range(int(sys.argv[3])):
    t = time.perf_counter()
    buf = img.write_to_buffer(sys.argv[2])
    times.append((time.perf_counter() - t) * 1000)
print(f'ms={statistics.median(times):.1f} size={len(buf)//1024}')
"#;

    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(ppm.path())
        .arg(format_suffix)
        .arg(runs.to_string())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let mut ms = None;
    let mut size = None;
    for part in stdout.split_whitespace() {
        if let Some(value) = part.strip_prefix("ms=") {
            ms = value.parse::<f64>().ok();
        } else if let Some(value) = part.strip_prefix("size=") {
            size = value.parse::<usize>().ok();
        }
    }
    Some((ms?, size?))
}

fn fmt_psnr(value: f64) -> String {
    if value.is_infinite() {
        "inf".to_string()
    } else {
        format!("{value:.1}")
    }
}

fn fmt_opt_f64(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "-".to_string())
}

fn fmt_opt_usize(value: Option<usize>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_string())
}

#[derive(Clone)]
struct BenchCase {
    label: &'static str,
    opts: EncodeOptions,
    vips_suffix: &'static str,
}

fn bench_cases() -> Vec<BenchCase> {
    vec![
        BenchCase {
            label: "JPEG Q75",
            opts: EncodeOptions::Jpeg(JpegEncodeOptions {
                quality: 75,
                ..Default::default()
            }),
            vips_suffix: ".jpg[Q=75]",
        },
        BenchCase {
            label: "JPEG Q85",
            opts: EncodeOptions::Jpeg(JpegEncodeOptions {
                quality: 85,
                ..Default::default()
            }),
            vips_suffix: ".jpg[Q=85]",
        },
        BenchCase {
            label: "JPEG Q95",
            opts: EncodeOptions::Jpeg(JpegEncodeOptions {
                quality: 95,
                ..Default::default()
            }),
            vips_suffix: ".jpg[Q=95]",
        },
        BenchCase {
            label: "JPEG Q85 progressive",
            opts: EncodeOptions::Jpeg(JpegEncodeOptions {
                quality: 85,
                progressive: true,
                ..Default::default()
            }),
            vips_suffix: ".jpg[Q=85,interlace=1]",
        },
        BenchCase {
            label: "JPEG Q90 4:4:4",
            opts: EncodeOptions::Jpeg(JpegEncodeOptions {
                quality: 90,
                chroma_subsampling: ChromaSubsampling::Cs444,
                ..Default::default()
            }),
            vips_suffix: ".jpg[Q=90,subsample_mode=off]",
        },
        BenchCase {
            label: "WebP Q80 effort0",
            opts: EncodeOptions::Webp(WebpEncodeOptions {
                quality: 80,
                effort: 0,
                ..Default::default()
            }),
            vips_suffix: ".webp[Q=80,effort=0]",
        },
        BenchCase {
            label: "WebP Q80 effort4",
            opts: EncodeOptions::Webp(WebpEncodeOptions {
                quality: 80,
                effort: 4,
                ..Default::default()
            }),
            vips_suffix: ".webp[Q=80,effort=4]",
        },
        BenchCase {
            label: "WebP Q80 effort6",
            opts: EncodeOptions::Webp(WebpEncodeOptions {
                quality: 80,
                effort: 6,
                ..Default::default()
            }),
            vips_suffix: ".webp[Q=80,effort=6]",
        },
        BenchCase {
            label: "WebP Q90 effort4",
            opts: EncodeOptions::Webp(WebpEncodeOptions {
                quality: 90,
                effort: 4,
                ..Default::default()
            }),
            vips_suffix: ".webp[Q=90,effort=4]",
        },
        BenchCase {
            label: "WebP lossless",
            opts: EncodeOptions::Webp(WebpEncodeOptions {
                lossless: true,
                ..Default::default()
            }),
            vips_suffix: ".webp[lossless=1]",
        },
        BenchCase {
            label: "PNG compression=1",
            opts: EncodeOptions::Png(PngEncodeOptions {
                compression: 1,
                ..Default::default()
            }),
            vips_suffix: ".png[compression=1]",
        },
        BenchCase {
            label: "PNG compression=6",
            opts: EncodeOptions::Png(PngEncodeOptions {
                compression: 6,
                ..Default::default()
            }),
            vips_suffix: ".png[compression=6]",
        },
        BenchCase {
            label: "PNG compression=9",
            opts: EncodeOptions::Png(PngEncodeOptions {
                compression: 9,
                ..Default::default()
            }),
            vips_suffix: ".png[compression=9]",
        },
        BenchCase {
            label: "JXL d=2.0 effort=3",
            opts: EncodeOptions::Jxl(JxlEncodeOptions {
                distance: Some(2.0),
                effort: 3,
                ..Default::default()
            }),
            vips_suffix: ".jxl[distance=2.0,effort=3]",
        },
        BenchCase {
            label: "JXL d=1.0 effort=5",
            opts: EncodeOptions::Jxl(JxlEncodeOptions {
                distance: Some(1.0),
                effort: 5,
                ..Default::default()
            }),
            vips_suffix: ".jxl[distance=1.0,effort=5]",
        },
        BenchCase {
            label: "JXL d=0.5 effort=7",
            opts: EncodeOptions::Jxl(JxlEncodeOptions {
                distance: Some(0.5),
                effort: 7,
                ..Default::default()
            }),
            vips_suffix: ".jxl[distance=0.5,effort=7]",
        },
        BenchCase {
            label: "JXL lossless effort=3",
            opts: EncodeOptions::Jxl(JxlEncodeOptions {
                lossless: true,
                effort: 3,
                ..Default::default()
            }),
            vips_suffix: ".jxl[lossless=1,effort=3]",
        },
        BenchCase {
            label: "TIFF lzw",
            opts: EncodeOptions::Tiff(TiffEncodeOptions {
                compression: TiffCompression::Lzw,
                ..Default::default()
            }),
            vips_suffix: ".tif[compression=lzw]",
        },
        BenchCase {
            label: "TIFF deflate",
            opts: EncodeOptions::Tiff(TiffEncodeOptions {
                compression: TiffCompression::Deflate,
                ..Default::default()
            }),
            vips_suffix: ".tif[compression=deflate]",
        },
        BenchCase {
            label: "TIFF none",
            opts: EncodeOptions::Tiff(TiffEncodeOptions {
                compression: TiffCompression::None,
                ..Default::default()
            }),
            vips_suffix: ".tif[compression=none]",
        },
    ]
}

fn main() {
    let runs = 5;
    let images = vec![
        (
            "photo",
            DecodedImage {
                pixels: synthetic_photo(800, 600),
                width: 800,
                height: 600,
            },
        ),
        (
            "document",
            DecodedImage {
                pixels: document_pattern(800, 600),
                width: 800,
                height: 600,
            },
        ),
        (
            "gradient",
            DecodedImage {
                pixels: gradient(800, 600),
                width: 800,
                height: 600,
            },
        ),
    ];
    let cases = bench_cases();

    for (name, img) in images {
        println!("## {name}\n");
        println!("| Codec/Option | Speed (ms) | Size (KB) | PSNR (dB) | vips (ms) | vips size |");
        println!("|--------------|------------|-----------|-----------|-----------|-----------|");
        for case in &cases {
            let (ms, size_kb, psnr_val) = bench_encode(case.label, &img, &case.opts, runs);
            let vips = vips_encode_bench(case.vips_suffix, &img, runs);
            println!(
                "| {} | {:>10.1} | {:>9} | {:>9} | {:>9} | {:>9} |",
                case.label,
                ms,
                size_kb,
                fmt_psnr(psnr_val),
                fmt_opt_f64(vips.map(|(v, _)| v)),
                fmt_opt_usize(vips.map(|(_, s)| s)),
            );
        }
        println!();
    }
}
