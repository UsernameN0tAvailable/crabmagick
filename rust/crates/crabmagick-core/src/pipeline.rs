use std::fs;
use std::io::{BufReader, Cursor};

use fast_image_resize as fir;
use image::{codecs::png::PngEncoder, ColorType, GenericImageView, ImageEncoder, RgbImage};
use jxl_encoder::{EncoderMode, LosslessConfig, LossyConfig, PixelLayout as JxlLayout};
use jxl_oxide::{CropInfo, JxlImage, PixelFormat};
#[cfg(feature = "pdf")]
use pdfium_render::prelude::*;
#[cfg(feature = "avif")]
use ravif::{Encoder as AvifEncoder, Img as AvifImg, RGB8 as AvifRgb8};
use resvg::{tiny_skia, usvg};
use tiff::decoder::{Decoder as TiffDecoder, DecodingResult};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout as ZenLayout, Unstoppable};

use crate::processor::{ImageInfo, OutputFormat, OxipixError};

#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    Jxl,
    Jpeg,
    Png,
    Webp,
    Tiff,
    Gif,
    Bmp,
    Pdf,
    Svg,
    Avif,
    Unknown,
}

#[derive(Debug, Clone, Copy)]
pub struct JxlEncodeOptions {
    pub lossless: bool,
    pub distance: Option<f32>,
    pub effort: u8,
    pub threads: usize,
    pub mode: EncoderMode,
    pub max_strategy_size: Option<u8>,
    pub force_strategy: Option<u8>,
    pub custom_orders: Option<bool>,
    pub adaptive_block_contexts: Option<bool>,
    pub patches: Option<bool>,
    pub lossless_tree_learning: Option<bool>,
    pub lossless_lz77: Option<bool>,
    pub lossless_squeeze: Option<bool>,
    pub gaborish: Option<bool>,
    pub pixel_domain_loss: Option<bool>,
    pub adaptive_quant: Option<bool>,
    pub adjust_quant_ac: Option<bool>,
    pub chromacity_adjustment: Option<bool>,
    pub cfl: Option<bool>,
    pub cfl_two_pass: Option<bool>,
    pub epf: Option<bool>,
    pub epf_dynamic_sharpness: Option<bool>,
    pub optimize_codes: Option<bool>,
}

impl Default for JxlEncodeOptions {
    fn default() -> Self {
        Self {
            lossless: false,
            distance: Some(1.0),
            effort: 5,
            threads: 0,
            mode: EncoderMode::Experimental,
            max_strategy_size: None,
            force_strategy: None,
            custom_orders: None,
            adaptive_block_contexts: None,
            patches: None,
            lossless_tree_learning: None,
            lossless_lz77: None,
            lossless_squeeze: None,
            gaborish: None,
            pixel_domain_loss: None,
            adaptive_quant: None,
            adjust_quant_ac: None,
            chromacity_adjustment: None,
            cfl: None,
            cfl_two_pass: None,
            epf: None,
            epf_dynamic_sharpness: None,
            optimize_codes: None,
        }
    }
}

pub fn detect_format(bytes: &[u8]) -> SourceFormat {
    if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0x0A {
        return SourceFormat::Jxl;
    }
    if bytes.len() >= 8
        && bytes[0] == 0x00
        && bytes[1] == 0x00
        && bytes[2] == 0x00
        && &bytes[4..8] == b"JXL "
    {
        return SourceFormat::Jxl;
    }
    if bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
        return SourceFormat::Jpeg;
    }
    if bytes.len() >= 4 && &bytes[..4] == b"\x89PNG" {
        return SourceFormat::Png;
    }
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return SourceFormat::Webp;
    }
    if bytes.len() >= 4 && (&bytes[..4] == b"II*\0" || &bytes[..4] == b"MM\0*") {
        return SourceFormat::Tiff;
    }
    if bytes.len() >= 4 && &bytes[..4] == b"GIF8" {
        return SourceFormat::Gif;
    }
    if bytes.len() >= 2 && &bytes[..2] == b"BM" {
        return SourceFormat::Bmp;
    }
    if bytes.len() >= 5 && &bytes[..5] == b"%PDF-" {
        return SourceFormat::Pdf;
    }
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        let brand = &bytes[8..12];
        if brand == b"avif" || brand == b"avis" {
            return SourceFormat::Avif;
        }
    }

    let probe_len = bytes.len().min(1024);
    let probe = String::from_utf8_lossy(&bytes[..probe_len]).to_ascii_lowercase();
    let trimmed = probe.trim_start_matches(['\u{feff}', ' ', '\t', '\r', '\n']);
    if trimmed.starts_with("<svg") || (trimmed.starts_with("<?xml") && probe.contains("<svg")) {
        return SourceFormat::Svg;
    }

    SourceFormat::Unknown
}

pub fn decode_jxl(path: &str) -> Result<DecodedImage, OxipixError> {
    let bytes = fs::read(path)?;
    decode_jxl_from_bytes(&bytes)
}

pub fn decode_jxl_from_bytes(bytes: &[u8]) -> Result<DecodedImage, OxipixError> {
    let image = JxlImage::builder()
        .read(Cursor::new(bytes))
        .map_err(|e| OxipixError::Decode(e.to_string()))?;

    let frame = image
        .render_frame(0)
        .map_err(|e| OxipixError::Decode(e.to_string()))?;
    let planes = frame.image_planar();

    Ok(planar_to_rgb(
        image.pixel_format(),
        planes[0].buf(),
        planes.get(1).map(|plane| plane.buf()),
        planes.get(2).map(|plane| plane.buf()),
        image.width(),
        image.height(),
    ))
}

pub fn decode_jxl_region(
    path: &str,
    left: u32,
    top: u32,
    width: u32,
    height: u32,
) -> Result<DecodedImage, OxipixError> {
    let bytes = fs::read(path)?;
    decode_jxl_region_from_bytes(&bytes, left, top, width, height)
}

pub fn decode_jxl_region_from_bytes(
    bytes: &[u8],
    left: u32,
    top: u32,
    width: u32,
    height: u32,
) -> Result<DecodedImage, OxipixError> {
    let mut image = JxlImage::builder()
        .read(Cursor::new(bytes))
        .map_err(|e| OxipixError::Decode(e.to_string()))?;

    image.set_image_region(CropInfo {
        left,
        top,
        width,
        height,
    });

    let frame = image
        .render_frame(0)
        .map_err(|e| OxipixError::Decode(e.to_string()))?;
    let planes = frame.image_planar();
    if planes.is_empty() {
        return Err(OxipixError::Decode("no planes decoded".to_string()));
    }

    Ok(planar_to_rgb(
        image.pixel_format(),
        planes[0].buf(),
        planes.get(1).map(|plane| plane.buf()),
        planes.get(2).map(|plane| plane.buf()),
        planes[0].width() as u32,
        planes[0].height() as u32,
    ))
}

pub fn decode_jxl_info(path: &str) -> Result<ImageInfo, OxipixError> {
    let bytes = fs::read(path)?;
    decode_jxl_info_from_bytes(&bytes)
}

pub fn decode_jxl_info_from_bytes(bytes: &[u8]) -> Result<ImageInfo, OxipixError> {
    let image = JxlImage::builder()
        .read(Cursor::new(bytes))
        .map_err(|e| OxipixError::Decode(e.to_string()))?;

    Ok(ImageInfo {
        width: image.width(),
        height: image.height(),
    })
}

pub fn decode_any(
    path: &str,
    region: Option<(u32, u32, u32, u32)>,
    square: bool,
) -> Result<DecodedImage, OxipixError> {
    decode_any_with_options(path, region, square, 0, None)
}

pub fn decode_any_info(path: &str, page: u32) -> Result<ImageInfo, OxipixError> {
    let bytes = fs::read(path)?;
    match detect_format(&bytes) {
        SourceFormat::Jxl => decode_jxl_info_from_bytes(&bytes),
        SourceFormat::Svg => decode_svg_info(&bytes),
        SourceFormat::Tiff if page > 0 => {
            let image = decode_tiff_page(&bytes, page)?;
            Ok(ImageInfo {
                width: image.width,
                height: image.height,
            })
        }
        SourceFormat::Pdf => {
            #[cfg(feature = "pdf")]
            {
                let image = decode_pdf_page(&bytes, page, 0, 0)?;
                return Ok(ImageInfo {
                    width: image.width,
                    height: image.height,
                });
            }
            #[cfg(not(feature = "pdf"))]
            {
                return Err(OxipixError::Decode(
                    "PDF support not compiled in (enable the `pdf` feature)".to_string(),
                ));
            }
        }
        SourceFormat::Avif => {
            #[cfg(feature = "avif")]
            {
                let decoded = decode_via_image(&bytes)?;
                return Ok(ImageInfo {
                    width: decoded.width,
                    height: decoded.height,
                });
            }
            #[cfg(not(feature = "avif"))]
            {
                return Err(OxipixError::Decode(
                    "AVIF support not compiled in (enable the `avif` feature)".to_string(),
                ));
            }
        }
        SourceFormat::Jpeg
        | SourceFormat::Png
        | SourceFormat::Webp
        | SourceFormat::Tiff
        | SourceFormat::Gif
        | SourceFormat::Bmp => {
            let image =
                image::load_from_memory(&bytes).map_err(|e| OxipixError::Decode(e.to_string()))?;
            let (width, height) = image.dimensions();
            Ok(ImageInfo { width, height })
        }
        SourceFormat::Unknown => Err(OxipixError::Decode(
            "unsupported or unrecognized image format".to_string(),
        )),
    }
}

pub(crate) fn decode_any_with_options(
    path: &str,
    region: Option<(u32, u32, u32, u32)>,
    square: bool,
    page: u32,
    render_size: Option<(u32, u32)>,
) -> Result<DecodedImage, OxipixError> {
    let bytes = fs::read(path)?;
    let format = detect_format(&bytes);

    let image = match format {
        SourceFormat::Jxl => decode_jxl_any(&bytes, region, square)?,
        SourceFormat::Jpeg
        | SourceFormat::Png
        | SourceFormat::Webp
        | SourceFormat::Gif
        | SourceFormat::Bmp => apply_post_decode_ops(decode_via_image(&bytes)?, region, square),
        SourceFormat::Tiff => {
            apply_post_decode_ops(decode_tiff_page(&bytes, page)?, region, square)
        }
        SourceFormat::Svg => {
            let (out_w, out_h) = render_size.unwrap_or((0, 0));
            apply_post_decode_ops(decode_svg(&bytes, out_w, out_h)?, region, square)
        }
        SourceFormat::Pdf => {
            #[cfg(feature = "pdf")]
            {
                let (out_w, out_h) = render_size.unwrap_or((0, 0));
                apply_post_decode_ops(decode_pdf_page(&bytes, page, out_w, out_h)?, region, square)
            }
            #[cfg(not(feature = "pdf"))]
            {
                return Err(OxipixError::Decode(
                    "PDF support not compiled in (enable the `pdf` feature)".to_string(),
                ));
            }
        }
        SourceFormat::Avif => {
            #[cfg(feature = "avif")]
            {
                apply_post_decode_ops(decode_via_image(&bytes)?, region, square)
            }
            #[cfg(not(feature = "avif"))]
            {
                return Err(OxipixError::Decode(
                    "AVIF support not compiled in (enable the `avif` feature)".to_string(),
                ));
            }
        }
        SourceFormat::Unknown => {
            return Err(OxipixError::Decode(
                "unsupported or unrecognized image format".to_string(),
            ));
        }
    };

    Ok(image)
}

pub fn resize_rgb(img: DecodedImage, out_w: u32, out_h: u32) -> DecodedImage {
    if img.width == 0 || img.height == 0 {
        return img;
    }

    let (target_w, target_h) = resolve_output_size(img.width, img.height, out_w, out_h);
    if target_w == img.width && target_h == img.height {
        return img;
    }

    let filter = if target_w < img.width / 2 || target_h < img.height / 2 {
        fir::FilterType::Bilinear
    } else {
        fir::FilterType::CatmullRom
    };

    let src =
        fir::images::Image::from_vec_u8(img.width, img.height, img.pixels, fir::PixelType::U8x3)
            .expect("validated RGB buffer");
    let mut dst = fir::images::Image::new(target_w, target_h, fir::PixelType::U8x3);

    let options = fir::ResizeOptions::new().resize_alg(fir::ResizeAlg::Convolution(filter));
    fir::Resizer::new()
        .resize(&src, &mut dst, Some(&options))
        .expect("resize should succeed for RGB buffers");

    DecodedImage {
        pixels: dst.buffer().to_vec(),
        width: target_w,
        height: target_h,
    }
}

pub fn rotate_rgb(img: DecodedImage, degrees: u16) -> DecodedImage {
    match degrees % 360 {
        0 => img,
        180 => {
            let mut pixels = vec![0u8; img.pixels.len()];
            let stride = img.width as usize * 3;
            for row in 0..img.height as usize {
                let src = &img.pixels[(img.height as usize - 1 - row) * stride..][..stride];
                let dst = &mut pixels[row * stride..][..stride];
                for col in 0..img.width as usize {
                    dst[col * 3..col * 3 + 3]
                        .copy_from_slice(&src[(img.width as usize - 1 - col) * 3..][..3]);
                }
            }
            DecodedImage {
                pixels,
                width: img.width,
                height: img.height,
            }
        }
        90 => {
            let (ow, oh) = (img.width as usize, img.height as usize);
            let mut pixels = vec![0u8; img.pixels.len()];
            for row in 0..oh {
                for col in 0..ow {
                    let src = &img.pixels[(row * ow + col) * 3..][..3];
                    let dst_row = col;
                    let dst_col = oh - 1 - row;
                    pixels[(dst_row * oh + dst_col) * 3..][..3].copy_from_slice(src);
                }
            }
            DecodedImage {
                pixels,
                width: img.height,
                height: img.width,
            }
        }
        270 => {
            let (ow, oh) = (img.width as usize, img.height as usize);
            let mut pixels = vec![0u8; img.pixels.len()];
            for row in 0..oh {
                for col in 0..ow {
                    let src = &img.pixels[(row * ow + col) * 3..][..3];
                    let dst_row = ow - 1 - col;
                    let dst_col = row;
                    pixels[(dst_row * oh + dst_col) * 3..][..3].copy_from_slice(src);
                }
            }
            DecodedImage {
                pixels,
                width: img.height,
                height: img.width,
            }
        }
        _ => img,
    }
}

pub fn encode(
    img: DecodedImage,
    format: OutputFormat,
    quality: u8,
) -> Result<Vec<u8>, OxipixError> {
    let DecodedImage {
        pixels,
        width,
        height,
    } = img;
    let rgb = RgbImage::from_raw(width, height, pixels)
        .ok_or_else(|| OxipixError::Encode("invalid RGB buffer dimensions".to_string()))?;

    match format {
        OutputFormat::Jpeg => {
            let config = EncoderConfig::ycbcr(quality.min(100), ChromaSubsampling::Quarter);
            let mut enc = config
                .encode_from_bytes(rgb.width(), rgb.height(), ZenLayout::Rgb8Srgb)
                .map_err(|e| OxipixError::Encode(e.to_string()))?;
            enc.push_packed(rgb.as_raw(), Unstoppable)
                .map_err(|e| OxipixError::Encode(e.to_string()))?;
            enc.finish().map_err(|e| OxipixError::Encode(e.to_string()))
        }
        OutputFormat::Webp => fast_webp::encode_lossy_webp(&rgb, quality.min(100))
            .map_err(|e| OxipixError::Encode(e.to_string())),
        OutputFormat::Png => {
            let mut out = Vec::new();
            PngEncoder::new(&mut out)
                .write_image(
                    rgb.as_raw(),
                    rgb.width(),
                    rgb.height(),
                    ColorType::Rgb8.into(),
                )
                .map_err(|e| OxipixError::Encode(e.to_string()))?;
            Ok(out)
        }
        OutputFormat::Jxl => encode_jxl_rgb(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            &JxlEncodeOptions {
                distance: Some(distance_from_quality(quality)),
                ..JxlEncodeOptions::default()
            },
        ),
        OutputFormat::Avif => encode_avif_rgb(rgb.as_raw(), rgb.width(), rgb.height(), quality),
    }
}

pub fn encode_jxl_rgb(
    pixels: &[u8],
    width: u32,
    height: u32,
    options: &JxlEncodeOptions,
) -> Result<Vec<u8>, OxipixError> {
    if options.lossless {
        let mut config = LosslessConfig::new()
            .with_effort(options.effort)
            .with_mode(options.mode)
            .with_threads(options.threads);
        if let Some(v) = options.patches {
            config = config.with_patches(v);
        }
        if let Some(v) = options.lossless_tree_learning {
            config = config.with_tree_learning(v);
        }
        if let Some(v) = options.lossless_lz77 {
            config = config.with_lz77(v);
        }
        if let Some(v) = options.lossless_squeeze {
            config = config.with_squeeze(v);
        }
        config
            .encode(pixels, width, height, JxlLayout::Rgb8)
            .map_err(|e| OxipixError::Encode(e.to_string()))
    } else {
        let mut config = LossyConfig::new(options.distance.unwrap_or(1.0))
            .with_effort(options.effort)
            .with_mode(options.mode)
            .with_threads(options.threads);
        if let Some(v) = options.max_strategy_size {
            config = config.with_max_strategy_size(Some(v));
        }
        if let Some(v) = options.force_strategy {
            config = config.with_force_strategy(Some(v));
        }
        if let Some(v) = options.custom_orders {
            config = config.with_custom_orders(v);
        }
        if let Some(v) = options.adaptive_block_contexts {
            config = config.with_adaptive_block_contexts(v);
        }
        if let Some(v) = options.patches {
            config = config.with_patches(v);
        }
        if let Some(v) = options.gaborish {
            config = config.with_gaborish(v);
        }
        if let Some(v) = options.pixel_domain_loss {
            config = config.with_pixel_domain_loss(v);
        }
        if let Some(v) = options.adaptive_quant {
            config = config.with_adaptive_quant(v);
        }
        if let Some(v) = options.adjust_quant_ac {
            config = config.with_adjust_quant_ac(v);
        }
        if let Some(v) = options.chromacity_adjustment {
            config = config.with_chromacity_adjustment(v);
        }
        if let Some(v) = options.cfl {
            config = config.with_cfl(v);
        }
        if let Some(v) = options.cfl_two_pass {
            config = config.with_cfl_two_pass(v);
        }
        if let Some(v) = options.epf {
            config = config.with_epf(v);
        }
        if let Some(v) = options.epf_dynamic_sharpness {
            config = config.with_epf_dynamic_sharpness(v);
        }
        if let Some(v) = options.optimize_codes {
            config = config.with_optimize_codes(v);
        }
        config
            .encode(pixels, width, height, JxlLayout::Rgb8)
            .map_err(|e| OxipixError::Encode(e.to_string()))
    }
}

fn decode_jxl_any(
    bytes: &[u8],
    region: Option<(u32, u32, u32, u32)>,
    square: bool,
) -> Result<DecodedImage, OxipixError> {
    if let Some((x, y, w, h)) = region {
        return decode_jxl_region_from_bytes(bytes, x, y, w, h);
    }
    if square {
        let info = decode_jxl_info_from_bytes(bytes)?;
        let side = info.width.min(info.height);
        let left = (info.width - side) / 2;
        let top = (info.height - side) / 2;
        return decode_jxl_region_from_bytes(bytes, left, top, side, side);
    }
    decode_jxl_from_bytes(bytes)
}

fn decode_via_image(bytes: &[u8]) -> Result<DecodedImage, OxipixError> {
    let image = image::load_from_memory(bytes).map_err(|e| OxipixError::Decode(e.to_string()))?;
    let rgb = image.to_rgb8();
    Ok(DecodedImage {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    })
}

fn decode_svg(bytes: &[u8], out_w: u32, out_h: u32) -> Result<DecodedImage, OxipixError> {
    let options = usvg::Options::default();
    let tree =
        usvg::Tree::from_data(bytes, &options).map_err(|e| OxipixError::Decode(e.to_string()))?;

    let natural = tree.size().to_int_size();
    let (width, height) = resolve_output_size(natural.width(), natural.height(), out_w, out_h);
    let sx = width as f32 / natural.width() as f32;
    let sy = height as f32 / natural.height() as f32;
    let transform = tiny_skia::Transform::from_scale(sx, sy);

    let mut pixmap = tiny_skia::Pixmap::new(width, height)
        .ok_or_else(|| OxipixError::Decode("failed to allocate SVG raster surface".to_string()))?;
    resvg::render(&tree, transform, &mut pixmap.as_mut());

    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for rgba in pixmap.data().chunks_exact(4) {
        let a = rgba[3] as u32;
        if a == 0 {
            pixels.extend_from_slice(&[0, 0, 0]);
            continue;
        }
        let r = ((rgba[0] as u32 * 255) / a).min(255) as u8;
        let g = ((rgba[1] as u32 * 255) / a).min(255) as u8;
        let b = ((rgba[2] as u32 * 255) / a).min(255) as u8;
        pixels.extend_from_slice(&[r, g, b]);
    }

    Ok(DecodedImage {
        pixels,
        width,
        height,
    })
}

fn decode_svg_info(bytes: &[u8]) -> Result<ImageInfo, OxipixError> {
    let options = usvg::Options::default();
    let tree =
        usvg::Tree::from_data(bytes, &options).map_err(|e| OxipixError::Decode(e.to_string()))?;
    let size = tree.size().to_int_size();
    Ok(ImageInfo {
        width: size.width(),
        height: size.height(),
    })
}

#[cfg(feature = "pdf")]
fn decode_pdf_page(
    bytes: &[u8],
    page: u32,
    out_w: u32,
    out_h: u32,
) -> Result<DecodedImage, OxipixError> {
    let pdfium = Pdfium::default();
    let document = pdfium
        .load_pdf_from_byte_vec(bytes.to_vec(), None)
        .map_err(|e| OxipixError::Decode(e.to_string()))?;
    let page = document
        .pages()
        .get(page as u16)
        .map_err(|e| OxipixError::Decode(e.to_string()))?;

    let mut config = PdfRenderConfig::new();
    if out_w > 0 {
        config = config.set_target_width(out_w as i32);
    }
    if out_h > 0 {
        config = config.set_target_height(out_h as i32);
    }

    let image = page
        .render_with_config(&config)
        .map_err(|e| OxipixError::Decode(e.to_string()))?
        .as_image();
    let rgb = image.to_rgb8();
    Ok(DecodedImage {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    })
}

fn decode_tiff_page(bytes: &[u8], page: u32) -> Result<DecodedImage, OxipixError> {
    let cursor = Cursor::new(bytes);
    let reader = BufReader::new(cursor);
    let mut decoder = TiffDecoder::new(reader).map_err(|e| OxipixError::Decode(e.to_string()))?;

    for _ in 0..page {
        if decoder.more_images() {
            decoder
                .next_image()
                .map_err(|e| OxipixError::Decode(e.to_string()))?;
        } else {
            return Err(OxipixError::Decode(format!(
                "TIFF page {page} out of range"
            )));
        }
    }

    let (width, height) = decoder
        .dimensions()
        .map_err(|e| OxipixError::Decode(e.to_string()))?;
    let color = decoder
        .colortype()
        .map_err(|e| OxipixError::Decode(e.to_string()))?;
    let image = decoder
        .read_image()
        .map_err(|e| OxipixError::Decode(e.to_string()))?;

    tiff_to_rgb(image, color, width, height)
}

fn tiff_to_rgb(
    image: DecodingResult,
    color: tiff::ColorType,
    width: u32,
    height: u32,
) -> Result<DecodedImage, OxipixError> {
    match (image, color) {
        (DecodingResult::U8(data), tiff::ColorType::Gray(8)) => {
            let mut pixels = Vec::with_capacity((width * height * 3) as usize);
            for value in data {
                pixels.extend_from_slice(&[value, value, value]);
            }
            Ok(DecodedImage {
                pixels,
                width,
                height,
            })
        }
        (DecodingResult::U16(data), tiff::ColorType::Gray(16)) => {
            let mut pixels = Vec::with_capacity((width * height * 3) as usize);
            for value in data {
                let gray = (value >> 8) as u8;
                pixels.extend_from_slice(&[gray, gray, gray]);
            }
            Ok(DecodedImage {
                pixels,
                width,
                height,
            })
        }
        (DecodingResult::U8(data), tiff::ColorType::RGB(8)) => Ok(DecodedImage {
            pixels: data,
            width,
            height,
        }),
        (DecodingResult::U16(data), tiff::ColorType::RGB(16)) => {
            let pixels = data.into_iter().map(|v| (v >> 8) as u8).collect();
            Ok(DecodedImage {
                pixels,
                width,
                height,
            })
        }
        (DecodingResult::U8(data), tiff::ColorType::RGBA(8)) => {
            let mut pixels = Vec::with_capacity((width * height * 3) as usize);
            for rgba in data.chunks_exact(4) {
                pixels.extend_from_slice(&rgba[..3]);
            }
            Ok(DecodedImage {
                pixels,
                width,
                height,
            })
        }
        (DecodingResult::U16(data), tiff::ColorType::RGBA(16)) => {
            let mut pixels = Vec::with_capacity((width * height * 3) as usize);
            for rgba in data.chunks_exact(4) {
                pixels.extend_from_slice(&[
                    (rgba[0] >> 8) as u8,
                    (rgba[1] >> 8) as u8,
                    (rgba[2] >> 8) as u8,
                ]);
            }
            Ok(DecodedImage {
                pixels,
                width,
                height,
            })
        }
        _ => Err(OxipixError::Decode(
            "unsupported TIFF pixel format".to_string(),
        )),
    }
}

fn planar_to_rgb(
    pixel_format: PixelFormat,
    r: &[f32],
    g: Option<&[f32]>,
    b: Option<&[f32]>,
    width: u32,
    height: u32,
) -> DecodedImage {
    #[inline(always)]
    fn f32_to_u8(v: f32) -> u8 {
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    }

    let mut pixels = vec![0u8; (width * height * 3) as usize];
    match pixel_format {
        PixelFormat::Gray | PixelFormat::Graya => {
            for (chunk, &value) in pixels.chunks_exact_mut(3).zip(r.iter()) {
                let gray = f32_to_u8(value);
                chunk[0] = gray;
                chunk[1] = gray;
                chunk[2] = gray;
            }
        }
        PixelFormat::Rgb | PixelFormat::Rgba | PixelFormat::Cmyk | PixelFormat::Cmyka => {
            let g = g.expect("green plane present for RGB-like JXL image");
            let b = b.expect("blue plane present for RGB-like JXL image");
            for (chunk, ((rv, gv), bv)) in pixels
                .chunks_exact_mut(3)
                .zip(r.iter().zip(g.iter()).zip(b.iter()))
            {
                chunk[0] = f32_to_u8(*rv);
                chunk[1] = f32_to_u8(*gv);
                chunk[2] = f32_to_u8(*bv);
            }
        }
    }

    DecodedImage {
        pixels,
        width,
        height,
    }
}

fn apply_post_decode_ops(
    img: DecodedImage,
    region: Option<(u32, u32, u32, u32)>,
    square: bool,
) -> DecodedImage {
    if let Some((x, y, w, h)) = region {
        return apply_region(img, x, y, w, h);
    }
    if square {
        return square_crop(img);
    }
    img
}

fn apply_region(img: DecodedImage, x: u32, y: u32, w: u32, h: u32) -> DecodedImage {
    if w == 0 || h == 0 {
        return img;
    }

    let start_x = x.min(img.width);
    let start_y = y.min(img.height);
    let end_x = start_x.saturating_add(w).min(img.width);
    let end_y = start_y.saturating_add(h).min(img.height);
    let crop_w = end_x.saturating_sub(start_x);
    let crop_h = end_y.saturating_sub(start_y);

    if crop_w == 0 || crop_h == 0 {
        return DecodedImage {
            pixels: Vec::new(),
            width: 0,
            height: 0,
        };
    }

    if crop_w == img.width && crop_h == img.height && start_x == 0 && start_y == 0 {
        return img;
    }

    let src_stride = img.width as usize * 3;
    let dst_stride = crop_w as usize * 3;
    let mut pixels = vec![0u8; (crop_w * crop_h * 3) as usize];

    for row in 0..crop_h as usize {
        let src_offset = (start_y as usize + row) * src_stride + start_x as usize * 3;
        let dst_offset = row * dst_stride;
        pixels[dst_offset..dst_offset + dst_stride]
            .copy_from_slice(&img.pixels[src_offset..src_offset + dst_stride]);
    }

    DecodedImage {
        pixels,
        width: crop_w,
        height: crop_h,
    }
}

fn square_crop(img: DecodedImage) -> DecodedImage {
    let side = img.width.min(img.height);
    let x = (img.width - side) / 2;
    let y = (img.height - side) / 2;
    apply_region(img, x, y, side, side)
}

fn encode_avif_rgb(
    pixels: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, OxipixError> {
    #[cfg(feature = "avif")]
    {
        let pixels: Vec<AvifRgb8> = pixels
            .chunks_exact(3)
            .map(|chunk| AvifRgb8::new(chunk[0], chunk[1], chunk[2]))
            .collect();
        let encoded = AvifEncoder::new()
            .with_quality(quality.clamp(1, 100) as f32)
            .with_speed(6)
            .encode_rgb(AvifImg::new(
                pixels.as_slice(),
                width as usize,
                height as usize,
            ))
            .map_err(|e| OxipixError::Encode(e.to_string()))?;
        Ok(encoded.avif_file)
    }
    #[cfg(not(feature = "avif"))]
    {
        let _ = (pixels, width, height, quality);
        Err(OxipixError::Encode(
            "AVIF output not compiled in (enable the `avif` feature)".to_string(),
        ))
    }
}

fn resolve_output_size(src_w: u32, src_h: u32, out_w: u32, out_h: u32) -> (u32, u32) {
    match (out_w, out_h) {
        (0, 0) => (src_w, src_h),
        (w, 0) => {
            let h = ((src_h as u64 * w as u64) / src_w.max(1) as u64).max(1) as u32;
            (w.max(1), h)
        }
        (0, h) => {
            let w = ((src_w as u64 * h as u64) / src_h.max(1) as u64).max(1) as u32;
            (w, h.max(1))
        }
        (w, h) => (w.max(1), h.max(1)),
    }
}

fn distance_from_quality(quality: u8) -> f32 {
    let quality = quality.clamp(1, 100) as f32;
    (100.0 - quality) / 25.0 + 0.5
}
