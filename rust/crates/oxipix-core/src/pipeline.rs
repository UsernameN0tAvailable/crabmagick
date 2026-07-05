use std::fs;
use std::io::Cursor;

use fast_image_resize as fir;
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder, RgbImage};
use jxl_encoder::{EncoderMode, LosslessConfig, LossyConfig, PixelLayout as JxlLayout};
use jxl_oxide::{CropInfo, JxlImage, PixelFormat};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout as ZenLayout, Unstoppable};

use crate::cache::DecodedImage;
use crate::processor::{ImageInfo, OutputFormat, OxipixError};

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

    let width = image.width();
    let height = image.height();
    let planes = frame.image_planar();
    let pixel_count = (width * height) as usize;

    #[inline(always)]
    fn f32_to_u8(v: f32) -> u8 {
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    }

    let mut pixels = vec![0u8; pixel_count * 3];
    match image.pixel_format() {
        PixelFormat::Gray | PixelFormat::Graya => {
            for (chunk, &value) in pixels.chunks_exact_mut(3).zip(planes[0].buf().iter()) {
                let gray = f32_to_u8(value);
                chunk[0] = gray;
                chunk[1] = gray;
                chunk[2] = gray;
            }
        }
        PixelFormat::Rgb | PixelFormat::Rgba | PixelFormat::Cmyk | PixelFormat::Cmyka => {
            let r = planes[0].buf();
            let g = planes[1].buf();
            let b = planes[2].buf();
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

    Ok(DecodedImage {
        pixels,
        width,
        height,
    })
}

/// Decode only the specified region of a JXL file. jxl-oxide skips groups
/// outside the crop window, making this much faster for small tile requests.
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

    image.set_image_region(CropInfo { left, top, width, height });

    let frame = image
        .render_frame(0)
        .map_err(|e| OxipixError::Decode(e.to_string()))?;

    let planes = frame.image_planar();
    if planes.is_empty() {
        return Err(OxipixError::Decode("no planes decoded".to_string()));
    }
    let actual_w = planes[0].width() as u32;
    let actual_h = planes[0].height() as u32;
    let pixel_count = (actual_w * actual_h) as usize;

    #[inline(always)]
    fn f32_to_u8(v: f32) -> u8 {
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    }

    let mut pixels = vec![0u8; pixel_count * 3];
    match image.pixel_format() {
        PixelFormat::Gray | PixelFormat::Graya => {
            for (chunk, &value) in pixels.chunks_exact_mut(3).zip(planes[0].buf().iter()) {
                let gray = f32_to_u8(value);
                chunk[0] = gray;
                chunk[1] = gray;
                chunk[2] = gray;
            }
        }
        PixelFormat::Rgb | PixelFormat::Rgba | PixelFormat::Cmyk | PixelFormat::Cmyka => {
            let r = planes[0].buf();
            let g = planes[1].buf();
            let b = planes[2].buf();
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

    Ok(DecodedImage {
        pixels,
        width: actual_w,
        height: actual_h,
    })
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

pub fn apply_region(img: &DecodedImage, x: u32, y: u32, w: u32, h: u32) -> DecodedImage {
    if w == 0 || h == 0 {
        return DecodedImage {
            pixels: img.pixels.clone(),
            width: img.width,
            height: img.height,
        };
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
        return DecodedImage {
            pixels: img.pixels.clone(),
            width: img.width,
            height: img.height,
        };
    }

    let mut pixels = vec![0u8; (crop_w * crop_h * 3) as usize];
    let src_stride = img.width as usize * 3;
    let dst_stride = crop_w as usize * 3;

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

pub fn resize_rgb(img: &DecodedImage, out_w: u32, out_h: u32) -> DecodedImage {
    if img.width == 0 || img.height == 0 {
        return DecodedImage {
            pixels: Vec::new(),
            width: 0,
            height: 0,
        };
    }

    let (target_w, target_h) = resolve_output_size(img.width, img.height, out_w, out_h);
    if target_w == img.width && target_h == img.height {
        return DecodedImage {
            pixels: img.pixels.clone(),
            width: img.width,
            height: img.height,
        };
    }

    let src = fir::images::Image::from_vec_u8(
        img.width,
        img.height,
        img.pixels.clone(),
        fir::PixelType::U8x3,
    )
    .expect("validated RGB buffer");

    let mut dst = fir::images::Image::new(target_w, target_h, fir::PixelType::U8x3);
    let filter = if target_w < img.width / 2 || target_h < img.height / 2 {
        fir::FilterType::Bilinear
    } else {
        fir::FilterType::CatmullRom
    };

    let options = fir::ResizeOptions::new().resize_alg(fir::ResizeAlg::Convolution(filter));
    let mut resizer = fir::Resizer::new();
    resizer
        .resize(&src, &mut dst, Some(&options))
        .expect("resize should succeed for RGB buffers");

    DecodedImage {
        pixels: dst.buffer().to_vec(),
        width: target_w,
        height: target_h,
    }
}

/// Crop the largest centred square from the image (IIIF "square" region).
pub fn square_crop(img: &DecodedImage) -> DecodedImage {
    let side = img.width.min(img.height);
    let x = (img.width - side) / 2;
    let y = (img.height - side) / 2;
    apply_region(img, x, y, side, side)
}

/// Rotate an RGB image clockwise by 0 / 90 / 180 / 270 degrees.
pub fn rotate_rgb(img: &DecodedImage, degrees: u16) -> DecodedImage {
    match degrees % 360 {
        0 => DecodedImage { pixels: img.pixels.clone(), width: img.width, height: img.height },
        180 => {
            let mut pixels = vec![0u8; img.pixels.len()];
            let stride = img.width as usize * 3;
            for row in 0..img.height as usize {
                let src = &img.pixels[(img.height as usize - 1 - row) * stride..][..stride];
                let dst = &mut pixels[row * stride..][..stride];
                for col in 0..img.width as usize {
                    dst[col * 3..col * 3 + 3].copy_from_slice(&src[(img.width as usize - 1 - col) * 3..][..3]);
                }
            }
            DecodedImage { pixels, width: img.width, height: img.height }
        }
        90 => {
            // 90° clockwise: new_w = old_h, new_h = old_w
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
            DecodedImage { pixels, width: img.height, height: img.width }
        }
        270 => {
            // 270° clockwise (= 90° CCW): new_w = old_h, new_h = old_w
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
            DecodedImage { pixels, width: img.height, height: img.width }
        }
        _ => DecodedImage { pixels: img.pixels.clone(), width: img.width, height: img.height },
    }
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

pub fn encode(
    img: &DecodedImage,
    format: OutputFormat,
    quality: u8,
) -> Result<Vec<u8>, OxipixError> {
    let rgb = RgbImage::from_raw(img.width, img.height, img.pixels.clone())
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
            let encoder = PngEncoder::new(&mut out);
            encoder
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
        if let Some(patches) = options.patches {
            config = config.with_patches(patches);
        }
        if let Some(tree_learning) = options.lossless_tree_learning {
            config = config.with_tree_learning(tree_learning);
        }
        if let Some(lz77) = options.lossless_lz77 {
            config = config.with_lz77(lz77);
        }
        if let Some(squeeze) = options.lossless_squeeze {
            config = config.with_squeeze(squeeze);
        }
        config
            .encode(pixels, width, height, JxlLayout::Rgb8)
            .map_err(|e| OxipixError::Encode(e.to_string()))
    } else {
        let mut config = LossyConfig::new(options.distance.unwrap_or(1.0))
            .with_effort(options.effort)
            .with_mode(options.mode)
            .with_threads(options.threads);
        if let Some(max_strategy_size) = options.max_strategy_size {
            config = config.with_max_strategy_size(Some(max_strategy_size));
        }
        if let Some(force_strategy) = options.force_strategy {
            config = config.with_force_strategy(Some(force_strategy));
        }
        if let Some(custom_orders) = options.custom_orders {
            config = config.with_custom_orders(custom_orders);
        }
        if let Some(adaptive_block_contexts) = options.adaptive_block_contexts {
            config = config.with_adaptive_block_contexts(adaptive_block_contexts);
        }
        if let Some(patches) = options.patches {
            config = config.with_patches(patches);
        }
        if let Some(gaborish) = options.gaborish {
            config = config.with_gaborish(gaborish);
        }
        if let Some(pixel_domain_loss) = options.pixel_domain_loss {
            config = config.with_pixel_domain_loss(pixel_domain_loss);
        }
        if let Some(adaptive_quant) = options.adaptive_quant {
            config = config.with_adaptive_quant(adaptive_quant);
        }
        if let Some(adjust_quant_ac) = options.adjust_quant_ac {
            config = config.with_adjust_quant_ac(adjust_quant_ac);
        }
        if let Some(chromacity_adjustment) = options.chromacity_adjustment {
            config = config.with_chromacity_adjustment(chromacity_adjustment);
        }
        if let Some(cfl) = options.cfl {
            config = config.with_cfl(cfl);
        }
        if let Some(cfl_two_pass) = options.cfl_two_pass {
            config = config.with_cfl_two_pass(cfl_two_pass);
        }
        if let Some(epf) = options.epf {
            config = config.with_epf(epf);
        }
        if let Some(epf_dynamic_sharpness) = options.epf_dynamic_sharpness {
            config = config.with_epf_dynamic_sharpness(epf_dynamic_sharpness);
        }
        if let Some(optimize_codes) = options.optimize_codes {
            config = config.with_optimize_codes(optimize_codes);
        }
        config
            .encode(pixels, width, height, JxlLayout::Rgb8)
            .map_err(|e| OxipixError::Encode(e.to_string()))
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
