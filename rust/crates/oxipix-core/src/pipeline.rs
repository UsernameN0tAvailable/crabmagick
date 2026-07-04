use std::fs;
use std::io::Cursor;

use fast_image_resize as fir;
use image::{ColorType, ImageEncoder, RgbImage, codecs::png::PngEncoder};
use jxl_encoder::{LossyConfig, PixelLayout as JxlLayout};
use jxl_oxide::{JxlImage, PixelFormat};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout as ZenLayout, Unstoppable};

use crate::cache::DecodedImage;
use crate::processor::{ImageInfo, OutputFormat, OxipixError};

pub fn decode_jxl(path: &str) -> Result<DecodedImage, OxipixError> {
    let bytes = fs::read(path)?;
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

pub fn decode_jxl_info(path: &str) -> Result<ImageInfo, OxipixError> {
    let bytes = fs::read(path)?;
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
            enc.finish()
                .map_err(|e| OxipixError::Encode(e.to_string()))
        }
        OutputFormat::Webp => fast_webp::encode_lossy_webp(&rgb, quality.min(100))
            .map_err(|e| OxipixError::Encode(e.to_string())),
        OutputFormat::Png => {
            let mut out = Vec::new();
            let encoder = PngEncoder::new(&mut out);
            encoder
                .write_image(rgb.as_raw(), rgb.width(), rgb.height(), ColorType::Rgb8.into())
                .map_err(|e| OxipixError::Encode(e.to_string()))?;
            Ok(out)
        }
        OutputFormat::Jxl => LossyConfig::new(distance_from_quality(quality))
            .with_effort(5)
            .encode(rgb.as_raw(), rgb.width(), rgb.height(), JxlLayout::Rgb8)
            .map_err(|e| OxipixError::Encode(e.to_string())),
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
