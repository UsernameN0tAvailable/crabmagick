//! Low-level decode, transform, and encode primitives used by CrabMagick.

// Allow cfg-guarded returns and chunk patterns that can't use unstable as_chunks.
#![allow(clippy::needless_return, clippy::collapsible_if, clippy::chunks_exact_to_as_chunks)]

use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::sync::{Arc, Mutex, OnceLock};

use fast_image_resize as fir;
use image::{codecs::png::PngEncoder, ColorType, GenericImageView, ImageEncoder, RgbImage};
use crate::jxl_oxide_vendored::jxl_oxide::{CropInfo, JxlImage, PixelFormat};
use lru::LruCache;
use memmap2::Mmap;
#[cfg(feature = "pdf")]
use pdfium_render::prelude::*;
#[cfg(feature = "avif")]
use ravif::{Encoder as AvifEncoder, Img as AvifImg, RGB8 as AvifRgb8};
use rayon::prelude::*;
use resvg::{tiny_skia, usvg};
use tiff::decoder::{Decoder as TiffDecoder, DecodingResult};

use crate::jxl_encoder::{EncoderMode, LosslessConfig, LossyConfig, PixelLayout as JxlLayout};
use crate::processor::{CrabMagickError, ImageInfo, OutputFormat, RequestedRegion};
use crate::zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig, PixelLayout as ZenLayout, Unstoppable,
};

// ── Core image and cache types ───────────────────────────────────────────────

/// Owned RGB pixel data produced by the decoding pipeline.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    /// Packed `RGBRGB...` pixel bytes.
    pub pixels: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

type CacheKey = (String, u32);

const SMALL_FILE_READ_THRESHOLD: u64 = 1_024 * 1_024;
const AVERAGE_DECODED_IMAGE_BYTES: usize = 4 * 1_024 * 1_024;

/// File-backed or owned input bytes used during decode.
pub(crate) enum FileBytes {
    Mmap(Mmap),
    Owned(Vec<u8>),
}

impl Deref for FileBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mmap(map) => map,
            Self::Owned(bytes) => bytes,
        }
    }
}

struct CachedDecodedImage {
    image: Arc<DecodedImage>,
    bytes: usize,
}

struct DecodedImageCache {
    entries: LruCache<CacheKey, CachedDecodedImage>,
    bytes_limit: usize,
    current_bytes: usize,
}

impl DecodedImageCache {
    fn new(bytes_limit: usize) -> Self {
        let estimated_items = bytes_limit
            .saturating_div(AVERAGE_DECODED_IMAGE_BYTES)
            .max(1);
        let capacity = NonZeroUsize::new(estimated_items).expect("cache capacity is non-zero");
        Self {
            entries: LruCache::new(capacity),
            bytes_limit,
            current_bytes: 0,
        }
    }

    fn reset(&mut self, bytes_limit: usize) {
        *self = Self::new(bytes_limit);
    }

    fn get(&mut self, key: &CacheKey) -> Option<Arc<DecodedImage>> {
        if self.bytes_limit == 0 {
            return None;
        }
        self.entries.get(key).map(|entry| Arc::clone(&entry.image))
    }

    fn put(&mut self, key: CacheKey, image: Arc<DecodedImage>) {
        if self.bytes_limit == 0 {
            return;
        }

        let bytes = image.pixels.len();
        if bytes > self.bytes_limit {
            return;
        }

        if let Some(previous) = self.entries.put(key, CachedDecodedImage { image, bytes }) {
            self.current_bytes = self.current_bytes.saturating_sub(previous.bytes);
        }
        self.current_bytes = self.current_bytes.saturating_add(bytes);

        while self.current_bytes > self.bytes_limit {
            let Some((_, evicted)) = self.entries.pop_lru() else {
                break;
            };
            self.current_bytes = self.current_bytes.saturating_sub(evicted.bytes);
        }
    }
}

static DECODED_IMAGE_CACHE: OnceLock<Mutex<DecodedImageCache>> = OnceLock::new();

/// Source format detected from file signatures and lightweight probes.
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

// ── Cache helpers ────────────────────────────────────────────────────────────

fn decoded_image_cache() -> &'static Mutex<DecodedImageCache> {
    DECODED_IMAGE_CACHE.get_or_init(|| Mutex::new(DecodedImageCache::new(0)))
}

fn cacheable_full_decode(format: SourceFormat, region: Option<(u32, u32, u32, u32)>, square: bool) -> bool {
    region.is_none()
        && !square
        && matches!(
            format,
            SourceFormat::Jxl
                | SourceFormat::Jpeg
                | SourceFormat::Png
                | SourceFormat::Webp
                | SourceFormat::Tiff
                | SourceFormat::Gif
                | SourceFormat::Bmp
                | SourceFormat::Avif
        )
}

fn cached_full_decode(path: &str, page: u32) -> Option<DecodedImage> {
    let key = (path.to_owned(), page);
    decoded_image_cache()
        .lock()
        .expect("decoded image cache lock poisoned")
        .get(&key)
        .map(|image| (*image).clone())
}

fn store_cached_full_decode(path: &str, page: u32, image: &DecodedImage) {
    let key = (path.to_owned(), page);
    decoded_image_cache()
        .lock()
        .expect("decoded image cache lock poisoned")
        .put(key, Arc::new(image.clone()));
}

#[inline]
pub(crate) fn init_decoded_image_cache(tile_cache_mb: u64) {
    let bytes_limit = tile_cache_mb
        .saturating_mul(1024)
        .saturating_mul(1024)
        .min(usize::MAX as u64) as usize;
    decoded_image_cache()
        .lock()
        .expect("decoded image cache lock poisoned")
        .reset(bytes_limit);
}

#[inline]
pub(crate) fn read_file_bytes(path: &str) -> Result<FileBytes, CrabMagickError> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if metadata.len() > SMALL_FILE_READ_THRESHOLD {
        // SAFETY: The mapping is read-only and the file handle lives until after the map is
        // created. The returned Mmap owns the OS mapping independently of the File.
        let mapped = unsafe { Mmap::map(&file)? };
        Ok(FileBytes::Mmap(mapped))
    } else {
        let mut reader = file;
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        reader.read_to_end(&mut bytes)?;
        Ok(FileBytes::Owned(bytes))
    }
}

// ── JPEG XL encode configuration ─────────────────────────────────────────────

/// Fine-grained JPEG XL encoder configuration for [`encode_jxl_rgb`].
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

// ── Error helpers ────────────────────────────────────────────────────────────

fn decode_malformed(message: impl Into<String>) -> CrabMagickError {
    CrabMagickError::decode_malformed(message)
}

fn decode_unsupported_format(message: impl Into<String>) -> CrabMagickError {
    CrabMagickError::decode_unsupported_format(message)
}

fn encode_error(message: impl Into<String>) -> CrabMagickError {
    CrabMagickError::encode_error(message)
}

fn region_out_of_bounds(region: RequestedRegion, image: ImageInfo) -> CrabMagickError {
    CrabMagickError::region_out_of_bounds(region, image)
}

// ── Format detection ─────────────────────────────────────────────────────────

/// Detects the source format from magic bytes and lightweight textual probes.
///
/// # Performance
///
/// This function reads only the leading bytes needed for signature matching and
/// falls back to a small UTF-8 probe for SVG detection.
#[inline]
#[must_use]
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

// ── Decode entry points ──────────────────────────────────────────────────────

/// Decodes a full JPEG XL image from disk into packed RGB pixels.
///
/// # Performance
///
/// Large files are memory-mapped, and the decode stays inside the bundled JXL
/// path without involving the generic `image` crate.
pub fn decode_jxl(path: &str) -> Result<DecodedImage, CrabMagickError> {
    let bytes = read_file_bytes(path)?;
    decode_jxl_from_bytes(&bytes)
}

/// Decodes a full JPEG XL image from memory into packed RGB pixels.
///
/// # Performance
///
/// This is the lowest-overhead full-image JXL decode path exposed publicly.
pub fn decode_jxl_from_bytes(bytes: &[u8]) -> Result<DecodedImage, CrabMagickError> {
    decode_jxl_from_bytes_with_hint(bytes, None)
}

fn decode_jxl_from_bytes_with_hint(
    bytes: &[u8],
    render_size: Option<(u32, u32)>,
) -> Result<DecodedImage, CrabMagickError> {
    let image = create_jxl_image(bytes, render_size)?;

    let frame = image
        .render_frame(0)
        .map_err(|error| decode_malformed(format!("JPEG XL frame rendering failed: {error}")))?;
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

/// Decodes a rectangular JPEG XL region from disk into packed RGB pixels.
///
/// # Performance
///
/// Uses `jxl-oxide` region decode so only the requested window is rendered.
pub fn decode_jxl_region(
    path: &str,
    left: u32,
    top: u32,
    width: u32,
    height: u32,
) -> Result<DecodedImage, CrabMagickError> {
    let bytes = read_file_bytes(path)?;
    decode_jxl_region_from_bytes(&bytes, left, top, width, height)
}

/// Decodes a rectangular JPEG XL region from memory into packed RGB pixels.
///
/// # Performance
///
/// Uses `jxl-oxide` region decode so only the requested window is rendered.
pub fn decode_jxl_region_from_bytes(
    bytes: &[u8],
    left: u32,
    top: u32,
    width: u32,
    height: u32,
) -> Result<DecodedImage, CrabMagickError> {
    decode_jxl_region_from_bytes_with_hint(bytes, left, top, width, height, None)
}

fn decode_jxl_region_from_bytes_with_hint(
    bytes: &[u8],
    left: u32,
    top: u32,
    width: u32,
    height: u32,
    render_size: Option<(u32, u32)>,
) -> Result<DecodedImage, CrabMagickError> {
    let mut image = create_jxl_image(bytes, render_size)?;
    let image_info = ImageInfo {
        width: image.width(),
        height: image.height(),
    };
    let region = RequestedRegion::new(left, top, width, height);
    validate_requested_region(region, image_info)?;

    image.set_image_region(CropInfo {
        left,
        top,
        width,
        height,
    });

    let frame = image
        .render_frame(0)
        .map_err(|error| decode_malformed(format!("JPEG XL region rendering failed: {error}")))?;
    let planes = frame.image_planar();
    if planes.is_empty() {
        return Err(decode_malformed(
            "JPEG XL decode produced no image planes for the requested region",
        ));
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

/// Returns JPEG XL dimensions without fully rasterizing the image.
pub fn decode_jxl_info(path: &str) -> Result<ImageInfo, CrabMagickError> {
    let bytes = read_file_bytes(path)?;
    decode_jxl_info_from_bytes(&bytes)
}

/// Returns JPEG XL dimensions from in-memory bytes.
pub fn decode_jxl_info_from_bytes(bytes: &[u8]) -> Result<ImageInfo, CrabMagickError> {
    let image = create_jxl_image(bytes, None)?;

    Ok(ImageInfo {
        width: image.width(),
        height: image.height(),
    })
}

fn create_jxl_image(
    bytes: &[u8],
    render_size: Option<(u32, u32)>,
) -> Result<JxlImage, CrabMagickError> {
    let _ = render_size;
    // jxl-oxide 0.12.6 exposes region cropping but does not currently expose a render-size or
    // decode-scale hint on JxlImage/JxlImageBuilder, so we keep the requested size only as a
    // future integration hook.
    JxlImage::builder()
        .read(Cursor::new(bytes))
        .map_err(|error| decode_malformed(format!("JPEG XL header parsing failed: {error}")))
}

/// Decodes any supported source format into packed RGB pixels.
///
/// # Performance
///
/// Delegates to optimized format-specific decoders and keeps region handling in
/// the lowest-cost path available for each codec.
pub fn decode_any(
    path: &str,
    region: Option<(u32, u32, u32, u32)>,
    square: bool,
) -> Result<DecodedImage, CrabMagickError> {
    decode_any_with_options(path, region, square, 0, None)
}

/// Returns source-image dimensions for any supported format.
pub fn decode_any_info(path: &str, page: u32) -> Result<ImageInfo, CrabMagickError> {
    let bytes = read_file_bytes(path)?;
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
                return Err(decode_unsupported_format(
                    "PDF decoding is unavailable in this build; enable the `pdf` feature",
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
                return Err(decode_unsupported_format(
                    "AVIF decoding is unavailable in this build; enable the `avif` feature",
                ));
            }
        }
        SourceFormat::Jpeg
        | SourceFormat::Png
        | SourceFormat::Webp
        | SourceFormat::Tiff
        | SourceFormat::Gif
        | SourceFormat::Bmp => {
            let image = image::load_from_memory(&bytes)
                .map_err(|error| decode_malformed(format!("source image decode failed: {error}")))?;
            let (width, height) = image.dimensions();
            Ok(ImageInfo { width, height })
        }
        SourceFormat::Unknown => Err(decode_unsupported_format(
            "file signature did not match any supported image format",
        )),
    }
}

/// Decodes any supported source format with optional region and render hints.
///
/// # Performance
///
/// - Reuses the shared decoded-image LRU for cacheable full-image requests.
/// - Routes JPEG and WebP through bundled fast decoders.
/// - Sends JXL region requests directly to the partial-decode path.
#[doc(hidden)]
pub fn decode_any_with_options(
    path: &str,
    region: Option<(u32, u32, u32, u32)>,
    square: bool,
    page: u32,
    render_size: Option<(u32, u32)>,
) -> Result<DecodedImage, CrabMagickError> {
    if region.is_none() && !square {
        if let Some(image) = cached_full_decode(path, page) {
            return Ok(image);
        }
    }

    let bytes = read_file_bytes(path)?;
    let format = detect_format(&bytes);

    let image = match format {
        SourceFormat::Jxl => {
            if let Some((x, y, w, h)) = region {
                decode_jxl_region_from_bytes_with_hint(&bytes, x, y, w, h, render_size)?
            } else if square {
                let info = decode_jxl_info_from_bytes(&bytes)?;
                let side = info.width.min(info.height);
                let left = (info.width - side) / 2;
                let top = (info.height - side) / 2;
                decode_jxl_region_from_bytes_with_hint(&bytes, left, top, side, side, render_size)?
            } else {
                decode_jxl_from_bytes_with_hint(&bytes, render_size)?
            }
        }
        SourceFormat::Jpeg => {
            let decoded = decode_jpeg_fast(&bytes).or_else(|_| decode_via_image(&bytes))?;
            apply_post_decode_ops(decoded, region, square)?
        }
        SourceFormat::Png | SourceFormat::Gif | SourceFormat::Bmp => {
            apply_post_decode_ops(decode_via_image(&bytes)?, region, square)?
        }
        SourceFormat::Webp => apply_post_decode_ops(decode_webp_fast(&bytes)?, region, square)?,
        SourceFormat::Tiff => {
            apply_post_decode_ops(decode_tiff_page(&bytes, page)?, region, square)?
        }
        SourceFormat::Svg => {
            let (out_w, out_h) = render_size.unwrap_or((0, 0));
            apply_post_decode_ops(decode_svg(&bytes, out_w, out_h)?, region, square)?
        }
        SourceFormat::Pdf => {
            #[cfg(feature = "pdf")]
            {
                let (out_w, out_h) = render_size.unwrap_or((0, 0));
                apply_post_decode_ops(decode_pdf_page(&bytes, page, out_w, out_h)?, region, square)?
            }
            #[cfg(not(feature = "pdf"))]
            {
                return Err(decode_unsupported_format(
                    "PDF decoding is unavailable in this build; enable the `pdf` feature",
                ));
            }
        }
        SourceFormat::Avif => {
            #[cfg(feature = "avif")]
            {
                apply_post_decode_ops(decode_via_image(&bytes)?, region, square)?
            }
            #[cfg(not(feature = "avif"))]
            {
                return Err(decode_unsupported_format(
                    "AVIF decoding is unavailable in this build; enable the `avif` feature",
                ));
            }
        }
        SourceFormat::Unknown => {
            return Err(decode_unsupported_format(
                "file signature did not match any supported image format",
            ));
        }
    };

    if cacheable_full_decode(format, region, square) {
        store_cached_full_decode(path, page, &image);
    }

    Ok(image)
}

// ── Image transforms ─────────────────────────────────────────────────────────

/// Resizes a packed RGB image while preserving aspect ratio when one dimension is `0`.
///
/// # Performance
///
/// Uses `fast_image_resize` with a downscale-friendly filter choice and avoids
/// work when the output size matches the input size.
#[inline]
#[must_use]
pub fn resize_rgb(
    image: DecodedImage,
    output_width: u32,
    output_height: u32,
) -> DecodedImage {
    if image.width == 0 || image.height == 0 {
        return image;
    }

    let (target_width, target_height) =
        resolve_output_size(image.width, image.height, output_width, output_height);
    if target_width == image.width && target_height == image.height {
        return image;
    }

    let filter = if target_width < image.width / 2 || target_height < image.height / 2 {
        fir::FilterType::Bilinear
    } else {
        fir::FilterType::CatmullRom
    };

    let source = fir::images::Image::from_vec_u8(
        image.width,
        image.height,
        image.pixels,
        fir::PixelType::U8x3,
    )
    .expect("validated RGB buffer");
    let mut destination =
        fir::images::Image::new(target_width, target_height, fir::PixelType::U8x3);

    let options = fir::ResizeOptions::new().resize_alg(fir::ResizeAlg::Convolution(filter));
    fir::Resizer::new()
        .resize(&source, &mut destination, Some(&options))
        .expect("resize should succeed for RGB buffers");

    DecodedImage {
        pixels: destination.buffer().to_vec(),
        width: target_width,
        height: target_height,
    }
}

/// Rotates a packed RGB image clockwise by `0`, `90`, `180`, or `270` degrees.
///
/// # Performance
///
/// The outer row loop is parallelized with Rayon for all non-trivial rotations.
#[inline]
#[must_use]
pub fn rotate_rgb(image: DecodedImage, degrees: u16) -> DecodedImage {
    match degrees % 360 {
        0 => image,
        180 => {
            let width = image.width;
            let height = image.height;
            let stride = width as usize * 3;
            let src_pixels = image.pixels;
            let src_height = height as usize;
            let src_width = width as usize;
            let mut pixels = vec![0u8; src_pixels.len()];
            pixels
                .par_chunks_mut(stride)
                .enumerate()
                .for_each(|(row, dst)| {
                    let src = &src_pixels[(src_height - 1 - row) * stride..][..stride];
                    for col in 0..src_width {
                        dst[col * 3..col * 3 + 3]
                            .copy_from_slice(&src[(src_width - 1 - col) * 3..][..3]);
                    }
                });
            DecodedImage {
                pixels,
                width,
                height,
            }
        }
        90 => {
            let width = image.width;
            let height = image.height;
            let (ow, oh) = (width as usize, height as usize);
            let src_pixels = image.pixels;
            let mut pixels = vec![0u8; src_pixels.len()];
            pixels
                .par_chunks_mut(oh * 3)
                .enumerate()
                .for_each(|(dst_row, dst_chunk)| {
                    for dst_col in 0..oh {
                        let src_row = oh - 1 - dst_col;
                        let src_col = dst_row;
                        dst_chunk[dst_col * 3..dst_col * 3 + 3]
                            .copy_from_slice(&src_pixels[(src_row * ow + src_col) * 3..][..3]);
                    }
                });
            DecodedImage {
                pixels,
                width: height,
                height: width,
            }
        }
        270 => {
            let width = image.width;
            let height = image.height;
            let (ow, oh) = (width as usize, height as usize);
            let src_pixels = image.pixels;
            let mut pixels = vec![0u8; src_pixels.len()];
            pixels
                .par_chunks_mut(oh * 3)
                .enumerate()
                .for_each(|(dst_row, dst_chunk)| {
                    for dst_col in 0..oh {
                        let src_row = dst_col;
                        let src_col = ow - 1 - dst_row;
                        dst_chunk[dst_col * 3..dst_col * 3 + 3]
                            .copy_from_slice(&src_pixels[(src_row * ow + src_col) * 3..][..3]);
                    }
                });
            DecodedImage {
                pixels,
                width: height,
                height: width,
            }
        }
        _ => image,
    }
}

// ── Encode entry points ──────────────────────────────────────────────────────

/// Encodes a packed RGB image into one of CrabMagick's output formats.
///
/// # Performance
///
/// Reuses format-specific fast paths and avoids constructing intermediary image
/// wrappers except where the downstream encoder requires them.
#[inline]
#[must_use]
pub fn encode(
    image: DecodedImage,
    format: OutputFormat,
    quality: u8,
) -> Result<Vec<u8>, CrabMagickError> {
    let DecodedImage {
        pixels,
        width,
        height,
    } = image;

    match format {
        OutputFormat::Jpeg => {
            // TODO: plumb jxl-oxide's planar render buffers directly into zenjpeg once the
            // encoder exposes a non-packed RGB/YCbCr entry point. Today the hot path still
            // expects packed RGB8 slices, so JXL->JPEG keeps this intermediate representation.
            let config = EncoderConfig::ycbcr(quality.min(100), ChromaSubsampling::Quarter);
            let mut enc = config
                .encode_from_bytes(width, height, ZenLayout::Rgb8Srgb)
                .map_err(|error| encode_error(format!("JPEG encoder initialization failed: {error}")))?;
            enc.push_packed(&pixels, Unstoppable)
                .map_err(|error| encode_error(format!("JPEG pixel upload failed: {error}")))?;
            enc.finish()
                .map_err(|error| encode_error(format!("JPEG finalization failed: {error}")))
        }
        OutputFormat::Webp => {
            let rgb = RgbImage::from_raw(width, height, pixels)
                .ok_or_else(|| encode_error("WebP encoding received an invalid RGB buffer shape"))?;
            crate::fast_webp::encode_lossy_webp(&rgb, quality.min(100))
                .map_err(|error| encode_error(format!("WebP encoding failed: {error}")))
        }
        OutputFormat::Png => {
            let mut out = Vec::new();
            PngEncoder::new(&mut out)
                .write_image(
                    &pixels,
                    width,
                    height,
                    ColorType::Rgb8.into(),
                )
                .map_err(|error| encode_error(format!("PNG encoding failed: {error}")))?;
            Ok(out)
        }
        OutputFormat::Jxl => encode_jxl_rgb(
            &pixels,
            width,
            height,
            &JxlEncodeOptions {
                distance: Some(distance_from_quality(quality)),
                ..JxlEncodeOptions::default()
            },
        ),
        OutputFormat::Avif => encode_avif_rgb(&pixels, width, height, quality),
    }
}

/// Encodes packed RGB pixels into JPEG XL using bundled encoder modules.
///
/// # Performance
///
/// Keeps the call entirely within the bundled same-crate encoder stack so the
/// optimizer can inline configuration and dispatch code across former crate
/// boundaries.
#[inline]
#[must_use]
pub fn encode_jxl_rgb(
    pixels: &[u8],
    width: u32,
    height: u32,
    options: &JxlEncodeOptions,
) -> Result<Vec<u8>, CrabMagickError> {
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
            .map_err(|error| encode_error(format!("lossless JPEG XL encoding failed: {error}")))
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
            .map_err(|error| encode_error(format!("lossy JPEG XL encoding failed: {error}")))
    }
}

fn decode_via_image(bytes: &[u8]) -> Result<DecodedImage, CrabMagickError> {
    let image = image::load_from_memory(bytes)
        .map_err(|error| decode_malformed(format!("generic image decode failed: {error}")))?;
    let rgb = image.to_rgb8();
    Ok(DecodedImage {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    })
}

#[inline]
fn decode_jpeg_fast(bytes: &[u8]) -> Result<DecodedImage, CrabMagickError> {
    use crate::zune_core::bytestream::ZCursor;
    use crate::zune_core::colorspace::ColorSpace;
    use crate::zune_core::options::DecoderOptions;
    use crate::zune_jpeg::JpegDecoder;

    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), opts);
    decoder
        .decode_headers()
        .map_err(|e| decode_malformed(format!("JPEG header: {e}")))?;
    let pixels = decoder
        .decode()
        .map_err(|e| decode_malformed(format!("JPEG decode: {e}")))?;
    let (width, height) = decoder
        .dimensions()
        .ok_or_else(|| decode_malformed("JPEG: no dimensions after decode"))?;
    Ok(DecodedImage {
        pixels,
        width: width as u32,
        height: height as u32,
    })
}

#[inline]
fn decode_webp_fast(bytes: &[u8]) -> Result<DecodedImage, CrabMagickError> {
    let reader = BufReader::new(Cursor::new(bytes));
    let mut decoder = crate::fast_webp::WebPDecoder::new(reader)
        .map_err(|error| decode_malformed(format!("WebP decoder initialization failed: {error}")))?;
    let (width, height) = decoder.dimensions();
    let has_alpha = decoder.has_alpha();
    let mut decoded = vec![
        0u8;
        decoder
            .output_buffer_size()
            .ok_or_else(|| decode_malformed("WebP output buffer would exceed addressable memory"))?
    ];
    decoder
        .read_image(&mut decoded)
        .map_err(|error| decode_malformed(format!("WebP pixel decode failed: {error}")))?;

    let pixels = if has_alpha {
        let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
        for rgba in decoded.chunks_exact(4) {
            rgb.extend_from_slice(&rgba[..3]);
        }
        rgb
    } else {
        decoded
    };

    Ok(DecodedImage {
        pixels,
        width,
        height,
    })
}

fn decode_svg(bytes: &[u8], out_w: u32, out_h: u32) -> Result<DecodedImage, CrabMagickError> {
    let options = usvg::Options::default();
    let tree = usvg::Tree::from_data(bytes, &options)
        .map_err(|error| decode_malformed(format!("SVG parsing failed: {error}")))?;

    let natural = tree.size().to_int_size();
    let (width, height) = resolve_output_size(natural.width(), natural.height(), out_w, out_h);
    let sx = width as f32 / natural.width() as f32;
    let sy = height as f32 / natural.height() as f32;
    let transform = tiny_skia::Transform::from_scale(sx, sy);

    let mut pixmap = tiny_skia::Pixmap::new(width, height)
        .ok_or_else(|| decode_malformed("failed to allocate the SVG raster surface"))?;
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

fn decode_svg_info(bytes: &[u8]) -> Result<ImageInfo, CrabMagickError> {
    let options = usvg::Options::default();
    let tree = usvg::Tree::from_data(bytes, &options)
        .map_err(|error| decode_malformed(format!("SVG parsing failed: {error}")))?;
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
) -> Result<DecodedImage, CrabMagickError> {
    let pdfium = Pdfium::default();
    let document = pdfium
        .load_pdf_from_byte_vec(bytes.to_vec(), None)
        .map_err(|error| decode_malformed(format!("PDF loading failed: {error}")))?;
    let page = document
        .pages()
        .get(page as u16)
        .map_err(|error| decode_malformed(format!("PDF page lookup failed: {error}")))?;

    let mut config = PdfRenderConfig::new();
    if out_w > 0 {
        config = config.set_target_width(out_w as i32);
    }
    if out_h > 0 {
        config = config.set_target_height(out_h as i32);
    }

    let image = page
        .render_with_config(&config)
        .map_err(|error| decode_malformed(format!("PDF rasterization failed: {error}")))?
        .as_image();
    let rgb = image.to_rgb8();
    Ok(DecodedImage {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    })
}

fn decode_tiff_page(bytes: &[u8], page: u32) -> Result<DecodedImage, CrabMagickError> {
    let cursor = Cursor::new(bytes);
    let reader = BufReader::new(cursor);
    let mut decoder = TiffDecoder::new(reader)
        .map_err(|error| decode_malformed(format!("TIFF decoder initialization failed: {error}")))?;

    for _ in 0..page {
        if decoder.more_images() {
            decoder
                .next_image()
                .map_err(|error| decode_malformed(format!("TIFF page advance failed: {error}")))?;
        } else {
            return Err(decode_malformed(format!(
                "TIFF page index {page} is out of range for this file",
            )));
        }
    }

    let (width, height) = decoder
        .dimensions()
        .map_err(|error| decode_malformed(format!("TIFF dimension read failed: {error}")))?;
    let color = decoder
        .colortype()
        .map_err(|error| decode_malformed(format!("TIFF color-type read failed: {error}")))?;
    let image = decoder
        .read_image()
        .map_err(|error| decode_malformed(format!("TIFF pixel decode failed: {error}")))?;

    tiff_to_rgb(image, color, width, height)
}

fn tiff_to_rgb(
    image: DecodingResult,
    color: tiff::ColorType,
    width: u32,
    height: u32,
) -> Result<DecodedImage, CrabMagickError> {
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
        _ => Err(decode_unsupported_format(
            "TIFF pixel format is unsupported; only Gray/RGB/RGBA 8-bit and 16-bit images are supported",
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
    image: DecodedImage,
    region: Option<(u32, u32, u32, u32)>,
    square: bool,
) -> Result<DecodedImage, CrabMagickError> {
    if let Some((x, y, w, h)) = region {
        let requested_region = RequestedRegion::new(x, y, w, h);
        let image_info = ImageInfo {
            width: image.width,
            height: image.height,
        };
        validate_requested_region(requested_region, image_info)?;
        return Ok(apply_region(image, requested_region));
    }
    if square {
        return Ok(square_crop(image));
    }
    Ok(image)
}

fn validate_requested_region(
    region: RequestedRegion,
    image: ImageInfo,
) -> Result<(), CrabMagickError> {
    if region.width == 0 || region.height == 0 {
        return Err(decode_malformed(
            "requested region width and height must both be greater than zero",
        ));
    }

    if image.contains_region(region) {
        Ok(())
    } else {
        Err(region_out_of_bounds(region, image))
    }
}

fn apply_region(image: DecodedImage, region: RequestedRegion) -> DecodedImage {
    if region.left == 0
        && region.top == 0
        && region.width == image.width
        && region.height == image.height
    {
        return image;
    }

    let source_stride = image.width as usize * 3;
    let destination_stride = region.width as usize * 3;
    let mut pixels = vec![0u8; (region.width * region.height * 3) as usize];

    for row in 0..region.height as usize {
        let src_offset = (region.top as usize + row) * source_stride + region.left as usize * 3;
        let dst_offset = row * destination_stride;
        pixels[dst_offset..dst_offset + destination_stride]
            .copy_from_slice(&image.pixels[src_offset..src_offset + destination_stride]);
    }

    DecodedImage {
        pixels,
        width: region.width,
        height: region.height,
    }
}

fn square_crop(image: DecodedImage) -> DecodedImage {
    let side = image.width.min(image.height);
    let region =
        RequestedRegion::new((image.width - side) / 2, (image.height - side) / 2, side, side);
    apply_region(image, region)
}

fn encode_avif_rgb(
    pixels: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, CrabMagickError> {
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
            .map_err(|error| encode_error(format!("AVIF encoding failed: {error}")))?;
        Ok(encoded.avif_file)
    }
    #[cfg(not(feature = "avif"))]
    {
        let _ = (pixels, width, height, quality);
        Err(encode_error(
            "AVIF output is unavailable in this build; enable the `avif` feature",
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

