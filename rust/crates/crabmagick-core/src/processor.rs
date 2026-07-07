//! High-level request orchestration and public error types for CrabMagick.

use std::fmt;

use crate::pipeline::{
    SourceFormat, decode_any_info, decode_any_with_options, detect_format, encode,
    init_decoded_image_cache, read_file_bytes, resize_rgb, rotate_rgb,
};

// ── Public request and response types ────────────────────────────────────────

/// Output codecs supported by [`process_image`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Baseline JPEG output encoded through bundled `zenjpeg`.
    Jpeg,
    /// Lossy WebP output encoded through bundled `fast-webp`.
    Webp,
    /// PNG output encoded through the `image` crate.
    Png,
    /// JPEG XL output encoded through bundled `jxl-encoder`.
    Jxl,
    /// AVIF output encoded through `ravif` when the `avif` feature is enabled.
    Avif,
}

/// A rectangular crop request in source-image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestedRegion {
    /// Left edge of the crop rectangle in pixels.
    pub left: u32,
    /// Top edge of the crop rectangle in pixels.
    pub top: u32,
    /// Requested crop width in pixels.
    pub width: u32,
    /// Requested crop height in pixels.
    pub height: u32,
}

impl RequestedRegion {
    /// Creates a region description from raw coordinates.
    #[must_use]
    pub const fn new(left: u32, top: u32, width: u32, height: u32) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }

    /// Returns the region as a tuple that matches the lower-level pipeline API.
    #[must_use]
    pub const fn as_tuple(self) -> (u32, u32, u32, u32) {
        (self.left, self.top, self.width, self.height)
    }
}

impl fmt::Display for RequestedRegion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "left={}, top={}, width={}, height={}",
            self.left, self.top, self.width, self.height
        )
    }
}

/// High-level processing instructions for [`process_image`].
#[derive(Debug, Clone)]
pub struct ProcessRequest {
    /// Left edge of the requested crop region in pixels.
    pub region_left: u32,
    /// Top edge of the requested crop region in pixels.
    pub region_top: u32,
    /// Width of the requested crop region in pixels.
    pub region_width: u32,
    /// Height of the requested crop region in pixels.
    pub region_height: u32,
    /// Desired output width in pixels. A value of `0` preserves aspect ratio.
    pub output_width: u32,
    /// Desired output height in pixels. A value of `0` preserves aspect ratio.
    pub output_height: u32,
    /// Output codec to encode.
    pub output_format: OutputFormat,
    /// Codec quality hint in the `1..=100` range.
    pub quality: u8,
    /// Multi-page image index for TIFF and PDF sources.
    pub page: u32,
    /// Clockwise rotation in degrees: `0`, `90`, `180`, or `270`.
    pub rotation: u16,
    /// If `true`, crop the largest centered square before resizing.
    pub square_region: bool,
}

impl ProcessRequest {
    /// Returns the explicit crop request, if one was provided.
    #[must_use]
    pub const fn requested_region(&self) -> Option<RequestedRegion> {
        if self.region_width > 0 && self.region_height > 0 {
            Some(RequestedRegion::new(
                self.region_left,
                self.region_top,
                self.region_width,
                self.region_height,
            ))
        } else {
            None
        }
    }

    /// Returns the requested render size, if resizing was requested.
    #[must_use]
    pub const fn requested_render_size(&self) -> Option<(u32, u32)> {
        if self.output_width == 0 && self.output_height == 0 {
            None
        } else {
            Some((self.output_width, self.output_height))
        }
    }
}

/// Lightweight source-image metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageInfo {
    /// Source-image width in pixels.
    pub width: u32,
    /// Source-image height in pixels.
    pub height: u32,
}

impl ImageInfo {
    /// Returns `true` when the region fits entirely inside the image bounds.
    #[must_use]
    pub const fn contains_region(self, region: RequestedRegion) -> bool {
        region.width > 0
            && region.height > 0
            && region.left <= self.width
            && region.top <= self.height
            && region.left.saturating_add(region.width) <= self.width
            && region.top.saturating_add(region.height) <= self.height
    }
}

impl fmt::Display for ImageInfo {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}x{}", self.width, self.height)
    }
}

// ── Error model ──────────────────────────────────────────────────────────────

/// Errors produced by CrabMagick decoding, transform, and encoding operations.
#[derive(Debug, thiserror::Error)]
pub enum CrabMagickError {
    /// The source format is not supported by the current build.
    #[error("unsupported source format: {0}")]
    DecodeUnsupportedFormat(String),
    /// Reading the source image from disk failed.
    #[error("failed to read source image: {0}")]
    DecodeIo(#[from] std::io::Error),
    /// The source bytes were malformed or internally inconsistent.
    #[error("malformed source image: {0}")]
    DecodeMalformed(String),
    /// Output encoding failed.
    #[error("failed to encode output image: {0}")]
    EncodeError(String),
    /// The requested crop falls outside the decoded image bounds.
    #[error("requested region {region} lies outside image bounds {image}")]
    RegionOutOfBounds {
        /// Requested crop rectangle.
        region: RequestedRegion,
        /// Decoded image bounds.
        image: ImageInfo,
    },
}

impl CrabMagickError {
    /// Builds a malformed-decode error with a descriptive message.
    pub(crate) fn decode_malformed(message: impl Into<String>) -> Self {
        Self::DecodeMalformed(message.into())
    }

    /// Builds an unsupported-format error with a descriptive message.
    pub(crate) fn decode_unsupported_format(message: impl Into<String>) -> Self {
        Self::DecodeUnsupportedFormat(message.into())
    }

    /// Builds an encoding error with a descriptive message.
    pub(crate) fn encode_error(message: impl Into<String>) -> Self {
        Self::EncodeError(message.into())
    }

    /// Builds an out-of-bounds region error from the requested crop and image size.
    pub(crate) const fn region_out_of_bounds(
        region: RequestedRegion,
        image: ImageInfo,
    ) -> Self {
        Self::RegionOutOfBounds { region, image }
    }
}

// ── High-level processor facade ──────────────────────────────────────────────

/// Thin convenience wrapper around the free-function API.
#[derive(Debug, Default)]
pub struct CrabMagickProcessor;

impl CrabMagickProcessor {
    /// Initializes shared caches used by the decoding pipeline.
    #[inline]
    pub fn init(tile_cache_mb: u64, output_cache_mb: u64) {
        init(tile_cache_mb, output_cache_mb);
    }

    /// Processes one image request from source bytes on disk to encoded output bytes.
    #[inline]
    pub fn process_image(
        source_path: &str,
        request: ProcessRequest,
    ) -> Result<Vec<u8>, CrabMagickError> {
        process_image(source_path, request)
    }

    /// Returns source-image metadata without fully decoding the image when possible.
    #[inline]
    pub fn get_info(source_path: &str) -> Result<ImageInfo, CrabMagickError> {
        get_info(source_path)
    }
}

// ── Top-level public API ─────────────────────────────────────────────────────

/// Initializes shared caches used by the image pipeline.
pub fn init(tile_cache_mb: u64, _output_cache_mb: u64) {
    init_decoded_image_cache(tile_cache_mb);
}

/// Decodes, transforms, and encodes one image request.
///
/// # Performance
///
/// - Short-circuits JXL passthrough requests without a decode/encode round-trip.
/// - Reuses the shared decoded-image LRU for full-image requests.
/// - Keeps region decode, resize, and rotation in the same optimized pipeline.
pub fn process_image(
    source_path: &str,
    request: ProcessRequest,
) -> Result<Vec<u8>, CrabMagickError> {
    let requested_region = request.requested_region();
    let is_full_image = requested_region.is_none() && !request.square_region;
    let has_resize = request.output_width > 0 || request.output_height > 0;
    let has_rotation = request.rotation != 0;

    if is_full_image
        && !has_resize
        && !has_rotation
        && request.page == 0
        && request.output_format == OutputFormat::Jxl
    {
        let bytes = read_file_bytes(source_path)?;
        if detect_format(&bytes) == SourceFormat::Jxl {
            return Ok(bytes.to_vec());
        }
    }

    let decoded_image = decode_any_with_options(
        source_path,
        requested_region.map(RequestedRegion::as_tuple),
        request.square_region,
        request.page,
        request.requested_render_size(),
    )?;
    let resized_image = match request.requested_render_size() {
        Some((output_width, output_height)) => {
            resize_rgb(decoded_image, output_width, output_height)
        }
        None => decoded_image,
    };
    let rotated_image = if has_rotation {
        rotate_rgb(resized_image, request.rotation)
    } else {
        resized_image
    };

    encode(rotated_image, request.output_format, request.quality)
}

/// Returns source-image dimensions without performing a full transform pipeline.
pub fn get_info(source_path: &str) -> Result<ImageInfo, CrabMagickError> {
    decode_any_info(source_path, 0)
}
