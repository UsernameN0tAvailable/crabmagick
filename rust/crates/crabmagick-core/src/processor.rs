//! High-level request orchestration and public error types for CrabMagick.

use std::fmt;

use crate::pipeline::{
    decode_any_info, decode_any_with_options, detect_format, encode, init_decoded_image_cache,
    read_file_bytes, resize_rgb, rotate_rgb, SourceFormat,
};

// ── Public request and response types ────────────────────────────────────────

/// Output codecs supported by [`process_image`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Baseline JPEG output encoded through bundled JPEG encoder.
    Jpeg,
    /// Lossy WebP output encoded through bundled WebP encoder.
    Webp,
    /// Lossless WebP output via VP8L (compatibility alias for WebP with `lossless=true`).
    WebpLossless,
    /// PNG output encoded through the `image` crate.
    Png,
    /// JPEG XL output encoded through bundled JXL encoder.
    Jxl,
    /// AVIF output encoded through `ravif` when the `avif` feature is enabled.
    Avif,
    /// TIFF output encoded through the bundled `tiff` crate.
    Tiff,
    /// GIF output encoded through the `image` crate (256-color quantized).
    Gif,
    /// BMP output encoded through the `image` crate (uncompressed BGR).
    Bmp,
}

/// JPEG chroma subsampling exposed by the high-level processor API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// Match libvips default behavior: 4:2:0 below Q90, 4:4:4 at Q90+.
    Auto,
    /// 4:2:0 chroma subsampling.
    Cs420,
    /// 4:2:2 chroma subsampling.
    Cs422,
    /// 4:4:4 chroma subsampling.
    Cs444,
}

/// JPEG encoder configuration.
#[derive(Debug, Clone, Copy)]
pub struct JpegEncodeOptions {
    /// Quality in the `1..=100` range.
    pub quality: u8,
    /// Emit progressive JPEG scans.
    pub progressive: bool,
    /// Chroma subsampling mode.
    pub chroma_subsampling: ChromaSubsampling,
    /// Restart interval in MCUs/rows, depending on the lower-level encoder mapping.
    pub restart_interval: u16,
    /// Build optimal Huffman tables (two-pass, ~10% smaller files, ~1.5× slower encode).
    /// When false, uses pre-built standard tables (single-pass, matches libjpeg-turbo speed).
    /// Progressive mode always forces this to true.
    pub optimize_huffman: bool,
}

impl Default for JpegEncodeOptions {
    fn default() -> Self {
        Self {
            quality: 85,
            progressive: false,
            chroma_subsampling: ChromaSubsampling::Auto,
            restart_interval: 0,
            optimize_huffman: false,
        }
    }
}

/// WebP encoder configuration.
#[derive(Debug, Clone, Copy)]
pub struct WebpEncodeOptions {
    /// Quality in the `0..=100` range.
    pub quality: u8,
    /// Use lossless VP8L encoding.
    pub lossless: bool,
    /// Prefer near-lossless preprocessing when available.
    pub near_lossless: bool,
    /// Encoding effort hint in the `0..=6` range.
    pub effort: u8,
    /// Alpha quality in the `0..=100` range.
    pub alpha_quality: u8,
}

impl Default for WebpEncodeOptions {
    fn default() -> Self {
        Self {
            quality: 80,
            lossless: false,
            near_lossless: false,
            effort: 4,
            alpha_quality: 100,
        }
    }
}

/// PNG row filter selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngFilter {
    /// No filtering.
    None,
    /// Sub filter.
    Sub,
    /// Up filter.
    Up,
    /// Average filter.
    Avg,
    /// Paeth filter.
    Paeth,
    /// Adaptive/all filters.
    All,
}

/// PNG encoder configuration.
#[derive(Debug, Clone, Copy)]
pub struct PngEncodeOptions {
    /// Compression level hint in the `0..=9` range.
    pub compression: u8,
    /// Adam7 interlacing.
    pub progressive: bool,
    /// PNG row filter strategy.
    pub filter: PngFilter,
    /// Output bit depth (`8` or `16`).
    pub bitdepth: u8,
}

impl Default for PngEncodeOptions {
    fn default() -> Self {
        Self {
            compression: 6,
            progressive: false,
            filter: PngFilter::All,
            bitdepth: 8,
        }
    }
}

/// JPEG XL encoder configuration.
#[derive(Debug, Clone, Copy)]
pub struct JxlEncodeOptions {
    /// Quality in the `0..=100` range, used when `distance` is `None`.
    pub quality: u8,
    /// Explicit Butteraugli distance override.
    pub distance: Option<f32>,
    /// Encode effort in the `1..=9` range.
    pub effort: u8,
    /// Enable lossless mode.
    pub lossless: bool,
    /// Decode-speed tier hint in the `0..=4` range.
    pub tier: u8,
}

impl Default for JxlEncodeOptions {
    fn default() -> Self {
        Self {
            quality: 75,
            distance: None,
            effort: 7,
            lossless: false,
            tier: 0,
        }
    }
}

/// TIFF compression selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffCompression {
    /// No compression.
    None,
    /// LZW compression.
    Lzw,
    /// Deflate compression.
    Deflate,
    /// JPEG compression.
    Jpeg,
    /// PackBits compression.
    Packbits,
}

/// TIFF encoder configuration.
#[derive(Debug, Clone, Copy)]
pub struct TiffEncodeOptions {
    /// Compression method.
    pub compression: TiffCompression,
    /// JPEG quality when JPEG compression is selected.
    pub quality: u8,
    /// Horizontal differencing predictor for LZW/Deflate.
    pub predictor: bool,
    /// Emit tiled TIFF output.
    pub tiled: bool,
    /// Tile width when `tiled=true`.
    pub tile_width: u16,
    /// Tile height when `tiled=true`.
    pub tile_height: u16,
}

impl Default for TiffEncodeOptions {
    fn default() -> Self {
        Self {
            compression: TiffCompression::Lzw,
            quality: 75,
            predictor: true,
            tiled: false,
            tile_width: 128,
            tile_height: 128,
        }
    }
}

/// GIF encoder configuration.
#[derive(Debug, Clone, Copy)]
pub struct GifEncodeOptions {
    /// Dither strength in the `0.0..=1.0` range.
    pub dither: f32,
    /// Encoding effort in the `1..=10` range.
    pub effort: u8,
    /// Output palette bit depth in the `1..=8` range.
    pub bitdepth: u8,
}

impl Default for GifEncodeOptions {
    fn default() -> Self {
        Self {
            dither: 1.0,
            effort: 7,
            bitdepth: 8,
        }
    }
}

/// AVIF encoder configuration.
#[derive(Debug, Clone, Copy)]
pub struct AvifEncodeOptions {
    /// Quality in the `1..=100` range.
    pub quality: u8,
    /// Prefer lossless output when supported by the backend.
    pub lossless: bool,
    /// Encoding effort / speed hint in the `1..=10` range.
    pub effort: u8,
}

impl Default for AvifEncodeOptions {
    fn default() -> Self {
        Self {
            quality: 80,
            lossless: false,
            effort: 4,
        }
    }
}

/// All supported encoder option families.
#[derive(Debug, Clone)]
pub enum EncodeOptions {
    /// JPEG options.
    Jpeg(JpegEncodeOptions),
    /// WebP options.
    Webp(WebpEncodeOptions),
    /// PNG options.
    Png(PngEncodeOptions),
    /// JPEG XL options.
    Jxl(JxlEncodeOptions),
    /// AVIF options.
    Avif(AvifEncodeOptions),
    /// TIFF options.
    Tiff(TiffEncodeOptions),
    /// GIF options.
    Gif(GifEncodeOptions),
    /// BMP has no configurable encoder options today.
    Bmp,
}

impl EncodeOptions {
    /// Builds encoder options from a legacy `(format, quality)` pair.
    #[must_use]
    pub fn with_quality(format: OutputFormat, quality: u8) -> Self {
        match format {
            OutputFormat::Jpeg => Self::Jpeg(JpegEncodeOptions {
                quality,
                ..JpegEncodeOptions::default()
            }),
            OutputFormat::Webp | OutputFormat::WebpLossless => Self::Webp(WebpEncodeOptions {
                quality,
                lossless: format == OutputFormat::WebpLossless,
                ..WebpEncodeOptions::default()
            }),
            OutputFormat::Png => Self::Png(PngEncodeOptions::default()),
            OutputFormat::Jxl => Self::Jxl(JxlEncodeOptions {
                quality,
                ..JxlEncodeOptions::default()
            }),
            OutputFormat::Avif => Self::Avif(AvifEncodeOptions {
                quality,
                ..AvifEncodeOptions::default()
            }),
            OutputFormat::Tiff => Self::Tiff(TiffEncodeOptions::default()),
            OutputFormat::Gif => Self::Gif(GifEncodeOptions::default()),
            OutputFormat::Bmp => Self::Bmp,
        }
    }

    /// Returns the effective output format for this option family.
    #[must_use]
    pub fn output_format(&self) -> OutputFormat {
        match self {
            Self::Jpeg(_) => OutputFormat::Jpeg,
            Self::Webp(options) if options.lossless => OutputFormat::WebpLossless,
            Self::Webp(_) => OutputFormat::Webp,
            Self::Png(_) => OutputFormat::Png,
            Self::Jxl(_) => OutputFormat::Jxl,
            Self::Avif(_) => OutputFormat::Avif,
            Self::Tiff(_) => OutputFormat::Tiff,
            Self::Gif(_) => OutputFormat::Gif,
            Self::Bmp => OutputFormat::Bmp,
        }
    }

    /// Returns a best-effort quality hint for compatibility callers.
    #[must_use]
    pub fn quality(&self) -> u8 {
        match self {
            Self::Jpeg(options) => options.quality,
            Self::Webp(options) => options.quality,
            Self::Png(_) => 100,
            Self::Jxl(options) => options.quality,
            Self::Avif(options) => options.quality,
            Self::Tiff(options) => options.quality,
            Self::Gif(_) | Self::Bmp => 100,
        }
    }
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
    /// Per-format encoder options.
    pub encode: EncodeOptions,
    /// Multi-page image index for TIFF and PDF sources.
    pub page: u32,
    /// Clockwise rotation in degrees: `0`, `90`, `180`, or `270`.
    pub rotation: u16,
    /// If `true`, crop the largest centered square before resizing.
    pub square_region: bool,
}

impl ProcessRequest {
    /// Creates a request with default transform settings and legacy quality-based encoding.
    #[must_use]
    pub fn with_quality(output_format: OutputFormat, quality: u8) -> Self {
        Self {
            region_left: 0,
            region_top: 0,
            region_width: 0,
            region_height: 0,
            output_width: 0,
            output_height: 0,
            encode: EncodeOptions::with_quality(output_format, quality),
            page: 0,
            rotation: 0,
            square_region: false,
        }
    }

    /// Returns the selected output format.
    #[must_use]
    pub fn output_format(&self) -> OutputFormat {
        self.encode.output_format()
    }

    /// Returns a compatibility quality hint.
    #[must_use]
    pub fn quality(&self) -> u8 {
        self.encode.quality()
    }

    /// Returns the configured encoder options.
    #[must_use]
    pub const fn encode_options(&self) -> &EncodeOptions {
        &self.encode
    }

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
    pub(crate) const fn region_out_of_bounds(region: RequestedRegion, image: ImageInfo) -> Self {
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
        && request.output_format() == OutputFormat::Jxl
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

    encode(rotated_image, request.encode_options())
}

/// Returns source-image dimensions without performing a full transform pipeline.
pub fn get_info(source_path: &str) -> Result<ImageInfo, CrabMagickError> {
    decode_any_info(source_path, 0)
}

/// Losslessly repackages a JPEG file into a JXL container (equivalent to `cjxl --lossless_jpeg=1`).
///
/// Reads the JPEG at `path`, extracts its DCT coefficients, and wraps them in a JXL bitstream.
/// The original JPEG can be recovered exactly from the output. Typical savings: 15–30%.
#[cfg(feature = "jpeg-reencoding")]
pub fn transcode_jpeg_to_jxl(path: &str) -> Result<Vec<u8>, CrabMagickError> {
    crate::pipeline::transcode_jpeg_file_to_jxl(path)
}
