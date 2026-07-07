//! CrabMagick's bundled pure-Rust image processing core.

#![recursion_limit = "256"]

extern crate alloc;

whereat::define_at_crate_info!(path = "crabmagick-core/");

/// Bundled fast WebP codec implementation used internally by the pipeline.
pub(crate) mod fast_webp;
/// Bundled JPEG XL encoder implementation used internally by the pipeline.
pub mod jxl_encoder;
/// Vendored `jxl-oxide` decoder ecosystem with upgraded AVX2/AVX512 DCT.
pub(crate) mod jxl_oxide_vendored;
/// Bundled SIMD helpers shared by the JPEG XL encoder.
pub(crate) mod jxl_encoder_simd;
/// Low-level decode, transform, and encode primitives.
pub mod pipeline;
/// High-level request types and orchestration helpers.
pub mod processor;
/// Bundled JPEG 2000 decoder implementation used internally by the pipeline.
pub(crate) mod zen_jp2;
/// Bundled JPEG codec implementation used internally by the pipeline.
pub(crate) mod zenjpeg;
/// Vendored `zune-core` support crate (shared types for the JPEG decoder).
pub(crate) mod zune_core;
/// Vendored `zune-jpeg` decoder with added AVX-512 IDCT and color-convert paths.
pub(crate) mod zune_jpeg;

pub use pipeline::{JxlEncodeOptions, decode_jxl_info_from_bytes, encode_jxl_rgb};
pub use jxl_encoder::{EncoderMode, PixelLayout as JxlPixelLayout};
pub use processor::{
    get_info, init, process_image, CrabMagickError, CrabMagickProcessor, ImageInfo, OutputFormat,
    ProcessRequest, RequestedRegion,
};
