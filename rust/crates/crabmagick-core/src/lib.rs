//! CrabMagick's bundled pure-Rust image processing core.

#![recursion_limit = "256"]

extern crate alloc;

whereat::define_at_crate_info!(path = "crabmagick-core/");

/// WebP decode implementation.
pub(crate) mod webp_decode;
/// JXL encode implementation.
pub mod jxl_encode;
/// JXL decode implementation (AVX2/AVX512 DCT).
pub(crate) mod jxl_decode;
/// JXL encoder SIMD kernels.
pub(crate) mod jxl_encode_simd;
/// Low-level decode, transform, and encode primitives.
pub mod pipeline;
/// High-level request types and orchestration helpers.
pub mod processor;
/// JPEG 2000 decode implementation.
pub(crate) mod jpeg2000_decode;
/// JPEG encode implementation.
pub(crate) mod jpeg_encode;
/// JPEG decoder core types.
pub(crate) mod jpeg_decode_core;
/// JPEG decode implementation (SIMD AVX-512/AVX2/NEON).
pub(crate) mod jpeg_decode;

pub use pipeline::{JxlEncodeOptions, decode_jxl_info_from_bytes, encode_jxl_rgb};
pub use jxl_encode::{EncoderMode, PixelLayout as JxlPixelLayout};
pub use processor::{
    get_info, init, process_image, CrabMagickError, CrabMagickProcessor, ImageInfo, OutputFormat,
    ProcessRequest, RequestedRegion,
};
