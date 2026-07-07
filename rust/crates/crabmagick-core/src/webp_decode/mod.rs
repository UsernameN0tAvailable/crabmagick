// Copyright (c) the image-rs contributors. Licensed MIT OR Apache-2.0.
#![allow(warnings, clippy::all, unexpected_cfgs)]
//! Decoding and Encoding of WebP Images

#![deny(missing_docs)]
// Increase recursion limit for the `quick_error!` macro.

pub use self::decoder::{
    DecodingError, LoopCount, UpsamplingMethod, WebPDecodeOptions, WebPDecoder,
};
pub use self::encoder::{ColorType, EncoderParams, EncodingError, WebPEncoder};
pub(crate) use self::vp8_enc::{encode_lossy_webp, WebPEncodeError};

mod alpha_blending;
mod decoder;
mod encoder;
mod extended;
mod huffman;
mod loop_filter;
mod lossless;
mod lossless_transform;
mod transform;
mod vp8_arithmetic_decoder;
mod vp8_enc;
mod yuv;

pub mod vp8;
