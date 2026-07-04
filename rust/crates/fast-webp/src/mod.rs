//! Decoding and Encoding of WebP Images

#[cfg(feature = "webp-encoder")]
pub use self::encoder::{WebPEncoder, WebPQuality};

#[cfg(feature = "webp-encoder")]
mod encoder;

#[cfg(feature = "webp")]
pub use self::decoder::WebPDecoder;
pub use self::vp8_enc::{encode_lossy_webp, WebPEncodeError};

#[cfg(feature = "webp")]
mod decoder;
#[cfg(feature = "webp")]
mod extended;
#[cfg(feature = "webp")]
mod huffman;
#[cfg(feature = "webp")]
mod loop_filter;
#[cfg(feature = "webp")]
mod lossless;
#[cfg(feature = "webp")]
mod lossless_transform;
#[cfg(feature = "webp")]
mod transform;
mod vp8_enc;

#[cfg(feature = "webp")]
pub mod vp8;
