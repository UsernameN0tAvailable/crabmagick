//! Simple public decode API for zen-jp2.

use alloc::vec::Vec;

use crate::zen_jp2::error::{DecodeError, FormatError};
use crate::zen_jp2::{DecodeSettings, DecoderContext, Image};

/// A decoded JPEG 2000 image with packed 8-bit pixels.
#[derive(Debug)]
pub struct DecodedImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of color components (for example 1 = grayscale, 3 = RGB).
    pub components: u8,
    /// Packed u8 pixels: grayscale is `[Y, Y, ...]`, RGB is `[R, G, B, ...]`.
    pub pixels: Vec<u8>,
}

/// Decode a JPEG 2000 image from raw bytes.
///
/// Accepts JP2 container images and bare J2K codestreams.
#[inline]
pub(crate) fn decode(data: &[u8]) -> Result<DecodedImage, DecodeError> {
    let signature = data.get(..12).unwrap_or(data);
    if !signature.starts_with(crate::zen_jp2::JP2_MAGIC) && !signature.starts_with(crate::zen_jp2::CODESTREAM_MAGIC) {
        return Err(DecodeError::Format(FormatError::InvalidSignature));
    }

    let image = Image::new(data, &DecodeSettings::default())?;
    let width = image.width();
    let height = image.height();

    let mut ctx = DecoderContext::default();
    let decoded = image.decode(&mut ctx)?;

    Ok(DecodedImage {
        width,
        height,
        components: decoded.components().len() as u8,
        pixels: decoded.data_u8(),
    })
}
