use crate::pipeline::{
    decode_any_info, decode_any_with_options, detect_format, encode, resize_rgb, rotate_rgb,
    SourceFormat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Jpeg,
    Webp,
    Png,
    Jxl,
    Avif,
}

#[derive(Debug, Clone)]
pub struct ProcessRequest {
    pub region_x: u32,
    pub region_y: u32,
    pub region_w: u32,
    pub region_h: u32,
    pub out_w: u32,
    pub out_h: u32,
    pub format: OutputFormat,
    pub quality: u8,
    pub page: u32,
    /// Clockwise rotation in degrees: 0, 90, 180, 270.
    pub rotation: u16,
    /// If true, crop the largest centred square before resizing.
    pub square_region: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum OxipixError {
    #[error("decode failed: {0}")]
    Decode(String),
    #[error("encode failed: {0}")]
    Encode(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Default)]
pub struct OxipixProcessor;

impl OxipixProcessor {
    pub fn process_image(source_path: &str, req: ProcessRequest) -> Result<Vec<u8>, OxipixError> {
        process_image(source_path, req)
    }

    pub fn get_info(source_path: &str) -> Result<ImageInfo, OxipixError> {
        get_info(source_path)
    }
}

pub fn process_image(source_path: &str, req: ProcessRequest) -> Result<Vec<u8>, OxipixError> {
    let is_full_image = req.region_w == 0 && req.region_h == 0 && !req.square_region;
    let no_resize = req.out_w == 0 && req.out_h == 0;
    let no_rotation = req.rotation == 0;
    if is_full_image && no_resize && no_rotation && req.page == 0 && req.format == OutputFormat::Jxl
    {
        let bytes = std::fs::read(source_path)?;
        if detect_format(&bytes) == SourceFormat::Jxl {
            return Ok(bytes);
        }
    }

    let region = (req.region_w > 0 && req.region_h > 0).then_some((
        req.region_x,
        req.region_y,
        req.region_w,
        req.region_h,
    ));
    let render_size = match (req.out_w, req.out_h) {
        (0, 0) => None,
        dims => Some(dims),
    };

    let decoded = decode_any_with_options(
        source_path,
        region,
        req.square_region,
        req.page,
        render_size,
    )?;
    let resized = if req.out_w > 0 || req.out_h > 0 {
        resize_rgb(decoded, req.out_w, req.out_h)
    } else {
        decoded
    };
    let rotated = if req.rotation != 0 {
        rotate_rgb(resized, req.rotation)
    } else {
        resized
    };

    encode(rotated, req.format, req.quality)
}

pub fn get_info(source_path: &str) -> Result<ImageInfo, OxipixError> {
    decode_any_info(source_path, 0)
}
