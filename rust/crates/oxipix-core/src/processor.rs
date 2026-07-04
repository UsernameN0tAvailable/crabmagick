use std::sync::Arc;

use once_cell::sync::OnceCell;
use tokio::runtime::Runtime;

use crate::cache::{DecodedImage, OutputCache, TileCache};
use crate::pipeline::{apply_region, decode_jxl, decode_jxl_info, encode, resize_rgb, rotate_rgb, square_crop};

static RT: OnceCell<Runtime> = OnceCell::new();
static TILE_CACHE: OnceCell<TileCache> = OnceCell::new();
static OUTPUT_CACHE: OnceCell<OutputCache> = OnceCell::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Jpeg,
    Webp,
    Png,
    Jxl,
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
    pub fn init(tile_cache_mb: u64, output_cache_mb: u64) {
        init(tile_cache_mb, output_cache_mb);
    }

    pub fn process_image(jxl_path: &str, req: ProcessRequest) -> Result<Vec<u8>, OxipixError> {
        process_image(jxl_path, req)
    }

    pub fn get_info(jxl_path: &str) -> Result<ImageInfo, OxipixError> {
        get_info(jxl_path)
    }
}

/// Initialize the global caches and background runtime.
pub fn init(tile_cache_mb: u64, output_cache_mb: u64) {
    TILE_CACHE.get_or_init(|| TileCache::new(tile_cache_mb));
    OUTPUT_CACHE.get_or_init(|| OutputCache::new(output_cache_mb));
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4),
            )
            .enable_all()
            .build()
            .expect("tokio runtime")
    });
}

/// Process an image file: output cache -> decoded cache -> crop -> resize -> encode.
pub fn process_image(jxl_path: &str, req: ProcessRequest) -> Result<Vec<u8>, OxipixError> {
    let tile_cache = TILE_CACHE.get_or_init(|| TileCache::new(512));
    let output_cache = OUTPUT_CACHE.get_or_init(|| OutputCache::new(512));

    let cache_key = OutputCache::make_key(jxl_path, &req);
    if let Some(cached) = output_cache.get(&cache_key) {
        return Ok((*cached).clone());
    }

    let decoded = if let Some(cached) = tile_cache.get(jxl_path) {
        cached
    } else {
        let img = decode_jxl(jxl_path)?;
        let arc = Arc::new(img);
        tile_cache.insert(jxl_path.to_string(), Arc::clone(&arc));
        arc
    };

    let cropped = if req.square_region {
        square_crop(&decoded)
    } else if req.region_w > 0 && req.region_h > 0 {
        apply_region(&decoded, req.region_x, req.region_y, req.region_w, req.region_h)
    } else {
        DecodedImage {
            pixels: decoded.pixels.clone(),
            width: decoded.width,
            height: decoded.height,
        }
    };

    let resized = if req.out_w > 0 || req.out_h > 0 {
        resize_rgb(&cropped, req.out_w, req.out_h)
    } else {
        cropped
    };

    let rotated = if req.rotation != 0 {
        rotate_rgb(&resized, req.rotation)
    } else {
        resized
    };

    let encoded = encode(&rotated, req.format, req.quality)?;
    output_cache.insert(cache_key, Arc::new(encoded.clone()));
    Ok(encoded)
}

pub fn get_info(jxl_path: &str) -> Result<ImageInfo, OxipixError> {
    let tile_cache = TILE_CACHE.get_or_init(|| TileCache::new(512));
    if let Some(cached) = tile_cache.get(jxl_path) {
        return Ok(ImageInfo {
            width: cached.width,
            height: cached.height,
        });
    }

    decode_jxl_info(jxl_path)
}
