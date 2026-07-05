use std::sync::Arc;

use once_cell::sync::OnceCell;
use tokio::runtime::Runtime;

use crate::cache::{DecodedImage, OutputCache, TileCache};
use crate::pipeline::{apply_region, decode_jxl, decode_jxl_info, decode_jxl_region, encode, resize_rgb, rotate_rgb, square_crop};

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

/// Process an image file: output cache -> tile cache -> crop-during-decode -> resize -> encode.
///
/// For region requests, jxl-oxide is asked to decode only the requested crop window,
/// skipping groups outside it. This makes first-request tile latency proportional to
/// the tile area, not the full image area.
pub fn process_image(jxl_path: &str, req: ProcessRequest) -> Result<Vec<u8>, OxipixError> {
    let tile_cache = TILE_CACHE.get_or_init(|| TileCache::new(512));
    let output_cache = OUTPUT_CACHE.get_or_init(|| OutputCache::new(512));

    let cache_key = OutputCache::make_key(jxl_path, &req);
    if let Some(cached) = output_cache.get(&cache_key) {
        return Ok((*cached).clone());
    }

    // Build a tile-cache key that includes the crop region so partial decodes are
    // cached separately from the full image. Full-image key = jxl_path.
    let tile_key: String = if req.region_w > 0 && req.region_h > 0 {
        format!("{}#r{},{},{},{}", jxl_path, req.region_x, req.region_y, req.region_w, req.region_h)
    } else if req.square_region {
        format!("{}#sq", jxl_path)
    } else {
        jxl_path.to_string()
    };

    let decoded = if let Some(cached) = tile_cache.get(&tile_key) {
        // Cache hit: already have the (potentially pre-cropped) pixels.
        cached
    } else if req.region_w > 0 && req.region_h > 0 {
        // Region request + cache miss → decode only the crop window (fast path).
        let img = decode_jxl_region(
            jxl_path,
            req.region_x,
            req.region_y,
            req.region_w,
            req.region_h,
        )?;
        let arc = Arc::new(img);
        tile_cache.insert(tile_key, Arc::clone(&arc));
        arc
    } else if req.square_region {
        // Square region: compute the crop from image dimensions, then decode with crop.
        // First try to get dimensions cheaply from the full-image tile cache or the header.
        let (img_w, img_h) = if let Some(full) = tile_cache.get(jxl_path) {
            (full.width, full.height)
        } else {
            let info = decode_jxl_info(jxl_path)?;
            (info.width, info.height)
        };
        let side = img_w.min(img_h);
        let cx = (img_w - side) / 2;
        let cy = (img_h - side) / 2;
        let img = decode_jxl_region(jxl_path, cx, cy, side, side)?;
        let arc = Arc::new(img);
        tile_cache.insert(tile_key, Arc::clone(&arc));
        arc
    } else {
        // Full image request → decode everything and cache it.
        let img = decode_jxl(jxl_path)?;
        let arc = Arc::new(img);
        tile_cache.insert(tile_key, Arc::clone(&arc));
        arc
    };

    // If we decoded a pre-cropped tile, the region is already applied.
    // Only apply in-memory region crop when we have a full-image tile cache hit.
    let cropped = if req.region_w > 0 && req.region_h > 0 && decoded.width == req.region_w && decoded.height == req.region_h {
        // Already the exact crop — skip in-memory apply_region.
        DecodedImage {
            pixels: decoded.pixels.clone(),
            width: decoded.width,
            height: decoded.height,
        }
    } else if req.square_region && decoded.width == decoded.height {
        // Already square-cropped.
        DecodedImage {
            pixels: decoded.pixels.clone(),
            width: decoded.width,
            height: decoded.height,
        }
    } else if req.square_region {
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
