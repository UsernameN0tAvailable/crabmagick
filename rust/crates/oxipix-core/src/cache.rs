use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;

use crate::processor::{OutputFormat, ProcessRequest};

/// Decoded pixel data for a full image (RGB, width, height).
#[derive(Debug)]
pub struct DecodedImage {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl DecodedImage {
    pub fn byte_len(&self) -> u64 {
        self.pixels.len() as u64
    }
}

/// Caches decoded pixel data by input path.
pub struct TileCache {
    map: DashMap<String, Arc<DecodedImage>>,
    used_bytes: AtomicU64,
    max_bytes: u64,
}

impl TileCache {
    pub fn new(max_mb: u64) -> Self {
        Self {
            map: DashMap::new(),
            used_bytes: AtomicU64::new(0),
            max_bytes: max_mb.saturating_mul(1024 * 1024),
        }
    }

    pub fn get(&self, path: &str) -> Option<Arc<DecodedImage>> {
        self.map.get(path).map(|entry| Arc::clone(&*entry))
    }

    pub fn insert(&self, path: String, img: Arc<DecodedImage>) {
        let img_bytes = img.byte_len();
        if img_bytes > self.max_bytes {
            return;
        }

        if let Some(previous) = self.map.insert(path, img) {
            self.used_bytes
                .fetch_sub(previous.byte_len(), Ordering::Relaxed);
        }
        self.used_bytes.fetch_add(img_bytes, Ordering::Relaxed);
        self.maybe_evict();
    }

    fn maybe_evict(&self) {
        if self.used_bytes.load(Ordering::Relaxed) <= self.max_bytes {
            return;
        }

        self.map.clear();
        self.used_bytes.store(0, Ordering::Relaxed);
    }
}

/// Caches fully encoded output bytes by a request key.
pub struct OutputCache {
    map: DashMap<String, Arc<Vec<u8>>>,
    used_bytes: AtomicU64,
    max_bytes: u64,
}

impl OutputCache {
    pub fn new(max_mb: u64) -> Self {
        Self {
            map: DashMap::new(),
            used_bytes: AtomicU64::new(0),
            max_bytes: max_mb.saturating_mul(1024 * 1024),
        }
    }

    pub fn get(&self, key: &str) -> Option<Arc<Vec<u8>>> {
        self.map.get(key).map(|entry| Arc::clone(&*entry))
    }

    pub fn insert(&self, key: String, data: Arc<Vec<u8>>) {
        let data_bytes = data.len() as u64;
        if data_bytes > self.max_bytes {
            return;
        }

        if let Some(previous) = self.map.insert(key, data) {
            self.used_bytes
                .fetch_sub(previous.len() as u64, Ordering::Relaxed);
        }
        self.used_bytes.fetch_add(data_bytes, Ordering::Relaxed);
        self.maybe_evict();
    }

    pub fn make_key(path: &str, req: &ProcessRequest) -> String {
        format!(
            "{path}:{}:{}:{}:{}:{}:{}:{}:{}",
            req.region_x,
            req.region_y,
            req.region_w,
            req.region_h,
            req.out_w,
            req.out_h,
            req.format.as_str(),
            req.quality,
        )
    }

    fn maybe_evict(&self) {
        if self.used_bytes.load(Ordering::Relaxed) <= self.max_bytes {
            return;
        }

        self.map.clear();
        self.used_bytes.store(0, Ordering::Relaxed);
    }
}

impl OutputFormat {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            OutputFormat::Jpeg => "jpeg",
            OutputFormat::Webp => "webp",
            OutputFormat::Png => "png",
            OutputFormat::Jxl => "jxl",
        }
    }
}
