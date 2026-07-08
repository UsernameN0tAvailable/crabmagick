use std::collections::HashMap;

use crabmagick_core::processor::{get_info, process_image, OutputFormat, ProcessRequest};
use ext_php_rs::binary::Binary;
use ext_php_rs::exception::PhpException;
use ext_php_rs::prelude::*;
use ext_php_rs::types::ZendClassObject;

#[php_class]
#[php(name = "Crabmagick\\Image")]
pub struct CrabmagickImage {
    path: String,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
    out_w: u32,
    out_h: u32,
    page: u32,
    rotation: u16,
    square_region: bool,
}

#[php_impl]
#[php(change_method_case = "none")]
impl CrabmagickImage {
    pub fn __construct(path: String) -> PhpResult<Self> {
        Ok(Self {
            path,
            region_x: 0,
            region_y: 0,
            region_w: 0,
            region_h: 0,
            out_w: 0,
            out_h: 0,
            page: 0,
            rotation: 0,
            square_region: false,
        })
    }

    pub fn region<'a>(
        self_: &'a mut ZendClassObject<CrabmagickImage>,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> &'a mut ZendClassObject<CrabmagickImage> {
        self_.region_x = x;
        self_.region_y = y;
        self_.region_w = w;
        self_.region_h = h;
        self_
    }

    pub fn resize<'a>(
        self_: &'a mut ZendClassObject<CrabmagickImage>,
        w: u32,
        h: u32,
    ) -> &'a mut ZendClassObject<CrabmagickImage> {
        self_.out_w = w;
        self_.out_h = h;
        self_
    }

    pub fn page<'a>(
        self_: &'a mut ZendClassObject<CrabmagickImage>,
        page: u32,
    ) -> &'a mut ZendClassObject<CrabmagickImage> {
        self_.page = page;
        self_
    }

    pub fn rotate<'a>(
        self_: &'a mut ZendClassObject<CrabmagickImage>,
        degrees: u16,
    ) -> &'a mut ZendClassObject<CrabmagickImage> {
        self_.rotation = degrees;
        self_
    }

    pub fn square<'a>(
        self_: &'a mut ZendClassObject<CrabmagickImage>,
    ) -> &'a mut ZendClassObject<CrabmagickImage> {
        self_.square_region = true;
        self_
    }

    pub fn encode(&self, format: String, quality: u8) -> PhpResult<Binary<u8>> {
        let mut req = ProcessRequest::with_quality(parse_format(&format)?, quality);
        req.region_left = self.region_x;
        req.region_top = self.region_y;
        req.region_width = self.region_w;
        req.region_height = self.region_h;
        req.output_width = self.out_w;
        req.output_height = self.out_h;
        req.page = self.page;
        req.rotation = self.rotation;
        req.square_region = self.square_region;

        process_image(&self.path, req)
            .map(Binary::from)
            .map_err(|e| PhpException::default(e.to_string()))
    }

    #[php(name = "getInfo")]
    pub fn get_info_method(&self) -> PhpResult<HashMap<String, u32>> {
        get_info(&self.path)
            .map(|i| {
                HashMap::from([
                    ("width".to_string(), i.width),
                    ("height".to_string(), i.height),
                ])
            })
            .map_err(|e| PhpException::default(e.to_string()))
    }

    #[php(name = "isAvailable")]
    pub fn is_available() -> bool {
        true
    }
}

#[php_function]
#[php(name = "Crabmagick\\process")]
pub fn crabmagick_process(
    path: String,
    rx: u32,
    ry: u32,
    rw: u32,
    rh: u32,
    ow: u32,
    oh: u32,
    format: String,
    quality: u8,
) -> PhpResult<Binary<u8>> {
    let mut req = ProcessRequest::with_quality(parse_format(&format)?, quality);
    req.region_left = rx;
    req.region_top = ry;
    req.region_width = rw;
    req.region_height = rh;
    req.output_width = ow;
    req.output_height = oh;

    process_image(&path, req)
        .map(Binary::from)
        .map_err(|e| PhpException::default(e.to_string()))
}

#[php_function]
#[php(name = "Crabmagick\\info")]
pub fn crabmagick_info(path: String) -> PhpResult<HashMap<String, u32>> {
    get_info(&path)
        .map(|i| {
            HashMap::from([
                ("width".to_string(), i.width),
                ("height".to_string(), i.height),
            ])
        })
        .map_err(|e| PhpException::default(e.to_string()))
}

#[php_module]
pub fn get_module(module: ModuleBuilder) -> ModuleBuilder {
    module
        .class::<CrabmagickImage>()
        .function(wrap_function!(crabmagick_process))
        .function(wrap_function!(crabmagick_info))
}

fn parse_format(s: &str) -> PhpResult<OutputFormat> {
    match s.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => Ok(OutputFormat::Jpeg),
        "webp" => Ok(OutputFormat::Webp),
        "webp-lossless" | "webp_lossless" | "webplossless" => Ok(OutputFormat::WebpLossless),
        "png" => Ok(OutputFormat::Png),
        "jxl" => Ok(OutputFormat::Jxl),
        "avif" => Ok(OutputFormat::Avif),
        "tiff" | "tif" => Ok(OutputFormat::Tiff),
        "gif" => Ok(OutputFormat::Gif),
        "bmp" => Ok(OutputFormat::Bmp),
        other => Err(PhpException::default(format!("unknown format: {other}"))),
    }
}
