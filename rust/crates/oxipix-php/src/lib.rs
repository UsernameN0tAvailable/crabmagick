use std::collections::HashMap;

use ext_php_rs::binary::Binary;
use ext_php_rs::exception::PhpException;
use ext_php_rs::prelude::*;
use ext_php_rs::types::ZendClassObject;
use oxipix_core::processor::{OutputFormat, ProcessRequest, get_info, init, process_image};

#[php_class]
#[php(name = "Oxipix\\Image")]
pub struct OxipixImage {
    path: String,
    region_x: u32,
    region_y: u32,
    region_w: u32,
    region_h: u32,
    out_w: u32,
    out_h: u32,
    rotation: u16,
    square_region: bool,
}

#[php_impl]
#[php(change_method_case = "none")]
impl OxipixImage {
    pub fn __construct(path: String) -> PhpResult<Self> {
        Ok(Self {
            path,
            region_x: 0,
            region_y: 0,
            region_w: 0,
            region_h: 0,
            out_w: 0,
            out_h: 0,
            rotation: 0,
            square_region: false,
        })
    }

    pub fn region<'a>(
        self_: &'a mut ZendClassObject<OxipixImage>,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    ) -> &'a mut ZendClassObject<OxipixImage> {
        self_.region_x = x;
        self_.region_y = y;
        self_.region_w = w;
        self_.region_h = h;
        self_
    }

    pub fn resize<'a>(
        self_: &'a mut ZendClassObject<OxipixImage>,
        w: u32,
        h: u32,
    ) -> &'a mut ZendClassObject<OxipixImage> {
        self_.out_w = w;
        self_.out_h = h;
        self_
    }

    pub fn rotate<'a>(
        self_: &'a mut ZendClassObject<OxipixImage>,
        degrees: u16,
    ) -> &'a mut ZendClassObject<OxipixImage> {
        self_.rotation = degrees;
        self_
    }

    pub fn square<'a>(
        self_: &'a mut ZendClassObject<OxipixImage>,
    ) -> &'a mut ZendClassObject<OxipixImage> {
        self_.square_region = true;
        self_
    }

    pub fn encode(&self, format: String, quality: u8) -> PhpResult<Binary<u8>> {
        let req = ProcessRequest {
            region_x: self.region_x,
            region_y: self.region_y,
            region_w: self.region_w,
            region_h: self.region_h,
            out_w: self.out_w,
            out_h: self.out_h,
            format: parse_format(&format)?,
            quality,
            rotation: self.rotation,
            square_region: self.square_region,
        };

        process_image(&self.path, req)
            .map(Binary::from)
            .map_err(|e| PhpException::default(e.to_string()))
    }

    #[php(name = "getInfo")]
    pub fn get_info_method(&self) -> PhpResult<HashMap<String, u32>> {
        get_info(&self.path)
            .map(|i| HashMap::from([("width".to_string(), i.width), ("height".to_string(), i.height)]))
            .map_err(|e| PhpException::default(e.to_string()))
    }

    /// Static method: returns true if the extension is loaded (it always is if this runs).
    #[php(name = "isAvailable")]
    pub fn is_available() -> bool {
        true
    }
}

#[php_function]
#[php(name = "Oxipix\\process")]
pub fn oxipix_process(
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
    let req = ProcessRequest {
        region_x: rx,
        region_y: ry,
        region_w: rw,
        region_h: rh,
        out_w: ow,
        out_h: oh,
        format: parse_format(&format)?,
        quality,
        rotation: 0,
        square_region: false,
    };

    process_image(&path, req)
        .map(Binary::from)
        .map_err(|e| PhpException::default(e.to_string()))
}

#[php_function]
#[php(name = "Oxipix\\info")]
pub fn oxipix_info(path: String) -> PhpResult<HashMap<String, u32>> {
    get_info(&path)
        .map(|i| HashMap::from([("width".to_string(), i.width), ("height".to_string(), i.height)]))
        .map_err(|e| PhpException::default(e.to_string()))
}

#[php_module]
pub fn get_module(module: ModuleBuilder) -> ModuleBuilder {
    init(512, 512);
    module
        .class::<OxipixImage>()
        .function(wrap_function!(oxipix_process))
        .function(wrap_function!(oxipix_info))
}

fn parse_format(s: &str) -> PhpResult<OutputFormat> {
    match s.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => Ok(OutputFormat::Jpeg),
        "webp" => Ok(OutputFormat::Webp),
        "png" => Ok(OutputFormat::Png),
        "jxl" => Ok(OutputFormat::Jxl),
        other => Err(PhpException::default(format!("unknown format: {other}"))),
    }
}
