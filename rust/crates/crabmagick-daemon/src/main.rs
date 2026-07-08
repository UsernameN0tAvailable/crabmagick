use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::Arc;

use crabmagick_core::processor::{
    get_info, process_image, AvifEncodeOptions, ChromaSubsampling, EncodeOptions, GifEncodeOptions,
    JpegEncodeOptions, JxlEncodeOptions, OutputFormat, PngEncodeOptions, PngFilter, ProcessRequest,
    TiffCompression, TiffEncodeOptions, WebpEncodeOptions,
};
use serde_json::Value;

// ── Protocol ─────────────────────────────────────────────────────────────────
//
// Request  (PHP → daemon):  [u32 LE: json_len] [json_bytes]
// Response (daemon → PHP):  [u8: status] [u32 LE: payload_len] [payload_bytes]
//
// status 0 = success, 1 = error
// process success  → raw image bytes
// info    success  → UTF-8 JSON  {"width":N,"height":N}
// error            → UTF-8 error string

fn default_quality() -> u8 {
    85
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let socket_path = parse_socket_arg();

    // Remove stale socket file if it exists
    if Path::new(&socket_path).exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    let listener = UnixListener::bind(&socket_path)
        .unwrap_or_else(|e| fatal(&format!("cannot bind {socket_path}: {e}")));

    // Ensure socket is group/world readable so PHP FPM workers (different uid)
    // can connect. Adjust permissions to 0o666.
    let _ = std::fs::set_permissions(
        &socket_path,
        std::os::unix::fs::PermissionsExt::from_mode(0o666),
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .build()
        .unwrap_or_else(|e| fatal(&format!("rayon pool: {e}")));

    let pool = Arc::new(pool);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let pool = Arc::clone(&pool);
                pool.spawn(move || handle_connection(stream));
            }
            Err(e) => eprintln!("[crabmagick-daemon] accept error: {e}"),
        }
    }
}

// ── Connection handler ────────────────────────────────────────────────────────

fn handle_connection(mut stream: UnixStream) {
    let result = (|| -> io::Result<()> {
        let json = read_frame(&mut stream)?;
        let response = dispatch(&json);
        write_response(&mut stream, response)?;
        Ok(())
    })();

    if let Err(e) = result {
        if e.kind() != io::ErrorKind::UnexpectedEof {
            eprintln!("[crabmagick-daemon] connection error: {e}");
        }
    }
}

fn dispatch(json: &[u8]) -> (u8, Vec<u8>) {
    let req: Value = match serde_json::from_slice(json) {
        Ok(r) => r,
        Err(e) => return (1, format!("invalid request: {e}").into_bytes()),
    };

    match req.get("cmd").and_then(Value::as_str) {
        Some("process") => match parse_process_request(&req) {
            Ok((path, request)) => match process_image(&path, request) {
                Ok(bytes) => (0, bytes),
                Err(e) => (1, e.to_string().into_bytes()),
            },
            Err(error) => (1, error.into_bytes()),
        },
        Some("info") => {
            let path = match get_required_str(&req, "path") {
                Ok(path) => path,
                Err(error) => return (1, error.into_bytes()),
            };
            match get_info(path) {
                Ok(info) => {
                    let json = format!(r#"{{"width":{},"height":{}}}"#, info.width, info.height);
                    (0, json.into_bytes())
                }
                Err(e) => (1, e.to_string().into_bytes()),
            }
        }
        Some(other) => (1, format!("unknown command: {other}").into_bytes()),
        None => (1, b"invalid request: missing cmd".to_vec()),
    }
}

// ── I/O framing ───────────────────────────────────────────────────────────────

fn read_frame(stream: &mut UnixStream) -> io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 64 * 1024 * 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "request too large",
        ));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

fn write_response(stream: &mut UnixStream, (status, payload): (u8, Vec<u8>)) -> io::Result<()> {
    let len = (payload.len() as u32).to_le_bytes();
    stream.write_all(&[status])?;
    stream.write_all(&len)?;
    stream.write_all(&payload)?;
    stream.flush()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_format(s: &str) -> Result<OutputFormat, String> {
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
        other => Err(format!("unknown format: {other}")),
    }
}

fn parse_process_request(request: &Value) -> Result<(String, ProcessRequest), String> {
    let path = get_required_str(request, "path")?.to_string();
    let format = parse_format(
        request
            .get("format")
            .and_then(Value::as_str)
            .unwrap_or("jpeg"),
    )?;

    let encode = parse_encode_options(request, format);
    Ok((
        path,
        ProcessRequest {
            region_left: get_u32(request, "region_x", 0),
            region_top: get_u32(request, "region_y", 0),
            region_width: get_u32(request, "region_w", 0),
            region_height: get_u32(request, "region_h", 0),
            output_width: get_u32(request, "out_w", 0),
            output_height: get_u32(request, "out_h", 0),
            encode,
            page: get_u32(request, "page", 0),
            rotation: get_u16(request, "rotation", 0),
            square_region: get_bool(request, "square", false),
        },
    ))
}

fn parse_encode_options(request: &Value, format: OutputFormat) -> EncodeOptions {
    match format {
        OutputFormat::Jpeg => EncodeOptions::Jpeg(JpegEncodeOptions {
            quality: get_u8(request, "quality", default_quality()).clamp(1, 100),
            progressive: get_bool(request, "progressive", false),
            chroma_subsampling: parse_chroma_subsampling(
                request.get("chroma_subsampling").and_then(Value::as_str),
            ),
            restart_interval: get_u16(request, "restart_interval", 0),
        }),
        OutputFormat::Webp | OutputFormat::WebpLossless => EncodeOptions::Webp(WebpEncodeOptions {
            quality: get_u8(request, "quality", default_quality()).min(100),
            lossless: get_bool(request, "lossless", format == OutputFormat::WebpLossless),
            near_lossless: get_bool(request, "near_lossless", false),
            effort: get_u8(request, "effort", 4),
            alpha_quality: get_u8(request, "alpha_quality", 100).min(100),
        }),
        OutputFormat::Png => EncodeOptions::Png(PngEncodeOptions {
            compression: get_u8(request, "compression", 6).min(9),
            progressive: get_bool(request, "progressive", false),
            filter: parse_png_filter(request.get("filter").and_then(Value::as_str)),
            bitdepth: match get_u8(request, "bitdepth", 8) {
                16 => 16,
                _ => 8,
            },
        }),
        OutputFormat::Jxl => EncodeOptions::Jxl(JxlEncodeOptions {
            quality: get_u8(request, "quality", default_quality()).min(100),
            distance: request
                .get("distance")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            effort: get_u8(request, "effort", 7),
            lossless: get_bool(request, "lossless", false),
            tier: get_u8(request, "tier", 0),
        }),
        OutputFormat::Avif => EncodeOptions::Avif(AvifEncodeOptions {
            quality: get_u8(request, "quality", default_quality()).clamp(1, 100),
            lossless: get_bool(request, "lossless", false),
            effort: get_u8(request, "effort", 4),
        }),
        OutputFormat::Tiff => EncodeOptions::Tiff(TiffEncodeOptions {
            compression: parse_tiff_compression(request.get("compression").and_then(Value::as_str)),
            quality: get_u8(request, "quality", default_quality()).clamp(1, 100),
            predictor: get_bool(request, "predictor", true),
            tiled: get_bool(request, "tiled", false),
            tile_width: get_u16(request, "tile_width", 128),
            tile_height: get_u16(request, "tile_height", 128),
        }),
        OutputFormat::Gif => EncodeOptions::Gif(GifEncodeOptions {
            dither: get_f32(request, "dither", 1.0),
            effort: get_u8(request, "effort", 7),
            bitdepth: get_u8(request, "bitdepth", 8).clamp(1, 8),
        }),
        OutputFormat::Bmp => EncodeOptions::Bmp,
    }
}

fn parse_chroma_subsampling(value: Option<&str>) -> ChromaSubsampling {
    match value.unwrap_or("auto").to_ascii_lowercase().as_str() {
        "420" => ChromaSubsampling::Cs420,
        "422" => ChromaSubsampling::Cs422,
        "444" => ChromaSubsampling::Cs444,
        _ => ChromaSubsampling::Auto,
    }
}

fn parse_png_filter(value: Option<&str>) -> PngFilter {
    match value.unwrap_or("all").to_ascii_lowercase().as_str() {
        "none" => PngFilter::None,
        "sub" => PngFilter::Sub,
        "up" => PngFilter::Up,
        "avg" => PngFilter::Avg,
        "paeth" => PngFilter::Paeth,
        _ => PngFilter::All,
    }
}

fn parse_tiff_compression(value: Option<&str>) -> TiffCompression {
    match value.unwrap_or("lzw").to_ascii_lowercase().as_str() {
        "none" => TiffCompression::None,
        "deflate" => TiffCompression::Deflate,
        "jpeg" => TiffCompression::Jpeg,
        "packbits" => TiffCompression::Packbits,
        _ => TiffCompression::Lzw,
    }
}

fn get_required_str<'a>(request: &'a Value, field: &str) -> Result<&'a str, String> {
    request
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("invalid request: missing {field}"))
}

fn get_u32(request: &Value, field: &str, default: u32) -> u32 {
    request
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(default)
}

fn get_u16(request: &Value, field: &str, default: u16) -> u16 {
    request
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| u16::try_from(value).ok())
        .unwrap_or(default)
}

fn get_u8(request: &Value, field: &str, default: u8) -> u8 {
    request
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| u8::try_from(value).ok())
        .unwrap_or(default)
}

fn get_bool(request: &Value, field: &str, default: bool) -> bool {
    request
        .get(field)
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

fn get_f32(request: &Value, field: &str, default: f32) -> f32 {
    request
        .get(field)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or(default)
}

fn parse_socket_arg() -> String {
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--socket" {
            if let Some(path) = args.get(i + 1) {
                return path.clone();
            }
        }
    }
    // Fallback: /tmp/crabmagick-<uid>.sock
    let uid = unsafe { libc::getuid() };
    format!("/tmp/crabmagick-{uid}.sock")
}

fn fatal(msg: &str) -> ! {
    eprintln!("[crabmagick-daemon] fatal: {msg}");
    std::process::exit(1);
}
