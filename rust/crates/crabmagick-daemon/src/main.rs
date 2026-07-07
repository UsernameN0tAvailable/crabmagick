use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::Arc;

use crabmagick_core::processor::{get_info, process_image, OutputFormat, ProcessRequest};
use serde::Deserialize;

// ── Protocol ─────────────────────────────────────────────────────────────────
//
// Request  (PHP → daemon):  [u32 LE: json_len] [json_bytes]
// Response (daemon → PHP):  [u8: status] [u32 LE: payload_len] [payload_bytes]
//
// status 0 = success, 1 = error
// process success  → raw image bytes
// info    success  → UTF-8 JSON  {"width":N,"height":N}
// error            → UTF-8 error string

#[derive(Deserialize)]
#[serde(tag = "cmd")]
enum Request {
    #[serde(rename = "process")]
    Process {
        path: String,
        #[serde(default)]
        region_x: u32,
        #[serde(default)]
        region_y: u32,
        #[serde(default)]
        region_w: u32,
        #[serde(default)]
        region_h: u32,
        #[serde(default)]
        out_w: u32,
        #[serde(default)]
        out_h: u32,
        #[serde(default = "default_format")]
        format: String,
        #[serde(default = "default_quality")]
        quality: u8,
        #[serde(default)]
        page: u32,
        #[serde(default)]
        rotation: u16,
        #[serde(default)]
        square: bool,
    },
    #[serde(rename = "info")]
    Info {
        path: String,
        #[serde(default)]
        #[allow(dead_code)] // reserved for future page-aware info; field kept for protocol compat
        page: u32,
    },
}

fn default_format() -> String {
    "jpeg".to_string()
}
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
    let req: Request = match serde_json::from_slice(json) {
        Ok(r) => r,
        Err(e) => return (1, format!("invalid request: {e}").into_bytes()),
    };

    match req {
        Request::Process {
            path,
            region_x, region_y, region_w, region_h,
            out_w, out_h,
            format,
            quality,
            page,
            rotation,
            square,
        } => {
            let fmt = match parse_format(&format) {
                Ok(f) => f,
                Err(e) => return (1, e.into_bytes()),
            };
            let request = ProcessRequest {
                region_left: region_x,
                region_top: region_y,
                region_width: region_w,
                region_height: region_h,
                output_width: out_w,
                output_height: out_h,
                output_format: fmt,
                quality,
                page,
                rotation,
                square_region: square,
            };
            match process_image(&path, request) {
                Ok(bytes) => (0, bytes),
                Err(e) => (1, e.to_string().into_bytes()),
            }
        }
        Request::Info { path, page: _ } => {
            match get_info(&path) {
                Ok(info) => {
                    let json = format!(r#"{{"width":{},"height":{}}}"#, info.width, info.height);
                    (0, json.into_bytes())
                }
                Err(e) => (1, e.to_string().into_bytes()),
            }
        }
    }
}

// ── I/O framing ───────────────────────────────────────────────────────────────

fn read_frame(stream: &mut UnixStream) -> io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 64 * 1024 * 1024 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "request too large"));
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
        "png" => Ok(OutputFormat::Png),
        "jxl" => Ok(OutputFormat::Jxl),
        "avif" => Ok(OutputFormat::Avif),
        other => Err(format!("unknown format: {other}")),
    }
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
