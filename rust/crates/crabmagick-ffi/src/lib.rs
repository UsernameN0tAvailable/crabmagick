use crabmagick_core::processor::{get_info, process_image, OutputFormat, ProcessRequest};
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::ptr;

/// Output format codes — must match oxipix.h and PHP Runtime::FORMAT_MAP.
#[repr(C)]
pub enum OxipixFormat {
    Jpeg = 0,
    Webp = 1,
    Png = 2,
    Jxl = 3,
    Avif = 4,
}

/// Must match the layout in ffi/oxipix.h exactly (same field order, same types).
#[repr(C)]
pub struct OxipixRequest {
    pub region_x: u32,
    pub region_y: u32,
    pub region_w: u32,
    pub region_h: u32,
    pub out_w: u32,
    pub out_h: u32,
    pub quality: u8,
    pub format: c_int,
    pub page: u32,
    pub rotation: u16,
    pub square_region: u8,
}

#[repr(C)]
pub struct OxipixImageInfo {
    pub width: u32,
    pub height: u32,
}

/// Read dimensions from an image file without full decode.
/// Returns 0 on success, -1 on error (error_message set if non-null).
#[no_mangle]
pub unsafe extern "C" fn oxipix_get_info(
    path: *const c_char,
    info: *mut OxipixImageInfo,
    error_message: *mut *mut c_char,
) -> c_int {
    clear_error(error_message);

    if info.is_null() {
        set_error(error_message, "null info pointer");
        return -1;
    }

    let path_str = match path_to_str(path) {
        Ok(s) => s,
        Err(e) => {
            set_error(error_message, &e);
            return -1;
        }
    };

    match get_info(path_str) {
        Ok(i) => {
            (*info).width = i.width;
            (*info).height = i.height;
            0
        }
        Err(e) => {
            set_error(error_message, &e.to_string());
            -1
        }
    }
}

/// Decode, crop/resize/rotate, and encode an image.
/// On success: sets *out_data/*out_len, returns 0. Caller must call oxipix_free(*out_data).
/// On error: sets *error_message (if non-null), returns -1.
#[no_mangle]
pub unsafe extern "C" fn oxipix_process(
    path: *const c_char,
    request: *const OxipixRequest,
    out_data: *mut *mut u8,
    out_len: *mut usize,
    error_message: *mut *mut c_char,
) -> c_int {
    clear_error(error_message);

    if request.is_null() {
        set_error(error_message, "null request pointer");
        return -1;
    }
    if out_data.is_null() {
        set_error(error_message, "null out_data pointer");
        return -1;
    }
    if out_len.is_null() {
        set_error(error_message, "null out_len pointer");
        return -1;
    }

    *out_data = ptr::null_mut();
    *out_len = 0;

    let path_str = match path_to_str(path) {
        Ok(s) => s,
        Err(e) => {
            set_error(error_message, &e);
            return -1;
        }
    };

    let req = &*request;
    let format = match req.format {
        x if x == OxipixFormat::Jpeg as c_int => OutputFormat::Jpeg,
        x if x == OxipixFormat::Webp as c_int => OutputFormat::Webp,
        x if x == OxipixFormat::Png as c_int => OutputFormat::Png,
        x if x == OxipixFormat::Jxl as c_int => OutputFormat::Jxl,
        x if x == OxipixFormat::Avif as c_int => OutputFormat::Avif,
        _ => {
            set_error(error_message, "unknown format code");
            return -1;
        }
    };

    let proc_req = ProcessRequest {
        region_x: req.region_x,
        region_y: req.region_y,
        region_w: req.region_w,
        region_h: req.region_h,
        out_w: req.out_w,
        out_h: req.out_h,
        format,
        quality: req.quality,
        page: req.page,
        rotation: req.rotation,
        square_region: req.square_region != 0,
    };

    match process_image(path_str, proc_req) {
        Ok(bytes) => match alloc_buffer(&bytes) {
            Some(ptr) => {
                *out_data = ptr;
                *out_len = bytes.len();
                0
            }
            None => {
                set_error(error_message, "allocation failed");
                -1
            }
        },
        Err(e) => {
            set_error(error_message, &e.to_string());
            -1
        }
    }
}

/// Free a buffer returned by oxipix_process or an error string from any function.
/// Also safe to call with a null pointer.
#[no_mangle]
pub unsafe extern "C" fn oxipix_free(ptr: *mut libc::c_void) {
    if !ptr.is_null() {
        libc::free(ptr);
    }
}

unsafe fn path_to_str<'a>(ptr: *const c_char) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err("null path pointer".into());
    }

    CStr::from_ptr(ptr)
        .to_str()
        .map_err(|e| format!("invalid path utf-8: {e}"))
}

unsafe fn clear_error(dst: *mut *mut c_char) {
    if !dst.is_null() {
        *dst = ptr::null_mut();
    }
}

unsafe fn set_error(dst: *mut *mut c_char, msg: &str) {
    if dst.is_null() {
        return;
    }

    *dst = alloc_c_string(msg).unwrap_or(ptr::null_mut());
}

unsafe fn alloc_buffer(bytes: &[u8]) -> Option<*mut u8> {
    let len = bytes.len();
    let alloc_len = len.max(1);
    let ptr = libc::malloc(alloc_len) as *mut u8;
    if ptr.is_null() {
        return None;
    }

    if len > 0 {
        ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len);
    }

    Some(ptr)
}

unsafe fn alloc_c_string(msg: &str) -> Option<*mut c_char> {
    let mut bytes = msg
        .as_bytes()
        .iter()
        .copied()
        .filter(|b| *b != 0)
        .collect::<Vec<_>>();
    bytes.push(0);

    let ptr = libc::malloc(bytes.len()) as *mut c_char;
    if ptr.is_null() {
        return None;
    }

    ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, ptr, bytes.len());
    Some(ptr)
}
