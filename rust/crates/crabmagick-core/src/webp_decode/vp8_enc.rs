use std::error::Error;
use std::fmt::{Display, Formatter};

use image::RgbImage;
use rayon::prelude::*;

use crate::webp_decode::transform;
use crate::webp_decode::vp8::{
    AC_QUANT, COEFF_BANDS, COEFF_PROBS, COEFF_UPDATE_PROBS, DCT_0, DCT_1, DCT_2, DCT_3, DCT_4,
    DCT_CAT1, DCT_CAT2, DCT_CAT3, DCT_CAT4, DCT_CAT5, DCT_CAT6, DCT_CAT_BASE, DCT_EOB, DC_QUANT,
    KEYFRAME_BPRED_MODE_PROBS, PROB_DCT_CAT, ZIGZAG,
};

const KEYFRAME_YMODE_B_PRED_PROB: u8 = 145;
const KEYFRAME_UV_MODE_PROBS: [u8; 3] = [142, 114, 183];

const LUMA_BLOCKS_PER_MB: usize = 16;
const CHROMA_BLOCKS_PER_MB: usize = 4;
const B_DC_PRED: u8 = 0;
const B_TM_PRED: u8 = 1;
const B_VE_PRED: u8 = 2;
const B_HE_PRED: u8 = 3;
const B_LD_PRED: u8 = 4;
const B_RD_PRED: u8 = 5;
const B_VR_PRED: u8 = 6;
const B_VL_PRED: u8 = 7;
const B_HD_PRED: u8 = 8;
const B_HU_PRED: u8 = 9;
const LUMA_WS_STRIDE: usize = 1 + 16 + 4;
const LUMA_WS_SIZE: usize = (1 + 16) * LUMA_WS_STRIDE;

/// Errors returned by the lossy VP8/WebP encoder.
#[derive(Debug, Clone)]
pub struct WebPEncodeError(String);

impl Display for WebPEncodeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for WebPEncodeError {}

#[derive(Clone)]
struct MacroBlockData {
    y_blocks: [[i16; 16]; LUMA_BLOCKS_PER_MB],
    u_blocks: [[i16; 16]; CHROMA_BLOCKS_PER_MB],
    v_blocks: [[i16; 16]; CHROMA_BLOCKS_PER_MB],
    y_nonzero: [bool; LUMA_BLOCKS_PER_MB],
    u_nonzero: [bool; CHROMA_BLOCKS_PER_MB],
    v_nonzero: [bool; CHROMA_BLOCKS_PER_MB],
    bpred_modes: [u8; LUMA_BLOCKS_PER_MB],
    chroma_mode: u8,
}

pub(crate) struct BoolEncoder {
    range: u32,
    low: u64,
    bits: Vec<bool>,
}

impl BoolEncoder {
    pub fn new() -> Self {
        Self {
            range: 255,
            low: 0,
            bits: Vec::new(),
        }
    }

    pub fn write_bool(&mut self, bit: bool, prob: u8) {
        let split = 1 + (((self.range - 1) as u64 * prob as u64) >> 8) as u32;
        if bit {
            self.low += split as u64;
            self.range -= split;
        } else {
            self.range = split;
        }

        while self.range < 128 {
            if self.low >= 0x100000000u64 {
                let mut i = self.bits.len() as isize - 1;
                let mut carry = 1u32;
                while carry > 0 && i >= 0 {
                    if !self.bits[i as usize] {
                        self.bits[i as usize] = true;
                        carry = 0;
                    } else {
                        self.bits[i as usize] = false;
                    }
                    i -= 1;
                }
                self.low -= 0x100000000u64;
            }

            self.bits.push((self.low & 0x80000000u64) != 0);
            self.low = (self.low << 1) & 0xFFFFFFFFu64;
            self.range <<= 1;
        }
    }

    pub fn write_flag(&mut self, bit: bool) {
        self.write_bool(bit, 128);
    }

    pub fn write_literal(&mut self, n: u32, value: u32) {
        let mut b = 1u32 << (n - 1);
        while b > 0 {
            self.write_flag((value & b) != 0);
            b >>= 1;
        }
    }

    #[allow(dead_code)]
    pub fn write_signed(&mut self, n: u32, value: i32) {
        self.write_literal(n, value.unsigned_abs());
        if value != 0 {
            self.write_flag(value < 0);
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.write_flag(false);
        }

        self.bits
            .chunks(8)
            .map(|chunk| {
                chunk.iter().enumerate().fold(
                    0u8,
                    |b, (i, &bit)| {
                        if bit {
                            b | (0x80u8 >> i)
                        } else {
                            b
                        }
                    },
                )
            })
            .collect()
    }
}

/// Encode an RGB image as a lossy VP8-based WebP image.
#[inline]
pub(crate) fn encode_lossy_webp(img: &RgbImage, quality: u8) -> Result<Vec<u8>, WebPEncodeError> {
    let width = img.width();
    let height = img.height();

    if width == 0 || height == 0 {
        return Err(WebPEncodeError(
            "VP8 encoder does not support empty images".to_owned(),
        ));
    }
    if width > 16_383 || height > 16_383 {
        return Err(WebPEncodeError(
            "VP8 encoder only supports dimensions up to 16383x16383".to_owned(),
        ));
    }

    let q_idx = quality_to_q_index(quality);
    let (y_plane, u_plane, v_plane) = rgb_to_yuv420(width, height, img.as_raw());

    let mb_width = width.div_ceil(16) as usize;
    let mb_height = height.div_ceil(16) as usize;
    let y_stride = mb_width * 16;
    let y_rows = mb_height * 16;
    let uv_stride = mb_width * 8;
    let uv_rows = mb_height * 8;

    let padded_y = pad_plane_clamped(&y_plane, width as usize, height as usize, y_stride, y_rows);
    let padded_u = pad_plane_clamped(
        &u_plane,
        width.div_ceil(2) as usize,
        height.div_ceil(2) as usize,
        uv_stride,
        uv_rows,
    );
    let padded_v = pad_plane_clamped(
        &v_plane,
        width.div_ceil(2) as usize,
        height.div_ceil(2) as usize,
        uv_stride,
        uv_rows,
    );

    let mut recon_y = vec![0u8; y_stride * y_rows];
    let mut recon_u = vec![0u8; uv_stride * uv_rows];
    let mut recon_v = vec![0u8; uv_stride * uv_rows];

    let mut macroblocks = Vec::with_capacity(mb_width * mb_height);

    for mb_y in 0..mb_height {
        for mb_x in 0..mb_width {
            macroblocks.push(encode_macroblock(
                &padded_y,
                &padded_u,
                &padded_v,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
                y_stride,
                uv_stride,
                mb_x,
                mb_y,
                q_idx,
            ));
        }
    }

    let first_partition = encode_first_partition(q_idx, &macroblocks, mb_width);
    let coeff_partition = encode_coeff_partition(&macroblocks, mb_width);

    let vp8_frame = build_vp8_frame(width, height, &first_partition, &coeff_partition)?;
    Ok(wrap_riff_webp(&vp8_frame))
}

#[inline(always)]
fn quality_to_q_index(quality: u8) -> usize {
    let q = quality.min(100) as usize;
    ((127 * (100 - q) + 79) / 80).min(127)
}

#[inline]
fn rgb_to_yuv420(width: u32, height: u32, rgb: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let uv_w = w.div_ceil(2);
    let uv_h = h.div_ceil(2);

    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; uv_w * uv_h];
    let mut v = vec![0u8; uv_w * uv_h];

    for row in 0..h {
        for col in 0..w {
            let rgb_off = (row * w + col) * 3;
            let r = rgb[rgb_off] as i32;
            let g = rgb[rgb_off + 1] as i32;
            let b = rgb[rgb_off + 2] as i32;

            let yv = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            let uv = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
            let vv = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;

            y[row * w + col] = yv.clamp(0, 255) as u8;
            if row % 2 == 0 && col % 2 == 0 {
                let uv_idx = (row / 2) * uv_w + col / 2;
                u[uv_idx] = uv.clamp(0, 255) as u8;
                v[uv_idx] = vv.clamp(0, 255) as u8;
            }
        }
    }

    (y, u, v)
}

#[inline]
fn pad_plane_clamped(
    src: &[u8],
    width: usize,
    height: usize,
    padded_width: usize,
    padded_height: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; padded_width * padded_height];
    let last_col = width.saturating_sub(1);
    let last_row = height.saturating_sub(1);

    out.par_chunks_mut(padded_width)
        .enumerate()
        .for_each(|(y, row)| {
            let sy = y.min(last_row);
            let src_row = &src[sy * width..sy * width + width];
            row[..width].copy_from_slice(src_row);
            row[width..].fill(src_row[last_col]);
        });

    out
}

fn fdct4x4(input: &[i16; 16]) -> [i16; 16] {
    let mut tmp = [0i32; 16];
    let mut out = [0i16; 16];

    for i in 0..4usize {
        let a = input[i * 4] as i32 + input[i * 4 + 3] as i32;
        let b = input[i * 4 + 1] as i32 + input[i * 4 + 2] as i32;
        let c = input[i * 4 + 1] as i32 - input[i * 4 + 2] as i32;
        let d = input[i * 4] as i32 - input[i * 4 + 3] as i32;
        tmp[i] = (a + b) * 8;
        tmp[4 + i] = (d * 2217 + c * 5352 + 14500) >> 12;
        tmp[8 + i] = (a - b) * 8;
        tmp[12 + i] = (d * 5352 - c * 2217 + 7500) >> 12;
    }

    for i in 0..4usize {
        let a = tmp[i] + tmp[12 + i];
        let b = tmp[4 + i] + tmp[8 + i];
        let c = tmp[4 + i] - tmp[8 + i];
        let d = tmp[i] - tmp[12 + i];
        out[i] = ((a + b + 7) >> 4) as i16;
        out[4 + i] = ((d * 2217 + c * 5352 + 12000) >> 16) as i16 + (d != 0) as i16;
        out[8 + i] = ((a - b + 7) >> 4) as i16;
        out[12 + i] = ((d * 5352 - c * 2217 + 51000) >> 16) as i16;
    }

    out
}

fn quantize_block(dct: &[i16; 16], q_idx: usize) -> ([i16; 16], bool) {
    let dc_q = DC_QUANT[q_idx] as i32;
    let ac_q = AC_QUANT[q_idx] as i32;
    let mut out = [0i16; 16];
    let mut any_nonzero = false;

    for (i, &coeff) in dct.iter().enumerate() {
        let q = if i == 0 { dc_q } else { ac_q };
        let v = coeff as i32;
        let sign = v.signum();
        let abs_v = v.abs();
        let qv = (abs_v + q / 2) / q;
        let qv = qv.min(2047) as i16;
        out[i] = (sign as i16) * qv;
        if qv != 0 {
            any_nonzero = true;
        }
    }

    (out, any_nonzero)
}

fn dequantize_block(coeffs: &[i16; 16], q_idx: usize) -> [i32; 16] {
    let dc_q = DC_QUANT[q_idx] as i32;
    let ac_q = AC_QUANT[q_idx] as i32;
    let mut out = [0i32; 16];

    for (i, coeff) in coeffs.iter().enumerate() {
        let q = if i == 0 { dc_q } else { ac_q };
        out[i] = i32::from(*coeff) * q;
    }

    out
}

fn encode_macroblock(
    src_y: &[u8],
    src_u: &[u8],
    src_v: &[u8],
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
    y_stride: usize,
    uv_stride: usize,
    mb_x: usize,
    mb_y: usize,
    q_idx: usize,
) -> MacroBlockData {
    let mut data = MacroBlockData {
        y_blocks: [[0i16; 16]; LUMA_BLOCKS_PER_MB],
        u_blocks: [[0i16; 16]; CHROMA_BLOCKS_PER_MB],
        v_blocks: [[0i16; 16]; CHROMA_BLOCKS_PER_MB],
        y_nonzero: [false; LUMA_BLOCKS_PER_MB],
        u_nonzero: [false; CHROMA_BLOCKS_PER_MB],
        v_nonzero: [false; CHROMA_BLOCKS_PER_MB],
        bpred_modes: [B_DC_PRED; LUMA_BLOCKS_PER_MB],
        chroma_mode: 0,
    };
    let mut luma_ws = init_luma_workspace(recon_y, y_stride, mb_x, mb_y);

    for sub_y in 0..4 {
        for sub_x in 0..4 {
            let block_index = sub_x + sub_y * 4;
            let x = mb_x * 16 + sub_x * 4;
            let y = mb_y * 16 + sub_y * 4;
            let x0 = sub_x * 4 + 1;
            let y0 = sub_y * 4 + 1;
            let src = read_block_4x4(src_y, y_stride, x, y);
            let (mode, pred) = best_luma_mode_4x4(&src, &luma_ws, x0, y0);
            let residual = residual_from_block_prediction(&src, &pred);
            let dct = fdct4x4(&residual);
            let (coeffs, nonzero) = quantize_block(&dct, q_idx);
            let recon = reconstruct_from_block_coeffs(&coeffs, q_idx, &pred);
            write_workspace_block_4x4(&mut luma_ws, x0, y0, &recon);
            data.y_blocks[block_index] = coeffs;
            data.y_nonzero[block_index] = nonzero;
            data.bpred_modes[block_index] = mode;
        }
    }

    for row in 0..16 {
        let dst = (mb_y * 16 + row) * y_stride + mb_x * 16;
        let src = (row + 1) * LUMA_WS_STRIDE + 1;
        recon_y[dst..dst + 16].copy_from_slice(&luma_ws[src..src + 16]);
    }

    let chroma_x = mb_x * 8;
    let chroma_y = mb_y * 8;
    let (chroma_mode, pred_u, pred_v) = best_chroma_mode_8x8(
        src_u, src_v, recon_u, recon_v, uv_stride, chroma_x, chroma_y,
    );
    data.chroma_mode = chroma_mode;

    for sub_y in 0..2 {
        for sub_x in 0..2 {
            let block_index = sub_x + sub_y * 2;
            let x = chroma_x + sub_x * 4;
            let y = chroma_y + sub_y * 4;
            let pred_offset = sub_y * 32 + sub_x * 4;

            let src_block_u = read_block_4x4(src_u, uv_stride, x, y);
            let pred_block_u = plane_block_4x4(&pred_u, pred_offset);
            let residual_u = residual_from_block_prediction(&src_block_u, &pred_block_u);
            let dct_u = fdct4x4(&residual_u);
            let (coeffs_u, nonzero_u) = quantize_block(&dct_u, q_idx);
            let recon_block_u = reconstruct_from_block_coeffs(&coeffs_u, q_idx, &pred_block_u);
            write_block_4x4(recon_u, uv_stride, x, y, &recon_block_u);
            data.u_blocks[block_index] = coeffs_u;
            data.u_nonzero[block_index] = nonzero_u;

            let src_block_v = read_block_4x4(src_v, uv_stride, x, y);
            let pred_block_v = plane_block_4x4(&pred_v, pred_offset);
            let residual_v = residual_from_block_prediction(&src_block_v, &pred_block_v);
            let dct_v = fdct4x4(&residual_v);
            let (coeffs_v, nonzero_v) = quantize_block(&dct_v, q_idx);
            let recon_block_v = reconstruct_from_block_coeffs(&coeffs_v, q_idx, &pred_block_v);
            write_block_4x4(recon_v, uv_stride, x, y, &recon_block_v);
            data.v_blocks[block_index] = coeffs_v;
            data.v_nonzero[block_index] = nonzero_v;
        }
    }

    data
}

fn avg3(left: u8, center: u8, right: u8) -> u8 {
    ((u16::from(left) + 2 * u16::from(center) + u16::from(right) + 2) >> 2) as u8
}

fn avg2(left: u8, right: u8) -> u8 {
    ((u16::from(left) + u16::from(right) + 1) >> 1) as u8
}

fn init_luma_workspace(
    recon: &[u8],
    stride: usize,
    mb_x: usize,
    mb_y: usize,
) -> [u8; LUMA_WS_SIZE] {
    let mut ws = [0u8; LUMA_WS_SIZE];
    let base_x = mb_x * 16;
    let base_y = mb_y * 16;

    if mb_y == 0 {
        ws[1..LUMA_WS_STRIDE].fill(127);
    } else {
        let top_row = (base_y - 1) * stride + base_x;
        ws[1..17].copy_from_slice(&recon[top_row..top_row + 16]);
        let fill = recon[top_row + 15];
        if base_x + 20 <= stride {
            ws[17..21].copy_from_slice(&recon[top_row + 16..top_row + 20]);
        } else {
            ws[17..21].fill(fill);
        }
    }

    for i in 17..LUMA_WS_STRIDE {
        ws[4 * LUMA_WS_STRIDE + i] = ws[i];
        ws[8 * LUMA_WS_STRIDE + i] = ws[i];
        ws[12 * LUMA_WS_STRIDE + i] = ws[i];
    }

    if mb_x == 0 {
        for y in 0..16 {
            ws[(y + 1) * LUMA_WS_STRIDE] = 129;
        }
    } else {
        for y in 0..16 {
            ws[(y + 1) * LUMA_WS_STRIDE] = recon[(base_y + y) * stride + base_x - 1];
        }
    }

    ws[0] = if mb_y == 0 {
        127
    } else if mb_x == 0 {
        129
    } else {
        recon[(base_y - 1) * stride + base_x - 1]
    };

    ws
}

fn predict_luma_4x4(mode: u8, ws: &[u8], x0: usize, y0: usize) -> [u8; 16] {
    let mut out = [0u8; 16];
    let top = (y0 - 1) * LUMA_WS_STRIDE + x0;
    let left = y0 * LUMA_WS_STRIDE + x0 - 1;
    let top_left = ws[top - 1];

    match mode {
        B_DC_PRED => {
            let mut sum = 4u32;
            for dx in 0..4 {
                sum += u32::from(ws[top + dx]);
            }
            for dy in 0..4 {
                sum += u32::from(ws[left + dy * LUMA_WS_STRIDE]);
            }
            out.fill((sum >> 3) as u8);
        }
        B_TM_PRED => {
            for dy in 0..4 {
                let left_val = i32::from(ws[left + dy * LUMA_WS_STRIDE]);
                for dx in 0..4 {
                    let top_val = i32::from(ws[top + dx]);
                    out[dy * 4 + dx] =
                        (left_val + top_val - i32::from(top_left)).clamp(0, 255) as u8;
                }
            }
        }
        B_VE_PRED => {
            let row = [
                avg3(top_left, ws[top], ws[top + 1]),
                avg3(ws[top], ws[top + 1], ws[top + 2]),
                avg3(ws[top + 1], ws[top + 2], ws[top + 3]),
                avg3(ws[top + 2], ws[top + 3], ws[top + 4]),
            ];
            for dy in 0..4 {
                out[dy * 4..dy * 4 + 4].copy_from_slice(&row);
            }
        }
        B_HE_PRED => {
            let rows = [
                avg3(top_left, ws[left], ws[left + LUMA_WS_STRIDE]),
                avg3(
                    ws[left],
                    ws[left + LUMA_WS_STRIDE],
                    ws[left + 2 * LUMA_WS_STRIDE],
                ),
                avg3(
                    ws[left + LUMA_WS_STRIDE],
                    ws[left + 2 * LUMA_WS_STRIDE],
                    ws[left + 3 * LUMA_WS_STRIDE],
                ),
                avg3(
                    ws[left + 2 * LUMA_WS_STRIDE],
                    ws[left + 3 * LUMA_WS_STRIDE],
                    ws[left + 3 * LUMA_WS_STRIDE],
                ),
            ];
            for (dy, &value) in rows.iter().enumerate() {
                out[dy * 4..dy * 4 + 4].fill(value);
            }
        }
        B_LD_PRED => {
            let avgs = [
                avg3(ws[top], ws[top + 1], ws[top + 2]),
                avg3(ws[top + 1], ws[top + 2], ws[top + 3]),
                avg3(ws[top + 2], ws[top + 3], ws[top + 4]),
                avg3(ws[top + 3], ws[top + 4], ws[top + 5]),
                avg3(ws[top + 4], ws[top + 5], ws[top + 6]),
                avg3(ws[top + 5], ws[top + 6], ws[top + 7]),
                avg3(ws[top + 6], ws[top + 7], ws[top + 7]),
            ];
            for dy in 0..4 {
                out[dy * 4..dy * 4 + 4].copy_from_slice(&avgs[dy..dy + 4]);
            }
        }
        B_RD_PRED => {
            let e0 = ws[left + 3 * LUMA_WS_STRIDE];
            let e1 = ws[left + 2 * LUMA_WS_STRIDE];
            let e2 = ws[left + LUMA_WS_STRIDE];
            let e3 = ws[left];
            let e4 = top_left;
            let e5 = ws[top];
            let e6 = ws[top + 1];
            let e7 = ws[top + 2];
            let e8 = ws[top + 3];
            let avgs = [
                avg3(e0, e1, e2),
                avg3(e1, e2, e3),
                avg3(e2, e3, e4),
                avg3(e3, e4, e5),
                avg3(e4, e5, e6),
                avg3(e5, e6, e7),
                avg3(e6, e7, e8),
            ];
            for dy in 0..4 {
                out[dy * 4..dy * 4 + 4].copy_from_slice(&avgs[3 - dy..7 - dy]);
            }
        }
        B_VR_PRED => {
            let e1 = ws[left + 2 * LUMA_WS_STRIDE];
            let e2 = ws[left + LUMA_WS_STRIDE];
            let e3 = ws[left];
            let e4 = top_left;
            let e5 = ws[top];
            let e6 = ws[top + 1];
            let e7 = ws[top + 2];
            let e8 = ws[top + 3];
            out[12] = avg3(e1, e2, e3);
            out[8] = avg3(e2, e3, e4);
            out[13] = avg3(e3, e4, e5);
            out[4] = avg3(e3, e4, e5);
            out[9] = avg2(e4, e5);
            out[0] = avg2(e4, e5);
            out[14] = avg3(e4, e5, e6);
            out[5] = avg3(e4, e5, e6);
            out[10] = avg2(e5, e6);
            out[1] = avg2(e5, e6);
            out[15] = avg3(e5, e6, e7);
            out[6] = avg3(e5, e6, e7);
            out[11] = avg2(e6, e7);
            out[2] = avg2(e6, e7);
            out[7] = avg3(e6, e7, e8);
            out[3] = avg2(e7, e8);
        }
        B_VL_PRED => {
            let a0 = ws[top];
            let a1 = ws[top + 1];
            let a2 = ws[top + 2];
            let a3 = ws[top + 3];
            let a4 = ws[top + 4];
            let a5 = ws[top + 5];
            let a6 = ws[top + 6];
            let a7 = ws[top + 7];
            out[0] = avg2(a0, a1);
            out[4] = avg3(a0, a1, a2);
            out[8] = avg2(a1, a2);
            out[1] = avg2(a1, a2);
            out[5] = avg3(a1, a2, a3);
            out[12] = avg3(a1, a2, a3);
            out[9] = avg2(a2, a3);
            out[2] = avg2(a2, a3);
            out[13] = avg3(a2, a3, a4);
            out[6] = avg3(a2, a3, a4);
            out[10] = avg2(a3, a4);
            out[3] = avg2(a3, a4);
            out[14] = avg3(a3, a4, a5);
            out[7] = avg3(a3, a4, a5);
            out[11] = avg3(a4, a5, a6);
            out[15] = avg3(a5, a6, a7);
        }
        B_HD_PRED => {
            let e0 = ws[left + 3 * LUMA_WS_STRIDE];
            let e1 = ws[left + 2 * LUMA_WS_STRIDE];
            let e2 = ws[left + LUMA_WS_STRIDE];
            let e3 = ws[left];
            let e4 = top_left;
            let e5 = ws[top];
            let e6 = ws[top + 1];
            let e7 = ws[top + 2];
            out[12] = avg2(e0, e1);
            out[13] = avg3(e0, e1, e2);
            out[8] = avg2(e1, e2);
            out[14] = avg2(e1, e2);
            out[9] = avg3(e1, e2, e3);
            out[15] = avg3(e1, e2, e3);
            out[10] = avg2(e2, e3);
            out[4] = avg2(e2, e3);
            out[11] = avg3(e2, e3, e4);
            out[5] = avg3(e2, e3, e4);
            out[6] = avg2(e3, e4);
            out[0] = avg2(e3, e4);
            out[7] = avg3(e3, e4, e5);
            out[1] = avg3(e3, e4, e5);
            out[2] = avg3(e4, e5, e6);
            out[3] = avg3(e5, e6, e7);
        }
        B_HU_PRED => {
            let l0 = ws[left];
            let l1 = ws[left + LUMA_WS_STRIDE];
            let l2 = ws[left + 2 * LUMA_WS_STRIDE];
            let l3 = ws[left + 3 * LUMA_WS_STRIDE];
            out[0] = avg2(l0, l1);
            out[1] = avg3(l0, l1, l2);
            out[2] = avg2(l1, l2);
            out[4] = avg2(l1, l2);
            out[3] = avg3(l1, l2, l3);
            out[5] = avg3(l1, l2, l3);
            out[6] = avg2(l2, l3);
            out[8] = avg2(l2, l3);
            out[7] = avg3(l2, l3, l3);
            out[9] = avg3(l2, l3, l3);
            out[10] = l3;
            out[11] = l3;
            out[12] = l3;
            out[13] = l3;
            out[14] = l3;
            out[15] = l3;
        }
        _ => unreachable!(),
    }

    out
}

fn best_luma_mode_4x4(src: &[u8; 16], ws: &[u8], x0: usize, y0: usize) -> (u8, [u8; 16]) {
    let modes = [
        B_DC_PRED, B_TM_PRED, B_VE_PRED, B_HE_PRED, B_LD_PRED, B_RD_PRED, B_VR_PRED, B_VL_PRED,
        B_HD_PRED, B_HU_PRED,
    ];
    let mut best_mode = B_DC_PRED;
    let mut best_pred = predict_luma_4x4(best_mode, ws, x0, y0);
    let mut best_sad = sad_4x4(src, &best_pred);

    for &mode in &modes[1..] {
        let pred = predict_luma_4x4(mode, ws, x0, y0);
        let sad = sad_4x4(src, &pred);
        if sad < best_sad {
            best_mode = mode;
            best_pred = pred;
            best_sad = sad;
        }
    }

    (best_mode, best_pred)
}

fn sad_4x4(src: &[u8; 16], pred: &[u8; 16]) -> u32 {
    src.iter()
        .zip(pred.iter())
        .map(|(&s, &p)| u32::from(s.abs_diff(p)))
        .sum()
}

fn write_workspace_block_4x4(ws: &mut [u8], x0: usize, y0: usize, block: &[u8; 16]) {
    for row in 0..4 {
        let start = (y0 + row) * LUMA_WS_STRIDE + x0;
        ws[start..start + 4].copy_from_slice(&block[row * 4..row * 4 + 4]);
    }
}

fn plane_block_4x4(plane: &[u8; 64], offset: usize) -> [u8; 16] {
    let mut out = [0u8; 16];
    for row in 0..4 {
        let src = offset + row * 8;
        out[row * 4..row * 4 + 4].copy_from_slice(&plane[src..src + 4]);
    }
    out
}

fn read_block_8x8(src: &[u8], stride: usize, x: usize, y: usize) -> [u8; 64] {
    let mut out = [0u8; 64];
    for row in 0..8 {
        let start = (y + row) * stride + x;
        out[row * 8..row * 8 + 8].copy_from_slice(&src[start..start + 8]);
    }
    out
}

fn sad_8x8(src: &[u8; 64], pred: &[u8; 64]) -> u32 {
    src.iter()
        .zip(pred.iter())
        .map(|(&s, &p)| u32::from(s.abs_diff(p)))
        .sum()
}

fn predict_chroma_plane(mode: u8, recon: &[u8], stride: usize, x: usize, y: usize) -> [u8; 64] {
    let mut out = [0u8; 64];
    match mode {
        0 => out.fill(predict_dc_8x8(recon, stride, x, y)),
        1 => {
            let top = if y == 0 {
                [127u8; 8]
            } else {
                let mut row = [0u8; 8];
                row.copy_from_slice(&recon[(y - 1) * stride + x..(y - 1) * stride + x + 8]);
                row
            };
            for row in out.chunks_exact_mut(8) {
                row.copy_from_slice(&top);
            }
        }
        2 => {
            for dy in 0..8 {
                let left = if x == 0 {
                    129
                } else {
                    recon[(y + dy) * stride + x - 1]
                };
                out[dy * 8..dy * 8 + 8].fill(left);
            }
        }
        3 => {
            let top_left = if y == 0 {
                127
            } else if x == 0 {
                129
            } else {
                recon[(y - 1) * stride + x - 1]
            };
            let mut top = [127u8; 8];
            if y != 0 {
                top.copy_from_slice(&recon[(y - 1) * stride + x..(y - 1) * stride + x + 8]);
            }
            for dy in 0..8 {
                let left = if x == 0 {
                    129
                } else {
                    recon[(y + dy) * stride + x - 1]
                };
                for dx in 0..8 {
                    out[dy * 8 + dx] = (i32::from(left) + i32::from(top[dx]) - i32::from(top_left))
                        .clamp(0, 255) as u8;
                }
            }
        }
        _ => unreachable!(),
    }
    out
}

fn best_chroma_mode_8x8(
    src_u: &[u8],
    src_v: &[u8],
    recon_u: &[u8],
    recon_v: &[u8],
    stride: usize,
    x: usize,
    y: usize,
) -> (u8, [u8; 64], [u8; 64]) {
    let src_block_u = read_block_8x8(src_u, stride, x, y);
    let src_block_v = read_block_8x8(src_v, stride, x, y);
    let modes = [0u8, 1, 2, 3];

    let mut best_mode = 0u8;
    let mut best_pred_u = predict_chroma_plane(best_mode, recon_u, stride, x, y);
    let mut best_pred_v = predict_chroma_plane(best_mode, recon_v, stride, x, y);
    let mut best_sad = sad_8x8(&src_block_u, &best_pred_u) + sad_8x8(&src_block_v, &best_pred_v);

    for &mode in &modes[1..] {
        let pred_u = predict_chroma_plane(mode, recon_u, stride, x, y);
        let pred_v = predict_chroma_plane(mode, recon_v, stride, x, y);
        let sad = sad_8x8(&src_block_u, &pred_u) + sad_8x8(&src_block_v, &pred_v);
        if sad < best_sad {
            best_mode = mode;
            best_pred_u = pred_u;
            best_pred_v = pred_v;
            best_sad = sad;
        }
    }

    (best_mode, best_pred_u, best_pred_v)
}

fn predict_dc_8x8(recon: &[u8], stride: usize, x: usize, y: usize) -> u8 {
    let above = y != 0;
    let left = x != 0;
    let mut sum = 0u32;
    let mut shf = 2u32;

    if left {
        for dy in 0..8 {
            sum += u32::from(recon[(y + dy) * stride + x - 1]);
        }
        shf += 1;
    }

    if above {
        for dx in 0..8 {
            sum += u32::from(recon[(y - 1) * stride + x + dx]);
        }
        shf += 1;
    }

    if !above && !left {
        128
    } else {
        ((sum + (1 << (shf - 1))) >> shf) as u8
    }
}

fn read_block_4x4(src: &[u8], stride: usize, x: usize, y: usize) -> [u8; 16] {
    let mut out = [0u8; 16];
    for row in 0..4 {
        let start = (y + row) * stride + x;
        out[row * 4..row * 4 + 4].copy_from_slice(&src[start..start + 4]);
    }
    out
}

fn write_block_4x4(dst: &mut [u8], stride: usize, x: usize, y: usize, block: &[u8; 16]) {
    for row in 0..4 {
        let start = (y + row) * stride + x;
        dst[start..start + 4].copy_from_slice(&block[row * 4..row * 4 + 4]);
    }
}

fn residual_from_block_prediction(src: &[u8; 16], pred: &[u8; 16]) -> [i16; 16] {
    let mut residual = [0i16; 16];
    for (dst, (&pixel, &pred)) in residual.iter_mut().zip(src.iter().zip(pred.iter())) {
        *dst = i16::from(pixel) - i16::from(pred);
    }
    residual
}

fn residual_from_prediction(src: &[u8; 16], pred: u8) -> [i16; 16] {
    let mut residual = [0i16; 16];
    for (dst, &pixel) in residual.iter_mut().zip(src.iter()) {
        *dst = i16::from(pixel) - i16::from(pred);
    }
    residual
}

fn reconstruct_from_block_coeffs(coeffs: &[i16; 16], q_idx: usize, pred: &[u8; 16]) -> [u8; 16] {
    let mut dequant = dequantize_block(coeffs, q_idx);
    transform::idct4x4(&mut dequant);

    let mut out = [0u8; 16];
    for (dst, (&prediction, residue)) in out.iter_mut().zip(pred.iter().zip(dequant.iter())) {
        *dst = (i32::from(prediction) + *residue).clamp(0, 255) as u8;
    }
    out
}

fn reconstruct_from_coeffs(coeffs: &[i16; 16], q_idx: usize, pred: u8) -> [u8; 16] {
    let mut dequant = dequantize_block(coeffs, q_idx);
    transform::idct4x4(&mut dequant);

    let mut out = [0u8; 16];
    for (dst, residue) in out.iter_mut().zip(dequant.iter()) {
        *dst = (i32::from(pred) + *residue).clamp(0, 255) as u8;
    }
    out
}

fn write_bpred_mode(enc: &mut BoolEncoder, top: u8, left: u8, mode: u8) {
    let probs = &KEYFRAME_BPRED_MODE_PROBS[top as usize][left as usize];
    match mode {
        B_DC_PRED => {
            enc.write_bool(false, probs[0]);
        }
        B_TM_PRED => {
            enc.write_bool(true, probs[0]);
            enc.write_bool(false, probs[1]);
        }
        B_VE_PRED => {
            enc.write_bool(true, probs[0]);
            enc.write_bool(true, probs[1]);
            enc.write_bool(false, probs[2]);
        }
        B_HE_PRED => {
            enc.write_bool(true, probs[0]);
            enc.write_bool(true, probs[1]);
            enc.write_bool(true, probs[2]);
            enc.write_bool(false, probs[3]);
            enc.write_bool(false, probs[4]);
        }
        B_RD_PRED => {
            enc.write_bool(true, probs[0]);
            enc.write_bool(true, probs[1]);
            enc.write_bool(true, probs[2]);
            enc.write_bool(false, probs[3]);
            enc.write_bool(true, probs[4]);
            enc.write_bool(false, probs[5]);
        }
        B_VR_PRED => {
            enc.write_bool(true, probs[0]);
            enc.write_bool(true, probs[1]);
            enc.write_bool(true, probs[2]);
            enc.write_bool(false, probs[3]);
            enc.write_bool(true, probs[4]);
            enc.write_bool(true, probs[5]);
        }
        B_LD_PRED => {
            enc.write_bool(true, probs[0]);
            enc.write_bool(true, probs[1]);
            enc.write_bool(true, probs[2]);
            enc.write_bool(true, probs[3]);
            enc.write_bool(false, probs[6]);
        }
        B_VL_PRED => {
            enc.write_bool(true, probs[0]);
            enc.write_bool(true, probs[1]);
            enc.write_bool(true, probs[2]);
            enc.write_bool(true, probs[3]);
            enc.write_bool(true, probs[6]);
            enc.write_bool(false, probs[7]);
        }
        B_HD_PRED => {
            enc.write_bool(true, probs[0]);
            enc.write_bool(true, probs[1]);
            enc.write_bool(true, probs[2]);
            enc.write_bool(true, probs[3]);
            enc.write_bool(true, probs[6]);
            enc.write_bool(true, probs[7]);
            enc.write_bool(false, probs[8]);
        }
        B_HU_PRED => {
            enc.write_bool(true, probs[0]);
            enc.write_bool(true, probs[1]);
            enc.write_bool(true, probs[2]);
            enc.write_bool(true, probs[3]);
            enc.write_bool(true, probs[6]);
            enc.write_bool(true, probs[7]);
            enc.write_bool(true, probs[8]);
        }
        _ => unreachable!(),
    }
}

fn write_uv_mode(enc: &mut BoolEncoder, mode: u8) {
    match mode {
        0 => enc.write_bool(false, KEYFRAME_UV_MODE_PROBS[0]),
        1 => {
            enc.write_bool(true, KEYFRAME_UV_MODE_PROBS[0]);
            enc.write_bool(false, KEYFRAME_UV_MODE_PROBS[1]);
        }
        2 => {
            enc.write_bool(true, KEYFRAME_UV_MODE_PROBS[0]);
            enc.write_bool(true, KEYFRAME_UV_MODE_PROBS[1]);
            enc.write_bool(false, KEYFRAME_UV_MODE_PROBS[2]);
        }
        3 => {
            enc.write_bool(true, KEYFRAME_UV_MODE_PROBS[0]);
            enc.write_bool(true, KEYFRAME_UV_MODE_PROBS[1]);
            enc.write_bool(true, KEYFRAME_UV_MODE_PROBS[2]);
        }
        _ => unreachable!(),
    }
}

fn encode_first_partition(
    q_idx: usize,
    macroblocks: &[MacroBlockData],
    mb_width: usize,
) -> Vec<u8> {
    let mut enc = BoolEncoder::new();

    enc.write_literal(1, 0);
    enc.write_literal(1, 0);
    enc.write_flag(false);
    enc.write_flag(false);
    enc.write_literal(6, 0);
    enc.write_literal(3, 0);
    enc.write_flag(false);
    enc.write_literal(2, 0);

    enc.write_literal(7, q_idx as u32);
    for _ in 0..5 {
        enc.write_flag(false);
    }

    enc.write_literal(1, 0);

    for tables in &COEFF_UPDATE_PROBS {
        for bands in tables {
            for ctxs in bands {
                for &prob in ctxs {
                    enc.write_bool(false, prob);
                }
            }
        }
    }

    enc.write_literal(1, 0);

    let mut top_modes = vec![[B_DC_PRED; 4]; mb_width];
    for row in macroblocks.chunks(mb_width) {
        let mut left_modes = [B_DC_PRED; 4];
        for (mb_x, mb) in row.iter().enumerate() {
            enc.write_bool(false, KEYFRAME_YMODE_B_PRED_PROB);
            for sub_y in 0..4 {
                let mut left = left_modes[sub_y];
                for sub_x in 0..4 {
                    let block_index = sub_x + sub_y * 4;
                    let top = top_modes[mb_x][sub_x];
                    let mode = mb.bpred_modes[block_index];
                    write_bpred_mode(&mut enc, top, left, mode);
                    top_modes[mb_x][sub_x] = mode;
                    left = mode;
                }
                left_modes[sub_y] = left;
            }
            write_uv_mode(&mut enc, mb.chroma_mode);
        }
    }

    enc.finish()
}

fn encode_coeff_partition(macroblocks: &[MacroBlockData], mb_width: usize) -> Vec<u8> {
    let mut enc = BoolEncoder::new();

    let mut top_y = vec![[0u8; 4]; mb_width];
    let mut top_u = vec![[0u8; 2]; mb_width];
    let mut top_v = vec![[0u8; 2]; mb_width];

    for row in macroblocks.chunks(mb_width) {
        let mut left_y = [0u8; 4];
        let mut left_u = [0u8; 2];
        let mut left_v = [0u8; 2];

        for (mb_x, mb) in row.iter().enumerate() {
            for sub_y in 0..4 {
                let mut left = left_y[sub_y];
                for sub_x in 0..4 {
                    let block_index = sub_x + sub_y * 4;
                    let ctx = (usize::from(top_y[mb_x][sub_x]) + usize::from(left)).min(2);
                    encode_block(&mut enc, &mb.y_blocks[block_index], 3, 0, ctx);
                    let coded = u8::from(mb.y_nonzero[block_index]);
                    top_y[mb_x][sub_x] = coded;
                    left = coded;
                }
                left_y[sub_y] = left;
            }

            for sub_y in 0..2 {
                let mut left = left_u[sub_y];
                for sub_x in 0..2 {
                    let block_index = sub_x + sub_y * 2;
                    let ctx = (usize::from(top_u[mb_x][sub_x]) + usize::from(left)).min(2);
                    encode_block(&mut enc, &mb.u_blocks[block_index], 2, 0, ctx);
                    let coded = u8::from(mb.u_nonzero[block_index]);
                    top_u[mb_x][sub_x] = coded;
                    left = coded;
                }
                left_u[sub_y] = left;
            }

            for sub_y in 0..2 {
                let mut left = left_v[sub_y];
                for sub_x in 0..2 {
                    let block_index = sub_x + sub_y * 2;
                    let ctx = (usize::from(top_v[mb_x][sub_x]) + usize::from(left)).min(2);
                    encode_block(&mut enc, &mb.v_blocks[block_index], 2, 0, ctx);
                    let coded = u8::from(mb.v_nonzero[block_index]);
                    top_v[mb_x][sub_x] = coded;
                    left = coded;
                }
                left_v[sub_y] = left;
            }
        }
    }

    enc.finish()
}

fn encode_block(
    enc: &mut BoolEncoder,
    coeffs: &[i16; 16],
    block_type: usize,
    first: usize,
    ctx: usize,
) {
    let probs = &COEFF_PROBS[block_type];
    let last_nz = (first..16).rev().find(|&i| coeffs[ZIGZAG[i] as usize] != 0);

    let Some(last_nz) = last_nz else {
        let band = COEFF_BANDS[first] as usize;
        let p = &probs[band][ctx];
        encode_token(enc, DCT_EOB, p, false);
        return;
    };

    let mut complexity = ctx;
    let mut skip = false;

    for i in first..=last_nz {
        let band = COEFF_BANDS[i] as usize;
        let p = &probs[band][complexity];
        let coeff = coeffs[ZIGZAG[i] as usize];

        if coeff == 0 {
            encode_token(enc, DCT_0, p, skip);
            complexity = 0;
            skip = true;
            continue;
        }

        let abs_v = coeff.unsigned_abs() as u32;
        let (token, extra_bits, extra_val) = value_to_token(abs_v);
        encode_token(enc, token, p, skip);
        encode_extra_bits(enc, token, extra_bits, extra_val);
        enc.write_flag(coeff < 0);

        complexity = if abs_v == 1 { 1 } else { 2 };
        skip = false;
    }

    if last_nz < 15 {
        let band = COEFF_BANDS[last_nz + 1] as usize;
        let p = &probs[band][complexity];
        encode_token(enc, DCT_EOB, p, false);
    }
}

fn encode_token(enc: &mut BoolEncoder, token: i8, probs: &[u8; 11], skip: bool) {
    if !skip {
        if token == DCT_EOB {
            enc.write_bool(false, probs[0]);
            return;
        }
        enc.write_bool(true, probs[0]);
    }

    if token == DCT_0 {
        enc.write_bool(false, probs[1]);
        return;
    }
    enc.write_bool(true, probs[1]);

    if token == DCT_1 {
        enc.write_bool(false, probs[2]);
        return;
    }
    enc.write_bool(true, probs[2]);

    if token == DCT_2 {
        enc.write_bool(false, probs[3]);
        return;
    }
    enc.write_bool(true, probs[3]);

    if token == DCT_3 {
        enc.write_bool(false, probs[4]);
        return;
    }
    enc.write_bool(true, probs[4]);

    if token == DCT_4 {
        enc.write_bool(false, probs[5]);
        return;
    }
    enc.write_bool(true, probs[5]);

    if token == DCT_CAT1 {
        enc.write_bool(false, probs[6]);
        return;
    }
    enc.write_bool(true, probs[6]);

    if token == DCT_CAT2 {
        enc.write_bool(false, probs[7]);
        return;
    }
    enc.write_bool(true, probs[7]);

    if token == DCT_CAT3 {
        enc.write_bool(false, probs[8]);
        return;
    }
    enc.write_bool(true, probs[8]);

    if token == DCT_CAT4 {
        enc.write_bool(false, probs[9]);
        return;
    }
    enc.write_bool(true, probs[9]);

    if token == DCT_CAT5 {
        enc.write_bool(false, probs[10]);
        return;
    }
    enc.write_bool(true, probs[10]);
}

fn value_to_token(abs_v: u32) -> (i8, usize, u32) {
    match abs_v {
        1 => (DCT_1, 0, 0),
        2 => (DCT_2, 0, 0),
        3 => (DCT_3, 0, 0),
        4 => (DCT_4, 0, 0),
        5..=6 => (DCT_CAT1, 1, abs_v - u32::from(DCT_CAT_BASE[0])),
        7..=10 => (DCT_CAT2, 2, abs_v - u32::from(DCT_CAT_BASE[1])),
        11..=18 => (DCT_CAT3, 3, abs_v - u32::from(DCT_CAT_BASE[2])),
        19..=34 => (DCT_CAT4, 4, abs_v - u32::from(DCT_CAT_BASE[3])),
        35..=66 => (DCT_CAT5, 5, abs_v - u32::from(DCT_CAT_BASE[4])),
        _ => (DCT_CAT6, 11, abs_v - u32::from(DCT_CAT_BASE[5])),
    }
}

fn encode_extra_bits(enc: &mut BoolEncoder, token: i8, extra_bits: usize, extra_val: u32) {
    let cat_index = match token {
        DCT_CAT1 => 0,
        DCT_CAT2 => 1,
        DCT_CAT3 => 2,
        DCT_CAT4 => 3,
        DCT_CAT5 => 4,
        DCT_CAT6 => 5,
        _ => return,
    };

    let cat_probs = &PROB_DCT_CAT[cat_index];
    for bit_index in 0..extra_bits {
        let shift = extra_bits - 1 - bit_index;
        let bit = ((extra_val >> shift) & 1) != 0;
        enc.write_bool(bit, cat_probs[bit_index]);
    }
}

fn build_vp8_frame(
    width: u32,
    height: u32,
    first_partition: &[u8],
    coeff_partition: &[u8],
) -> Result<Vec<u8>, WebPEncodeError> {
    let first_part_size = u32::try_from(first_partition.len())
        .map_err(|_| WebPEncodeError("First partition exceeds VP8 limits".to_owned()))?;
    if first_part_size >= (1 << 19) {
        return Err(WebPEncodeError(
            "First partition exceeds 19-bit VP8 frame tag capacity".to_owned(),
        ));
    }

    let mut out = Vec::with_capacity(10 + 4 + first_partition.len() + coeff_partition.len());
    let frame_tag = (first_part_size << 5) | 0x10;
    out.push((frame_tag & 0xFF) as u8);
    out.push(((frame_tag >> 8) & 0xFF) as u8);
    out.push(((frame_tag >> 16) & 0xFF) as u8);

    out.extend_from_slice(&[0x9D, 0x01, 0x2A]);
    out.extend_from_slice(&(width as u16).to_le_bytes());
    out.extend_from_slice(&(height as u16).to_le_bytes());
    out.extend_from_slice(first_partition);
    out.extend_from_slice(coeff_partition);
    Ok(out)
}

fn wrap_riff_webp(vp8_data: &[u8]) -> Vec<u8> {
    let vp8_size = vp8_data.len() as u32;
    let padding = vp8_size & 1;
    let riff_size = 4 + 8 + vp8_size + padding;

    let mut out = Vec::with_capacity((riff_size + 8) as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(b"VP8 ");
    out.extend_from_slice(&vp8_size.to_le_bytes());
    out.extend_from_slice(vp8_data);
    if padding != 0 {
        out.push(0);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::{Rgb, RgbImage};

    use crate::webp_decode::WebPDecoder;

    use super::encode_lossy_webp;

    #[test]
    fn encodes_decodable_lossy_webp() {
        let img = RgbImage::from_fn(16, 16, |x, y| {
            Rgb([
                ((x * 13 + y * 7) & 0xFF) as u8,
                ((x * 3 + y * 19) & 0xFF) as u8,
                ((x * 11 + y * 5) & 0xFF) as u8,
            ])
        });

        let encoded = encode_lossy_webp(&img, 85).expect("encode should succeed");
        let mut decoder = WebPDecoder::new(Cursor::new(encoded)).expect("decode init should work");
        assert_eq!(decoder.dimensions(), (16, 16));

        let mut rgb = vec![0u8; decoder.output_buffer_size().expect("known size")];
        decoder
            .read_image(&mut rgb)
            .expect("encoded webp should decode");
        assert_eq!(rgb.len(), 16 * 16 * 3);
    }
}
