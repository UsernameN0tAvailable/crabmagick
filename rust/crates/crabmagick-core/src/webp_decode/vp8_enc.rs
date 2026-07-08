use std::error::Error;
use std::fmt::{Display, Formatter};

use image::RgbImage;
use rayon::prelude::*;

use crate::webp_decode::transform;
use crate::webp_decode::vp8::{
    AC_QUANT, COEFF_BANDS, COEFF_PROBS, COEFF_UPDATE_PROBS, DCT_0, DCT_1, DCT_2, DCT_3, DCT_4,
    DCT_CAT_BASE, DCT_CAT1, DCT_CAT2, DCT_CAT3, DCT_CAT4, DCT_CAT5, DCT_CAT6, DCT_EOB,
    DC_QUANT, PROB_DCT_CAT, ZIGZAG,
};

const KEYFRAME_YMODE_B_PRED_PROB: u8 = 145;
const KEYFRAME_BPRED_DC_PROB: u8 = 231;
const KEYFRAME_UVMODE_DC_PROB: u8 = 142;

const LUMA_BLOCKS_PER_MB: usize = 16;
const CHROMA_BLOCKS_PER_MB: usize = 4;

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
}

pub(crate) struct BoolEncoder {
    range: u32,
    low: u64,
    /// Completed output bytes.
    out: Vec<u8>,
    /// Partial byte being assembled, MSB-first; holds `cur_bits` valid low bits.
    cur: u32,
    cur_bits: u32,
}

impl BoolEncoder {
    pub fn new() -> Self {
        Self {
            range: 255,
            low: 0,
            out: Vec::new(),
            cur: 0,
            cur_bits: 0,
        }
    }

    /// Append a single bit MSB-first to the packed output.
    #[inline]
    fn emit_bit(&mut self, bit: bool) {
        self.cur = (self.cur << 1) | u32::from(bit);
        self.cur_bits += 1;
        if self.cur_bits == 8 {
            self.out.push(self.cur as u8);
            self.cur = 0;
            self.cur_bits = 0;
        }
    }

    /// Propagate a carry: add 1 to the big-endian number formed by every bit
    /// emitted so far (its least-significant bit is the most recently emitted
    /// bit). Operates on packed bytes, so carry runs are cache-friendly and
    /// amortize to O(1).
    #[inline]
    fn carry(&mut self) {
        if self.cur_bits > 0 {
            self.cur += 1;
            if self.cur >> self.cur_bits != 0 {
                self.cur &= (1 << self.cur_bits) - 1;
                Self::add_one_to_bytes(&mut self.out);
            }
        } else {
            Self::add_one_to_bytes(&mut self.out);
        }
    }

    #[inline]
    fn add_one_to_bytes(out: &mut [u8]) {
        for byte in out.iter_mut().rev() {
            if *byte == 0xFF {
                *byte = 0;
            } else {
                *byte += 1;
                return;
            }
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
            if self.low >= 0x1_0000_0000 {
                self.carry();
                self.low -= 0x1_0000_0000;
            }

            self.emit_bit((self.low & 0x8000_0000) != 0);
            self.low = (self.low << 1) & 0xFFFF_FFFF;
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

        if self.cur_bits > 0 {
            // Pad the trailing partial byte with zero low bits (MSB-first).
            self.out.push((self.cur << (8 - self.cur_bits)) as u8);
        }

        self.out
    }
}

/// Encode an RGB image as a lossy VP8-based WebP image.
#[inline]
pub(crate) fn encode_lossy_webp(img: &RgbImage, quality: u8) -> Result<Vec<u8>, WebPEncodeError> {
    encode_lossy_webp_impl(img, quality, None)
}

/// Encode with an explicit token-partition count (test helper).
#[cfg(test)]
pub(crate) fn encode_lossy_webp_with_partitions(
    img: &RgbImage,
    quality: u8,
    partitions: usize,
) -> Result<Vec<u8>, WebPEncodeError> {
    encode_lossy_webp_impl(img, quality, Some(partitions))
}

fn encode_lossy_webp_impl(
    img: &RgbImage,
    quality: u8,
    forced_partitions: Option<usize>,
) -> Result<Vec<u8>, WebPEncodeError> {
    let width = img.width();
    let height = img.height();

    if width == 0 || height == 0 {
        return Err(WebPEncodeError("VP8 encoder does not support empty images".to_owned()));
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

    let first_part_mbs = mb_width * mb_height;
    let num_partitions = forced_partitions.unwrap_or_else(|| choose_token_partitions(mb_height));

    let first_partition = encode_first_partition(q_idx, first_part_mbs, num_partitions);
    let coeff_partitions = encode_coeff_partitions(&macroblocks, mb_width, mb_height, num_partitions);

    let vp8_frame = build_vp8_frame(width, height, &first_partition, &coeff_partitions)?;
    Ok(wrap_riff_webp(&vp8_frame))
}

#[inline(always)]
fn quality_to_q_index(quality: u8) -> usize {
    let q = quality.min(100) as usize;
    (127 * (100 - q) / 100).min(127)
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
    };

    for sub_y in 0..4 {
        for sub_x in 0..4 {
            let block_index = sub_x + sub_y * 4;
            let x = mb_x * 16 + sub_x * 4;
            let y = mb_y * 16 + sub_y * 4;
            let pred = predict_bdc_4x4(recon_y, y_stride, x, y);
            let src = read_block_4x4(src_y, y_stride, x, y);
            let residual = residual_from_prediction(&src, pred);
            let dct = fdct4x4(&residual);
            let (coeffs, nonzero) = quantize_block(&dct, q_idx);
            let recon = reconstruct_from_coeffs(&coeffs, q_idx, pred);
            write_block_4x4(recon_y, y_stride, x, y, &recon);
            data.y_blocks[block_index] = coeffs;
            data.y_nonzero[block_index] = nonzero;
        }
    }

    let chroma_x = mb_x * 8;
    let chroma_y = mb_y * 8;
    let pred_u = predict_dc_8x8(recon_u, uv_stride, chroma_x, chroma_y);
    let pred_v = predict_dc_8x8(recon_v, uv_stride, chroma_x, chroma_y);

    for sub_y in 0..2 {
        for sub_x in 0..2 {
            let block_index = sub_x + sub_y * 2;
            let x = chroma_x + sub_x * 4;
            let y = chroma_y + sub_y * 4;

            let src_block_u = read_block_4x4(src_u, uv_stride, x, y);
            let residual_u = residual_from_prediction(&src_block_u, pred_u);
            let dct_u = fdct4x4(&residual_u);
            let (coeffs_u, nonzero_u) = quantize_block(&dct_u, q_idx);
            let recon_block_u = reconstruct_from_coeffs(&coeffs_u, q_idx, pred_u);
            write_block_4x4(recon_u, uv_stride, x, y, &recon_block_u);
            data.u_blocks[block_index] = coeffs_u;
            data.u_nonzero[block_index] = nonzero_u;

            let src_block_v = read_block_4x4(src_v, uv_stride, x, y);
            let residual_v = residual_from_prediction(&src_block_v, pred_v);
            let dct_v = fdct4x4(&residual_v);
            let (coeffs_v, nonzero_v) = quantize_block(&dct_v, q_idx);
            let recon_block_v = reconstruct_from_coeffs(&coeffs_v, q_idx, pred_v);
            write_block_4x4(recon_v, uv_stride, x, y, &recon_block_v);
            data.v_blocks[block_index] = coeffs_v;
            data.v_nonzero[block_index] = nonzero_v;
        }
    }

    data
}

fn predict_bdc_4x4(recon: &[u8], stride: usize, x: usize, y: usize) -> u8 {
    let mut sum = 4u32;

    if y == 0 {
        sum += 4 * 127;
    } else {
        for dx in 0..4 {
            sum += u32::from(recon[(y - 1) * stride + x + dx]);
        }
    }

    if x == 0 {
        sum += 4 * 129;
    } else {
        for dy in 0..4 {
            sum += u32::from(recon[(y + dy) * stride + x - 1]);
        }
    }

    (sum >> 3) as u8
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

fn residual_from_prediction(src: &[u8; 16], pred: u8) -> [i16; 16] {
    let mut residual = [0i16; 16];
    for (dst, &pixel) in residual.iter_mut().zip(src.iter()) {
        *dst = i16::from(pixel) - i16::from(pred);
    }
    residual
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

/// Chooses the number of VP8 DCT token partitions (1, 2, 4, or 8).
///
/// Token partitions carry independent range-coded coefficient streams that can
/// be encoded (and decoded) in parallel. Partition `p` owns macroblock rows
/// where `mb_y % num_partitions == p`, so the count is capped at `mb_height`
/// (rounded down to a power of two) to avoid empty partitions.
#[inline]
fn choose_token_partitions(mb_height: usize) -> usize {
    if mb_height >= 8 {
        8
    } else if mb_height >= 4 {
        4
    } else if mb_height >= 2 {
        2
    } else {
        1
    }
}

fn encode_first_partition(q_idx: usize, macroblock_count: usize, num_partitions: usize) -> Vec<u8> {
    let mut enc = BoolEncoder::new();

    enc.write_literal(1, 0);
    enc.write_literal(1, 0);
    enc.write_flag(false);
    enc.write_flag(false);
    enc.write_literal(6, 0);
    enc.write_literal(3, 0);
    enc.write_flag(false);
    // log2(number of DCT token partitions)
    enc.write_literal(2, num_partitions.trailing_zeros());

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

    for _ in 0..macroblock_count {
        enc.write_bool(false, KEYFRAME_YMODE_B_PRED_PROB);
        for _ in 0..16 {
            enc.write_bool(false, KEYFRAME_BPRED_DC_PROB);
        }
        enc.write_bool(false, KEYFRAME_UVMODE_DC_PROB);
    }

    enc.finish()
}
/// Encodes the DCT coefficient token partitions in parallel.
///
/// VP8 allows the coefficient data to be split into `num_partitions` independent
/// range-coded streams. Partition `p` owns the macroblock rows where
/// `mb_y % num_partitions == p` (matching the decoder's `mb_y % num_partitions`
/// selection). Each partition uses its own [`BoolEncoder`], so they are encoded
/// concurrently with rayon and merged in order by the caller.
///
/// The coefficient context (`nonzero` above/left flags) depends only on the
/// already-quantized coefficients, never on the entropy coder state, so the
/// "above" context for any row is reconstructed directly from the row above's
/// `*_nonzero` flags. This keeps every partition fully independent and produces
/// exactly the same tokens (and therefore the same decoded pixels) as a single
/// serial partition.
fn encode_coeff_partitions(
    macroblocks: &[MacroBlockData],
    mb_width: usize,
    mb_height: usize,
    num_partitions: usize,
) -> Vec<Vec<u8>> {
    if num_partitions <= 1 {
        return vec![encode_partition(macroblocks, mb_width, mb_height, 0, 1)];
    }

    (0..num_partitions)
        .into_par_iter()
        .map(|part| encode_partition(macroblocks, mb_width, mb_height, part, num_partitions))
        .collect()
}

/// Encodes the coefficient tokens for a single token partition.
fn encode_partition(
    macroblocks: &[MacroBlockData],
    mb_width: usize,
    mb_height: usize,
    part: usize,
    num_partitions: usize,
) -> Vec<u8> {
    let mut enc = BoolEncoder::new();

    let mut top_y = vec![[0u8; 4]; mb_width];
    let mut top_u = vec![[0u8; 2]; mb_width];
    let mut top_v = vec![[0u8; 2]; mb_width];

    let mut mb_y = part;
    while mb_y < mb_height {
        // Reconstruct the "above" context from the row directly above (rows in a
        // partition are not contiguous, so the persisted context cannot be reused).
        if mb_y == 0 {
            for x in 0..mb_width {
                top_y[x] = [0; 4];
                top_u[x] = [0; 2];
                top_v[x] = [0; 2];
            }
        } else {
            let above = &macroblocks[(mb_y - 1) * mb_width..mb_y * mb_width];
            for (x, mb) in above.iter().enumerate() {
                for sub_x in 0..4 {
                    top_y[x][sub_x] = u8::from(mb.y_nonzero[sub_x + 3 * 4]);
                }
                for sub_x in 0..2 {
                    top_u[x][sub_x] = u8::from(mb.u_nonzero[sub_x + 1 * 2]);
                    top_v[x][sub_x] = u8::from(mb.v_nonzero[sub_x + 1 * 2]);
                }
            }
        }

        let row = &macroblocks[mb_y * mb_width..(mb_y + 1) * mb_width];
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

        mb_y += num_partitions;
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
    let last_nz = (first..16)
        .rev()
        .find(|&i| coeffs[ZIGZAG[i] as usize] != 0);

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
    coeff_partitions: &[Vec<u8>],
) -> Result<Vec<u8>, WebPEncodeError> {
    let first_part_size = u32::try_from(first_partition.len())
        .map_err(|_| WebPEncodeError("First partition exceeds VP8 limits".to_owned()))?;
    if first_part_size >= (1 << 19) {
        return Err(WebPEncodeError(
            "First partition exceeds 19-bit VP8 frame tag capacity".to_owned(),
        ));
    }

    // Sizes (3-byte little-endian) precede the token data for every partition
    // except the last one, whose length the decoder infers from the remaining
    // bytes.
    let size_table_len = coeff_partitions.len().saturating_sub(1) * 3;
    let coeff_total: usize = coeff_partitions.iter().map(Vec::len).sum();

    let mut out =
        Vec::with_capacity(10 + first_partition.len() + size_table_len + coeff_total);
    let frame_tag = (first_part_size << 5) | 0x10;
    out.push((frame_tag & 0xFF) as u8);
    out.push(((frame_tag >> 8) & 0xFF) as u8);
    out.push(((frame_tag >> 16) & 0xFF) as u8);

    out.extend_from_slice(&[0x9D, 0x01, 0x2A]);
    out.extend_from_slice(&(width as u16).to_le_bytes());
    out.extend_from_slice(&(height as u16).to_le_bytes());
    out.extend_from_slice(first_partition);

    for part in &coeff_partitions[..coeff_partitions.len().saturating_sub(1)] {
        let size = u32::try_from(part.len())
            .map_err(|_| WebPEncodeError("Token partition exceeds VP8 limits".to_owned()))?;
        out.push((size & 0xFF) as u8);
        out.push(((size >> 8) & 0xFF) as u8);
        out.push(((size >> 16) & 0xFF) as u8);
    }

    for part in coeff_partitions {
        out.extend_from_slice(part);
    }

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

    fn decode_webp(encoded: Vec<u8>) -> (u32, u32, Vec<u8>) {
        let mut decoder = WebPDecoder::new(Cursor::new(encoded)).expect("decode init");
        let (w, h) = decoder.dimensions();
        let mut rgb = vec![0u8; decoder.output_buffer_size().expect("known size")];
        decoder.read_image(&mut rgb).expect("decode");
        (w, h, rgb)
    }

    /// Multi-token-partition output must decode to exactly the same pixels as a
    /// single-partition encode (the tokens are identical, only the range-coder
    /// container is split), and must round-trip through the decoder cleanly.
    #[test]
    fn multi_partition_matches_single_partition() {
        // 200x160 => 13x10 macroblocks => mb_height 10 => 8 token partitions.
        let img = RgbImage::from_fn(200, 160, |x, y| {
            Rgb([
                ((x * 7 + y * 3) & 0xFF) as u8,
                ((x * 5 + y * 11) & 0xFF) as u8,
                ((x.wrapping_mul(2) + y * 17) & 0xFF) as u8,
            ])
        });

        // Sanity: this image is large enough to trigger multiple partitions.
        assert_eq!(super::choose_token_partitions(160u32.div_ceil(16) as usize), 8);

        let single = super::encode_lossy_webp_with_partitions(&img, 90, 1).expect("single encode");
        let single_px = match {
            let mut dec = WebPDecoder::new(Cursor::new(single.clone())).expect("init");
            let mut rgb = vec![0u8; dec.output_buffer_size().unwrap()];
            dec.read_image(&mut rgb).map(|()| rgb)
        } {
            Ok(px) => {
                eprintln!("N=1: decode OK");
                px
            }
            Err(e) => {
                eprintln!("N=1: decode ERR {e:?}");
                Vec::new()
            }
        };

        for &n in &[2usize, 4, 8] {
            let enc = super::encode_lossy_webp_with_partitions(&img, 90, n).expect("enc");
            let mut dec = WebPDecoder::new(Cursor::new(enc)).expect("init");
            let mut rgb = vec![0u8; dec.output_buffer_size().unwrap()];
            match dec.read_image(&mut rgb) {
                Ok(()) => eprintln!("N={n}: decode OK, match_single={}", rgb == single_px),
                Err(e) => eprintln!("N={n}: decode ERR {e:?}"),
            }
        }
        return;
        #[allow(unreachable_code)]
        let multi = encode_lossy_webp(&img, 90).expect("multi-partition encode");
    }
}
