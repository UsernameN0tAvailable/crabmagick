// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Transform pipeline: DCT dispatch and block-level transform + quantize.

use super::ac_group::{num_nonzero_8x8_except_dc, num_nonzero_except_llf};
use super::ac_strategy::{
    AcStrategyMap, RAW_STRATEGY_AFV0, RAW_STRATEGY_AFV1, RAW_STRATEGY_AFV2, RAW_STRATEGY_AFV3,
    RAW_STRATEGY_DCT2X2, RAW_STRATEGY_DCT4X4, RAW_STRATEGY_DCT4X8, RAW_STRATEGY_DCT8X4,
    RAW_STRATEGY_DCT8X16, RAW_STRATEGY_DCT16X8, RAW_STRATEGY_DCT16X16, RAW_STRATEGY_DCT16X32,
    RAW_STRATEGY_DCT32X16, RAW_STRATEGY_DCT32X32, RAW_STRATEGY_DCT32X64, RAW_STRATEGY_DCT64X32,
    RAW_STRATEGY_DCT64X64, RAW_STRATEGY_IDENTITY,
};
use super::afv::{afv_transform_from_pixels, dc_from_afv};
use super::block_extract::extract_block_8x8;
use super::chroma_from_luma::{CflMap, ytob_ratio, ytox_ratio};
use super::common::*;
use super::dct::{
    dc_from_dct_4x4_full, dc_from_dct_4x8_full, dc_from_dct_8x4_full, dc_from_dct_8x16,
    dc_from_dct_16x8, dc_from_dct_16x16, dc_from_dct_16x32, dc_from_dct_32x16, dc_from_dct_32x32,
    dc_from_dct_32x64, dc_from_dct_64x32, dc_from_dct_64x64, dct_4x4_full, dct_4x8_full,
    dct_8x4_full, dct_8x8, dct_8x16, dct_16x8, dct_16x16, dct_16x32, dct_32x16, dct_32x32,
    dct_32x64, dct_64x32, dct_64x64, dct2x2_transform, identity_transform,
};
use super::encoder::VarDctEncoder;
use super::frame::DistanceParams;
use super::quant::INV_DC_QUANT;
use super::quantize::adjust_quant_bias;
use crate::jxl_encode::debug_rect;
use std::cell::RefCell;

struct TransformScratch {
    local_x: Vec<f32>,
    local_y: Vec<f32>,
    local_b: Vec<f32>,
    error_scratch: Vec<f32>,
    quant_flat_scratch: Vec<i32>,
    nz_full_block_scratch: Vec<i32>,
    nz_flat_scratch: Vec<u8>,
}

impl TransformScratch {
    fn new() -> Self {
        Self {
            local_x: Vec::new(),
            local_y: Vec::new(),
            local_b: Vec::new(),
            error_scratch: Vec::new(),
            quant_flat_scratch: Vec::new(),
            nz_full_block_scratch: Vec::new(),
            nz_flat_scratch: Vec::new(),
        }
    }

    fn ensure_local_channels(&mut self, len: usize) {
        if self.local_x.len() < len {
            self.local_x.resize(len, 0.0);
        }
        if self.local_y.len() < len {
            self.local_y.resize(len, 0.0);
        }
        if self.local_b.len() < len {
            self.local_b.resize(len, 0.0);
        }
    }

    fn ensure_error_scratch(&mut self, len: usize) {
        if self.error_scratch.len() < len {
            self.error_scratch.resize(len, 0.0);
        }
    }

    fn ensure_quant_scratch(&mut self, len: usize) {
        if self.quant_flat_scratch.len() < len {
            self.quant_flat_scratch.resize(len, 0);
        }
    }

    fn ensure_nz_full_scratch(&mut self, len: usize) {
        if self.nz_full_block_scratch.len() < len {
            self.nz_full_block_scratch.resize(len, 0);
        }
    }

    fn ensure_nz_flat_scratch(&mut self, len: usize) {
        if self.nz_flat_scratch.len() < len {
            self.nz_flat_scratch.resize(len, 0);
        }
    }
}

std::thread_local! {
    static TRANSFORM_SCRATCH: RefCell<TransformScratch> =
        RefCell::new(TransformScratch::new());
}

fn with_transform_scratch<T>(
    local_n: usize,
    need_error_scratch: bool,
    need_large_block_scratch: bool,
    nz_flat_len: usize,
    f: impl FnOnce(
        &mut [f32],
        &mut [f32],
        &mut [f32],
        &mut Vec<f32>,
        &mut [i32],
        &mut [i32],
        &mut [u8],
    ) -> T,
) -> T {
    const MAX_BLOCK_SIZE: usize = 4096;

    TRANSFORM_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.ensure_local_channels(local_n);
        if need_error_scratch {
            scratch.ensure_error_scratch(MAX_BLOCK_SIZE);
        }
        scratch.ensure_nz_flat_scratch(nz_flat_len);
        if need_large_block_scratch {
            scratch.ensure_quant_scratch(MAX_BLOCK_SIZE);
            scratch.ensure_nz_full_scratch(MAX_BLOCK_SIZE);
        }

        let TransformScratch {
            local_x,
            local_y,
            local_b,
            error_scratch,
            quant_flat_scratch,
            nz_full_block_scratch,
            nz_flat_scratch,
        } = &mut *scratch;

        f(
            &mut local_x[..local_n],
            &mut local_y[..local_n],
            &mut local_b[..local_n],
            error_scratch,
            quant_flat_scratch.as_mut_slice(),
            nz_full_block_scratch.as_mut_slice(),
            &mut nz_flat_scratch[..nz_flat_len],
        )
    })
}

/// Pre-allocated output buffers for `transform_and_quantize`.
///
/// Reuse across butteraugli iterations to avoid re-allocating Vec<Vec<>> arrays.
pub(crate) struct TransformOutput {
    pub quant_dc: [Vec<Vec<i16>>; 3],
    pub quant_ac: [Vec<Vec<[i32; DCT_BLOCK_SIZE]>>; 3],
    pub nzeros: [Vec<Vec<u8>>; 3],
    pub raw_nzeros: [Vec<Vec<u16>>; 3],
    /// Raw (pre-CfL, pre-quantization) float DC values from dc_from_dct_NxN.
    /// These are the correct per-8×8-block DC values that account for multi-block
    /// transform structure (e.g., for DCT16, these come from the 16×16 DCT's LLF
    /// via inverse reinterpreting DCT, NOT from simple 8×8 sub-block pixel averages).
    /// Layout: `[channel][by * xsize_blocks + bx]` in XYB channel order.
    pub float_dc: [Vec<f32>; 3],
}

impl TransformOutput {
    pub fn new(xsize_blocks: usize, ysize_blocks: usize) -> Self {
        let n = xsize_blocks * ysize_blocks;
        Self {
            quant_dc: core::array::from_fn(|_| vec![vec![0i16; xsize_blocks]; ysize_blocks]),
            quant_ac: core::array::from_fn(|_| {
                vec![vec![[0i32; DCT_BLOCK_SIZE]; xsize_blocks]; ysize_blocks]
            }),
            nzeros: core::array::from_fn(|_| vec![vec![0u8; xsize_blocks]; ysize_blocks]),
            raw_nzeros: core::array::from_fn(|_| vec![vec![0u16; xsize_blocks]; ysize_blocks]),
            float_dc: core::array::from_fn(|_| vec![0.0f32; n]),
        }
    }
}

/// Per-group transform results with locally-indexed output arrays.
///
/// Used by `transform_blocks_into` to produce results for a block range
/// that can later be scattered into the global `TransformOutput`.
///
/// All per-block arrays are stored as **flat** `Vec<T>` indexed by
/// `ly * width + lx` to eliminate inner-Vec allocation overhead and
/// improve cache locality when accessing consecutive blocks.
pub(crate) struct GroupTransformResult {
    pub start_bx: usize,
    pub start_by: usize,
    pub width: usize,
    pub height: usize,
    /// Flat: `quant_dc[c][ly * width + lx]`
    pub quant_dc: [Vec<i16>; 3],
    /// Flat: `quant_ac[c][(ly * width + lx)]` (8×8 block, indexed `[lx * width + lx][pos]`)
    pub quant_ac: [Vec<[i32; DCT_BLOCK_SIZE]>; 3],
    /// Flat: `nzeros[c][ly * width + lx]`
    pub nzeros: [Vec<u8>; 3],
    /// Flat: `raw_nzeros[c][ly * width + lx]`
    pub raw_nzeros: [Vec<u16>; 3],
    pub float_dc: [Vec<f32>; 3],
    /// Quant field adjustments: (global_index, new_value).
    pub quant_adjustments: Vec<(usize, u8)>,
}

impl GroupTransformResult {
    pub fn new(start_bx: usize, start_by: usize, width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            start_bx,
            start_by,
            width,
            height,
            quant_dc: core::array::from_fn(|_| vec![0i16; n]),
            quant_ac: core::array::from_fn(|_| vec![[0i32; DCT_BLOCK_SIZE]; n]),
            nzeros: core::array::from_fn(|_| vec![0u8; n]),
            raw_nzeros: core::array::from_fn(|_| vec![0u16; n]),
            float_dc: core::array::from_fn(|_| vec![0.0f32; n]),
            quant_adjustments: Vec::with_capacity(n),
        }
    }

    /// Copy this group's results into the global TransformOutput.
    ///
    /// Uses `copy_from_slice` for all fields so each row maps to a single
    /// `memcpy` call instead of iterating element-by-element.
    pub fn scatter_into(self, out: &mut TransformOutput, xsize_blocks: usize) {
        Self::scatter_into_raw(&self, out, xsize_blocks);
        // quant_adjustments are handled by the caller (scatter loop)
    }

    fn scatter_into_raw(s: &GroupTransformResult, out: &mut TransformOutput, xsize_blocks: usize) {
        for c in 0..3 {
            for ly in 0..s.height {
                let gy = s.start_by + ly;
                let src = ly * s.width;
                let dst = s.start_bx;
                let w = s.width;

                out.quant_dc[c][gy][dst..dst + w].copy_from_slice(&s.quant_dc[c][src..src + w]);
                out.quant_ac[c][gy][dst..dst + w]
                    .copy_from_slice(&s.quant_ac[c][ly * s.width..ly * s.width + w]);
                out.nzeros[c][gy][dst..dst + w].copy_from_slice(&s.nzeros[c][src..src + w]);
                out.raw_nzeros[c][gy][dst..dst + w].copy_from_slice(&s.raw_nzeros[c][src..src + w]);

                let out_fdc = gy * xsize_blocks + dst;
                out.float_dc[c][out_fdc..out_fdc + w].copy_from_slice(&s.float_dc[c][src..src + w]);
            }
        }
    }
}

impl VarDctEncoder {
    #[inline]
    fn apply_dct8(channel_data: &[f32], stride: usize, bx: usize, by: usize, output: &mut [f32]) {
        use super::common::{as_array_mut, uninit_buf};

        let mut block = uninit_buf::<64>();
        extract_block_8x8(channel_data, stride, bx, by, &mut block);
        dct_8x8(&block, as_array_mut(output, 0));
    }

    /// Apply DCT to a single channel at block position (bx, by).
    ///
    /// The `channel_data` must be padded to block boundaries (stride = padded_width).
    /// No bounds checking is performed - caller must ensure data is properly padded.
    pub(crate) fn apply_dct(
        channel_data: &[f32],
        stride: usize, // padded_width (row stride)
        bx: usize,
        by: usize,
        raw_strategy: u8,
        output: &mut [f32],
    ) {
        use super::common::{as_array_mut, uninit_buf};

        match raw_strategy {
            0 => {
                Self::apply_dct8(channel_data, stride, bx, by, output);
            }
            RAW_STRATEGY_DCT16X8 => {
                let mut block = uninit_buf::<128>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..16 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 8..dy * 8 + 8].copy_from_slice(&channel_data[src..src + 8]);
                }
                dct_16x8(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT8X16 => {
                let mut block = uninit_buf::<128>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..8 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 16..dy * 16 + 16].copy_from_slice(&channel_data[src..src + 16]);
                }
                dct_8x16(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT16X16 => {
                let mut block = uninit_buf::<256>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..16 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 16..dy * 16 + 16].copy_from_slice(&channel_data[src..src + 16]);
                }
                dct_16x16(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT32X32 => {
                let mut block = uninit_buf::<1024>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..32 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 32..dy * 32 + 32].copy_from_slice(&channel_data[src..src + 32]);
                }
                dct_32x32(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT4X8 => {
                let mut block = uninit_buf::<64>();
                extract_block_8x8(channel_data, stride, bx, by, &mut block);
                dct_4x8_full(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT8X4 => {
                let mut block = uninit_buf::<64>();
                extract_block_8x8(channel_data, stride, bx, by, &mut block);
                dct_8x4_full(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT4X4 => {
                let mut block = uninit_buf::<64>();
                extract_block_8x8(channel_data, stride, bx, by, &mut block);
                dct_4x4_full(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_IDENTITY => {
                let mut input = uninit_buf::<64>();
                extract_block_8x8(channel_data, stride, bx, by, &mut input);
                identity_transform(&input, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT2X2 => {
                let mut input = uninit_buf::<64>();
                extract_block_8x8(channel_data, stride, bx, by, &mut input);
                dct2x2_transform(&input, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT32X16 => {
                let mut block = uninit_buf::<512>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..32 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 16..dy * 16 + 16].copy_from_slice(&channel_data[src..src + 16]);
                }
                dct_32x16(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT16X32 => {
                let mut block = uninit_buf::<512>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..16 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 32..dy * 32 + 32].copy_from_slice(&channel_data[src..src + 32]);
                }
                dct_16x32(&block, as_array_mut(output, 0));
            }
            RAW_STRATEGY_DCT64X64 => {
                let mut block = uninit_buf::<4096>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..64 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 64..dy * 64 + 64].copy_from_slice(&channel_data[src..src + 64]);
                }
                dct_64x64(&block, &mut output[..4096]);
            }
            RAW_STRATEGY_DCT64X32 => {
                let mut block = uninit_buf::<2048>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..64 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 32..dy * 32 + 32].copy_from_slice(&channel_data[src..src + 32]);
                }
                dct_64x32(&block, &mut output[..2048]);
            }
            RAW_STRATEGY_DCT32X64 => {
                let mut block = uninit_buf::<2048>();
                let x0 = bx * BLOCK_DIM;
                for dy in 0..32 {
                    let src = (by * BLOCK_DIM + dy) * stride + x0;
                    block[dy * 64..dy * 64 + 64].copy_from_slice(&channel_data[src..src + 64]);
                }
                dct_32x64(&block, &mut output[..2048]);
            }
            RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2 | RAW_STRATEGY_AFV3 => {
                let mut pixels = uninit_buf::<64>();
                extract_block_8x8(channel_data, stride, bx, by, &mut pixels);
                let afv_kind = (raw_strategy - RAW_STRATEGY_AFV0) as usize;
                afv_transform_from_pixels(&pixels, afv_kind, as_array_mut(output, 0));
            }
            _ => unreachable!(),
        }
    }

    /// Process a RANGE of blocks, writing to a `GroupTransformResult` with local coordinates.
    ///
    /// This is the same algorithm as `transform_and_quantize_into` but operates on a
    /// sub-rectangle `[start_by..end_by, start_bx..end_bx]` and writes results into
    /// locally-indexed arrays in `result`. The `quant_field` is read-only; any quant
    /// adjustments are recorded in `result.quant_adjustments` for later application.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transform_blocks_into(
        &self,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        padded_width: usize,
        xsize_blocks: usize,
        params: &DistanceParams,
        quant_field: &[u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        start_by: usize,
        end_by: usize,
        start_bx: usize,
        end_bx: usize,
        result: &mut GroupTransformResult,
    ) {
        let yoff = start_by;
        let xoff = start_bx;
        let width = result.width;

        let quant_dc = &mut result.quant_dc;
        let quant_ac = &mut result.quant_ac;
        let nzeros = &mut result.nzeros;
        let raw_nzeros = &mut result.raw_nzeros;
        let float_dc = &mut result.float_dc;
        let cfl_zero_map = cfl_map.is_zero;

        // Hoist constant computations out of the block loop
        let x_qm_mul = crate::jxl_encode_simd::fast_powf(1.25, params.x_qm_scale as f32 - 2.0);
        let b_qm_mul = crate::jxl_encode_simd::fast_powf(1.25, params.b_qm_scale as f32 - 2.0);

        // Pre-allocate scratch buffers for DCT coefficients. The import profile
        // forces DCT8, so avoid allocating DCT64-sized scratch per group there.
        const MAX_BLOCK_SIZE: usize = 4096;
        let forced_dct8 = self.force_strategy == Some(0);
        let scratch_block_size = if forced_dct8 {
            DCT_BLOCK_SIZE
        } else {
            MAX_BLOCK_SIZE
        };
        let mut dct_scratch: [Vec<f32>; 3] =
            core::array::from_fn(|_| crate::jxl_encode_simd::vec_f32_dirty(scratch_block_size));

        // Pre-compute zigzag orders for error diffusion (avoids per-block Vec allocation).
        // Index by (cx, cy) pair. Only 7 distinct pairs across all strategies.
        use super::coeff_order::natural_coeff_order;
        let zigzag_cache: Vec<(usize, usize, Vec<u32>)> = if self.error_diffusion {
            [(1, 1), (2, 1), (2, 2), (4, 2), (4, 4), (8, 4), (8, 8)]
                .iter()
                .map(|&(cx, cy)| (cx, cy, natural_coeff_order(cx, cy)))
                .collect()
        } else {
            Vec::new()
        };
        // ── XYB group cache blocking ─────────────────────────────────
        // Pre-copy this group's pixel rows into compact local buffers so
        // extract_block_8x8 reads with stride = local_pixel_width instead of
        // padded_width (~9-12KB for large scales), eliminating cache misses.
        let group_pixel_x0 = start_bx * BLOCK_DIM;
        let group_pixel_y0 = start_by * BLOCK_DIM;
        let group_pixel_x1 = end_bx * BLOCK_DIM;
        let group_pixel_y1 = end_by * BLOCK_DIM;
        let local_pixel_w = group_pixel_x1 - group_pixel_x0;
        let local_pixel_h = group_pixel_y1 - group_pixel_y0;
        let local_n = local_pixel_w * local_pixel_h;

        with_transform_scratch(
            local_n,
            self.error_diffusion,
            !forced_dct8,
            7 * width + 8,
            |local_x,
             local_y,
             local_b,
             error_scratch,
             quant_flat_scratch,
             nz_full_block_scratch,
             nz_flat_scratch| {
                for dy in 0..local_pixel_h {
                    let src_row = (group_pixel_y0 + dy) * padded_width + group_pixel_x0;
                    let dst_row = dy * local_pixel_w;
                    local_x[dst_row..dst_row + local_pixel_w]
                        .copy_from_slice(&xyb_x[src_row..src_row + local_pixel_w]);
                    local_y[dst_row..dst_row + local_pixel_w]
                        .copy_from_slice(&xyb_y[src_row..src_row + local_pixel_w]);
                    local_b[dst_row..dst_row + local_pixel_w]
                        .copy_from_slice(&xyb_b[src_row..src_row + local_pixel_w]);
                }
                // Use local buffers + local stride for DCT; block coords are local (bx-start_bx, by-start_by)
                let local_channels: [&[f32]; 3] = [local_x, local_y, local_b];

                for by in start_by..end_by {
                    for bx in start_bx..end_bx {
                        // Skip non-first blocks of multi-block transforms
                        if !forced_dct8 && !ac_strategy.is_first(bx, by) {
                            continue;
                        }

                        let raw_strategy = if forced_dct8 {
                            0
                        } else {
                            ac_strategy.raw_strategy(bx, by)
                        };
                        #[cfg(feature = "debug-dc")]
                        eprintln!(
                            "Block (by={}, bx={}): raw_strategy={}",
                            by, bx, raw_strategy
                        );
                        let (covered_x, covered_y) = if forced_dct8 {
                            (1, 1)
                        } else {
                            (
                                ac_strategy.covered_blocks_x(bx, by),
                                ac_strategy.covered_blocks_y(bx, by),
                            )
                        };
                        let covered_blocks = covered_x * covered_y;
                        let size = covered_blocks * DCT_BLOCK_SIZE;

                        // CfL factors for this tile
                        let (x_factor, b_factor) = if cfl_zero_map {
                            (0.0, 1.0)
                        } else {
                            let tx = bx / TILE_DIM_IN_BLOCKS;
                            let ty_cfl = by / TILE_DIM_IN_BLOCKS;
                            (
                                ytox_ratio(cfl_map.ytox_at(tx, ty_cfl)),
                                ytob_ratio(cfl_map.ytob_at(tx, ty_cfl)),
                            )
                        };

                        // Coefficient layout: after C++ swap(cx,cy) so cx >= cy,
                        // stride = cx * 8. Both DCT16X8 and DCT8X16 produce 8×16 layout.
                        let (cx, cy) = if covered_y > covered_x {
                            (covered_y, covered_x)
                        } else {
                            (covered_x, covered_y)
                        };
                        let block_width = cx * BLOCK_DIM;
                        let block_height = cy * BLOCK_DIM;

                        // No fill needed — apply_dct writes all output positions
                        // Alias for readability — dct_coeffs[c] is dct_scratch[c][..size]
                        let dct_coeffs = &mut dct_scratch;

                        // ── Step 1: DCT Y channel ──────────────────────────────────
                        if forced_dct8 {
                            Self::apply_dct8(
                                local_channels[1],
                                local_pixel_w,
                                bx - start_bx,
                                by - start_by,
                                &mut dct_coeffs[1],
                            );
                        } else {
                            Self::apply_dct(
                                local_channels[1],
                                local_pixel_w,
                                bx - start_bx,
                                by - start_by,
                                raw_strategy,
                                &mut dct_coeffs[1],
                            );
                        }

                        // ── Step 2: Extract Y DC (before roundtrip quantization) ───
                        // Inlined instead of using extract_dc to avoid borrow conflict.
                        {
                            let inv_factor = INV_DC_QUANT[1] * params.scale_dc;
                            if forced_dct8 {
                                float_dc[1][(by - yoff) * width + (bx - xoff)] = dct_coeffs[1][0];
                                quant_dc[1][(by - yoff) * width + (bx - xoff)] =
                                    (dct_coeffs[1][0] * inv_factor).round() as i16;
                            } else {
                                match raw_strategy {
                                    0 => {
                                        #[cfg(feature = "debug-dc")]
                                        eprintln!(
                                            "DCT8 Y DC: dct[0]={:.6}, inv_factor={:.4}, scale_dc={:.6}, quant_dc={}",
                                            dct_coeffs[1][0],
                                            inv_factor,
                                            params.scale_dc,
                                            (dct_coeffs[1][0] * inv_factor).round() as i16
                                        );
                                        float_dc[1][(by - yoff) * width + (bx - xoff)] =
                                            dct_coeffs[1][0];
                                        quant_dc[1][(by - yoff) * width + (bx - xoff)] =
                                            (dct_coeffs[1][0] * inv_factor).round() as i16;
                                    }
                                    RAW_STRATEGY_DCT16X8 => {
                                        let dcs = dc_from_dct_16x8(as_array_ref::<128>(
                                            &dct_coeffs[1],
                                            0,
                                        ));
                                        for iy in 0..2 {
                                            float_dc[1][(by - yoff + iy) * width + (bx - xoff)] =
                                                dcs[iy];
                                            quant_dc[1][(by - yoff + iy) * width + (bx - xoff)] =
                                                (dcs[iy] * inv_factor).round() as i16;
                                        }
                                    }
                                    RAW_STRATEGY_DCT8X16 => {
                                        let dcs = dc_from_dct_8x16(as_array_ref::<128>(
                                            &dct_coeffs[1],
                                            0,
                                        ));
                                        for ix in 0..2 {
                                            float_dc[1][(by - yoff) * width + (bx - xoff + ix)] =
                                                dcs[ix];
                                            quant_dc[1][(by - yoff) * width + (bx - xoff + ix)] =
                                                (dcs[ix] * inv_factor).round() as i16;
                                        }
                                    }
                                    RAW_STRATEGY_DCT16X16 => {
                                        let dcs = dc_from_dct_16x16(as_array_ref::<256>(
                                            &dct_coeffs[1],
                                            0,
                                        ));
                                        #[cfg(feature = "debug-dc")]
                                        eprintln!(
                                            "DCT16x16 block (by={}, bx={}): dcs=[{:.4}, {:.4}, {:.4}, {:.4}], LLF=[{:.6}, {:.6}, {:.6}, {:.6}]",
                                            by,
                                            bx,
                                            dcs[0],
                                            dcs[1],
                                            dcs[2],
                                            dcs[3],
                                            dct_coeffs[1][0],
                                            dct_coeffs[1][1],
                                            dct_coeffs[1][16],
                                            dct_coeffs[1][17]
                                        );
                                        for iy in 0..2 {
                                            for ix in 0..2 {
                                                float_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 2 + ix];
                                                let qdc =
                                                    (dcs[iy * 2 + ix] * inv_factor).round() as i16;
                                                #[cfg(feature = "debug-dc")]
                                                eprintln!(
                                                    "  quant_dc[1][{}][{}] = {} (raw dc={:.4}, inv_factor={:.4})",
                                                    by + iy,
                                                    bx + ix,
                                                    qdc,
                                                    dcs[iy * 2 + ix],
                                                    inv_factor
                                                );
                                                quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    qdc;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT32X32 => {
                                        let dcs = dc_from_dct_32x32(as_array_ref::<1024>(
                                            &dct_coeffs[1],
                                            0,
                                        ));
                                        #[cfg(feature = "debug-dc")]
                                        eprintln!(
                                            "DCT32x32 block (by={}, bx={}): dcs[0..4]=[{:.4}, {:.4}, {:.4}, {:.4}], LLF=[{:.6}, {:.6}, {:.6}, {:.6}]",
                                            by,
                                            bx,
                                            dcs[0],
                                            dcs[1],
                                            dcs[2],
                                            dcs[3],
                                            dct_coeffs[1][0],
                                            dct_coeffs[1][1],
                                            dct_coeffs[1][32],
                                            dct_coeffs[1][33]
                                        );
                                        // dcs = 16 DC values in row-major 4x4
                                        for iy in 0..4 {
                                            for ix in 0..4 {
                                                float_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 4 + ix];
                                                let qdc =
                                                    (dcs[iy * 4 + ix] * inv_factor).round() as i16;
                                                #[cfg(feature = "debug-dc")]
                                                eprintln!(
                                                    "  quant_dc[1][{}][{}] = {} (raw dc={:.4})",
                                                    by + iy,
                                                    bx + ix,
                                                    qdc,
                                                    dcs[iy * 4 + ix]
                                                );
                                                quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    qdc;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT32X16 => {
                                        // DCT32X16: 4 block rows × 2 block cols, returns 8 DC values in 4×2 order
                                        let dcs = dc_from_dct_32x16(as_array_ref::<512>(
                                            &dct_coeffs[1],
                                            0,
                                        ));
                                        for iy in 0..4 {
                                            for ix in 0..2 {
                                                float_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 2 + ix];
                                                let qdc =
                                                    (dcs[iy * 2 + ix] * inv_factor).round() as i16;
                                                quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    qdc;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT16X32 => {
                                        // DCT16X32: 2×4 blocks, returns 8 DC values in row-major 2x4
                                        let dcs = dc_from_dct_16x32(as_array_ref::<512>(
                                            &dct_coeffs[1],
                                            0,
                                        ));
                                        for iy in 0..2 {
                                            for ix in 0..4 {
                                                float_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 4 + ix];
                                                let qdc =
                                                    (dcs[iy * 4 + ix] * inv_factor).round() as i16;
                                                quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    qdc;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT64X64 => {
                                        // DCT64X64: 8×8 blocks, returns 64 DC values in row-major 8x8
                                        let dcs = dc_from_dct_64x64(&dct_coeffs[1]);
                                        for iy in 0..8 {
                                            for ix in 0..8 {
                                                float_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 8 + ix];
                                                let qdc =
                                                    (dcs[iy * 8 + ix] * inv_factor).round() as i16;
                                                quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    qdc;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT64X32 => {
                                        // DCT64X32: 8 block rows × 4 block cols, returns 32 DC values in 8×4 order
                                        let dcs = dc_from_dct_64x32(&dct_coeffs[1]);
                                        for iy in 0..8 {
                                            for ix in 0..4 {
                                                float_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 4 + ix];
                                                let qdc =
                                                    (dcs[iy * 4 + ix] * inv_factor).round() as i16;
                                                quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    qdc;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT32X64 => {
                                        // DCT32X64: 4×8 blocks, returns 32 DC values in row-major 4x8
                                        let dcs = dc_from_dct_32x64(&dct_coeffs[1]);
                                        for iy in 0..4 {
                                            for ix in 0..8 {
                                                float_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 8 + ix];
                                                let qdc =
                                                    (dcs[iy * 8 + ix] * inv_factor).round() as i16;
                                                quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    qdc;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT4X8 => {
                                        let dc = dc_from_dct_4x8_full(as_array_ref::<64>(
                                            &dct_coeffs[1],
                                            0,
                                        ));
                                        float_dc[1][(by - yoff) * width + (bx - xoff)] = dc;
                                        quant_dc[1][(by - yoff) * width + (bx - xoff)] =
                                            (dc * inv_factor).round() as i16;
                                    }
                                    RAW_STRATEGY_DCT8X4 => {
                                        let dc = dc_from_dct_8x4_full(as_array_ref::<64>(
                                            &dct_coeffs[1],
                                            0,
                                        ));
                                        float_dc[1][(by - yoff) * width + (bx - xoff)] = dc;
                                        quant_dc[1][(by - yoff) * width + (bx - xoff)] =
                                            (dc * inv_factor).round() as i16;
                                    }
                                    RAW_STRATEGY_DCT4X4 => {
                                        let dc = dc_from_dct_4x4_full(as_array_ref::<64>(
                                            &dct_coeffs[1],
                                            0,
                                        ));
                                        float_dc[1][(by - yoff) * width + (bx - xoff)] = dc;
                                        quant_dc[1][(by - yoff) * width + (bx - xoff)] =
                                            (dc * inv_factor).round() as i16;
                                    }
                                    RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2
                                    | RAW_STRATEGY_AFV3 => {
                                        let dc = dc_from_afv(as_array_ref::<64>(&dct_coeffs[1], 0));
                                        float_dc[1][(by - yoff) * width + (bx - xoff)] = dc;
                                        quant_dc[1][(by - yoff) * width + (bx - xoff)] =
                                            (dc * inv_factor).round() as i16;
                                    }
                                    RAW_STRATEGY_IDENTITY | RAW_STRATEGY_DCT2X2 => {
                                        // IDENTITY/DCT2X2: 1×1 coverage, DC at position [0]
                                        float_dc[1][(by - yoff) * width + (bx - xoff)] =
                                            dct_coeffs[1][0];
                                        quant_dc[1][(by - yoff) * width + (bx - xoff)] =
                                            (dct_coeffs[1][0] * inv_factor).round() as i16;
                                    }
                                    _ => unreachable!(),
                                }
                            }
                        }

                        // ── Step 2b: DCT X and B channels (before AdjustQuantBlockAC) ──
                        // libjxl DCTs all 3 channels before running AdjustQuantBlockAC.
                        // X/B coefficients here are pre-CfL (CfL subtraction happens later in Step 6).
                        for &c in &[0usize, 2] {
                            if forced_dct8 {
                                Self::apply_dct8(
                                    local_channels[c],
                                    local_pixel_w,
                                    bx - start_bx,
                                    by - start_by,
                                    &mut dct_coeffs[c],
                                );
                            } else {
                                Self::apply_dct(
                                    local_channels[c],
                                    local_pixel_w,
                                    bx - start_bx,
                                    by - start_by,
                                    raw_strategy,
                                    &mut dct_coeffs[c],
                                );
                            }
                        }

                        // ── Step 2c: AdjustQuantBlockAC ──────────────────────────────
                        // Ported from libjxl enc_group.cc QuantizeRoundtripYBlockAC.
                        // libjxl gates on speed_tier <= kHare (effort >= 5):
                        // adjusts per-block quant and Y thresholds based on coefficient
                        // statistics across all 3 channels.
                        // At effort < 5: uses fixed thresholds, no per-block adjustment.
                        let mut thresholds_y;
                        let qac;
                        {
                            let quant_idx = by * xsize_blocks + bx;
                            let mut quant_int = quant_field[quant_idx] as i32;
                            if self.profile.adjust_quant_ac {
                                // effort >= Hare: run AdjustQuantBlockAC for all 3 channels
                                let orig_qac = params.scale * quant_int as f32;
                                thresholds_y = self.profile.adjust_thresholds;
                                let mut max_quant = quant_int;
                                for &c in &[1usize, 0, 2] {
                                    let mut thres = self.profile.adjust_thresholds;
                                    let mut quant_c = quant_int;
                                    let qm_mul = if c == 0 {
                                        x_qm_mul
                                    } else if c == 2 {
                                        b_qm_mul
                                    } else {
                                        1.0
                                    };
                                    let weights_c =
                                        super::quant::quant_weights(raw_strategy as usize, c);
                                    let (hflags, vals, err, activity) = Self::adjust_quant_block_ac(
                                        &dct_coeffs[c],
                                        weights_c,
                                        orig_qac,
                                        qm_mul,
                                        c,
                                        raw_strategy,
                                        block_width,
                                        block_height,
                                        cx,
                                        cy,
                                        &mut thres,
                                        &mut quant_c,
                                    );
                                    if c == 1 {
                                        thresholds_y = thres;
                                        debug_rect!(
                                            "quant/heur",
                                            bx * 8,
                                            by * 8,
                                            cx * 8,
                                            cy * 8,
                                            "c=Y flags={:06b} vals={:.0} err={:.1} act={} q={}→{}",
                                            hflags,
                                            vals,
                                            err,
                                            activity,
                                            quant_int,
                                            quant_c
                                        );
                                    }
                                    max_quant = max_quant.max(quant_c);
                                }
                                quant_int = max_quant;
                                let new_quant = quant_int.clamp(1, 255) as u8;
                                result.quant_adjustments.push((quant_idx, new_quant));
                                debug_rect!(
                                    "quant/adjust",
                                    bx * 8,
                                    by * 8,
                                    cx * 8,
                                    cy * 8,
                                    "strat={} q={}→{} (e>=5 AdjustQuantBlockAC)",
                                    raw_strategy,
                                    quant_field[quant_idx],
                                    new_quant
                                );
                            } else {
                                // effort < Hare: fixed thresholds, no per-block adjustment
                                // (enc_group.cc:358-363)
                                thresholds_y = self.profile.fixed_thresholds_y;
                            }
                            qac = params.scale * quant_int as f32;
                        }

                        // ── Step 3: Quantize Y AC with thresholding ────────────────
                        {
                            let c = 1;
                            let weights = super::quant::quant_weights(raw_strategy as usize, c);
                            let zigzag = if self.error_diffusion {
                                zigzag_cache
                                    .iter()
                                    .find(|(cx2, cy2, _)| *cx2 == cx && *cy2 == cy)
                                    .map(|(_, _, v)| v.as_slice())
                            } else {
                                None
                            };
                            Self::quantize_ac_block(
                                &dct_coeffs[c],
                                weights,
                                qac,
                                1.0, // no x_qm_mul for Y
                                &thresholds_y,
                                block_width,
                                block_height,
                                covered_x,
                                covered_y,
                                covered_blocks,
                                size,
                                raw_strategy,
                                bx - xoff,
                                by - yoff,
                                &mut quant_ac[c],
                                width,
                                self.error_diffusion,
                                zigzag,
                                if self.error_diffusion {
                                    Some(error_scratch)
                                } else {
                                    None
                                },
                                quant_flat_scratch,
                            );
                        }

                        // ── Step 4: Dequantize Y back (AdjustQuantBias roundtrip) ──
                        let weights = super::quant::quant_weights(raw_strategy as usize, 1);
                        let inv_qac = 1.0 / qac;
                        let transpose_slots = covered_y > covered_x;
                        // Use post-swap dimensions for grid (matches C++ and quantize_ac_block).
                        // Nested loops eliminate per-element integer divisions.
                        // Pre-slice weights and dct_coeffs rows to eliminate inner bounds checks.
                        for coef_slot_y in 0..cy {
                            for pos_y in 0..BLOCK_DIM {
                                let y = coef_slot_y * BLOCK_DIM + pos_y;
                                let is_llf_row = y < cy;
                                let row_off = y * block_width;
                                let w_row = &weights[row_off..row_off + block_width];
                                let coeff_row = &mut dct_coeffs[1][row_off..row_off + block_width];
                                for coef_slot_x in 0..cx {
                                    for pos_x in 0..BLOCK_DIM {
                                        let x = coef_slot_x * BLOCK_DIM + pos_x;
                                        let is_llf = is_llf_row && x < cx;
                                        let q = if is_llf {
                                            Self::quantize_coeff_ac(
                                                coeff_row[x],
                                                1.0 / w_row[x],
                                                qac,
                                                1.0,
                                                &thresholds_y,
                                                y,
                                                x,
                                                block_height,
                                                block_width,
                                            )
                                        } else {
                                            let (phys_row_off, phys_col_off) = if transpose_slots {
                                                (coef_slot_x, coef_slot_y)
                                            } else {
                                                (coef_slot_y, coef_slot_x)
                                            };
                                            let pos_in_8x8 = pos_y * BLOCK_DIM + pos_x;
                                            quant_ac[1][(by - yoff + phys_row_off) * width
                                                + (bx - xoff + phys_col_off)][pos_in_8x8]
                                        };
                                        let adj = adjust_quant_bias(q, 1);
                                        coeff_row[x] = adj * w_row[x] * inv_qac;
                                    }
                                }
                            }
                        }

                        // ── Step 5: CfL on AC coefficients using roundtripped Y ───
                        // X/B DCTs were done in Step 2b (before AdjustQuantBlockAC).
                        // C++ applies CfL to ALL positions (0..size) including DC/LLF,
                        // but the decoder's DequantBlock calls LowestFrequenciesFromDC
                        // AFTER DequantLane, overwriting LLF positions with DC-derived
                        // values. So coefficient-level CfL on LLF is discarded by the
                        // decoder. We skip LLF here; DC CfL uses dc_cfl_factor instead.
                        #[allow(clippy::needless_range_loop)]
                        // Nested loops eliminate per-element div/mod; split_at_mut for disjoint refs;
                        // pre-slice rows to eliminate inner bounds checks.
                        {
                            let (dc_x, rest) = dct_coeffs.split_at_mut(1);
                            let (dc_y, dc_b) = rest.split_at_mut(1);
                            if cfl_zero_map {
                                for y in 0..block_height {
                                    let x_start = if y < cy { cx } else { 0 };
                                    let row_off = y * block_width;
                                    let yr = &dc_y[0][row_off..row_off + block_width];
                                    let br = &mut dc_b[0][row_off..row_off + block_width];
                                    for x in x_start..block_width {
                                        br[x] -= yr[x];
                                    }
                                }
                            } else {
                                for y in 0..block_height {
                                    let x_start = if y < cy { cx } else { 0 };
                                    let row_off = y * block_width;
                                    let yr = &dc_y[0][row_off..row_off + block_width];
                                    let xr = &mut dc_x[0][row_off..row_off + block_width];
                                    let br = &mut dc_b[0][row_off..row_off + block_width];
                                    for x in x_start..block_width {
                                        xr[x] -= x_factor * yr[x];
                                        br[x] -= b_factor * yr[x];
                                    }
                                }
                            }
                        }

                        // ── Step 7: Extract X/B DC + quantize X/B AC ───────────────
                        for &c in &[0usize, 2] {
                            let dc_cfl_factor = if c == 2 { 0.5f32 } else { 0.0f32 };
                            let inv_factor = INV_DC_QUANT[c] * params.scale_dc;
                            let qm_multiplier = if c == 0 {
                                x_qm_mul
                            } else if c == 2 {
                                b_qm_mul
                            } else {
                                1.0
                            };

                            // Extract DC from CfL-adjusted coefficients.
                            // Read Y DC into temporaries to avoid borrow conflict
                            // (can't have &quant_dc[1] and &mut quant_dc[c] simultaneously).
                            if forced_dct8 {
                                let dc = dct_coeffs[c][0];
                                float_dc[c][(by - yoff) * width + (bx - xoff)] = dc;
                                let y_dc = quant_dc[1][(by - yoff) * width + (bx - xoff)] as f32;
                                quant_dc[c][(by - yoff) * width + (bx - xoff)] =
                                    (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                            } else {
                                match raw_strategy {
                                    0 => {
                                        let dc = dct_coeffs[c][0];
                                        float_dc[c][(by - yoff) * width + (bx - xoff)] = dc;
                                        let y_dc =
                                            quant_dc[1][(by - yoff) * width + (bx - xoff)] as f32;
                                        quant_dc[c][(by - yoff) * width + (bx - xoff)] =
                                            (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                                    }
                                    RAW_STRATEGY_DCT16X8 => {
                                        let dcs = dc_from_dct_16x8(as_array_ref::<128>(
                                            &dct_coeffs[c],
                                            0,
                                        ));
                                        for iy in 0..2 {
                                            float_dc[c][(by - yoff + iy) * width + (bx - xoff)] =
                                                dcs[iy];
                                            let y_dc = quant_dc[1]
                                                [(by - yoff + iy) * width + (bx - xoff)]
                                                as f32;
                                            quant_dc[c][(by - yoff + iy) * width + (bx - xoff)] =
                                                (dcs[iy] * inv_factor - y_dc * dc_cfl_factor)
                                                    .round()
                                                    as i16;
                                        }
                                    }
                                    RAW_STRATEGY_DCT8X16 => {
                                        let dcs = dc_from_dct_8x16(as_array_ref::<128>(
                                            &dct_coeffs[c],
                                            0,
                                        ));
                                        for ix in 0..2 {
                                            float_dc[c][(by - yoff) * width + (bx - xoff + ix)] =
                                                dcs[ix];
                                            let y_dc = quant_dc[1]
                                                [(by - yoff) * width + (bx - xoff + ix)]
                                                as f32;
                                            quant_dc[c][(by - yoff) * width + (bx - xoff + ix)] =
                                                (dcs[ix] * inv_factor - y_dc * dc_cfl_factor)
                                                    .round()
                                                    as i16;
                                        }
                                    }
                                    RAW_STRATEGY_DCT16X16 => {
                                        let dcs = dc_from_dct_16x16(as_array_ref::<256>(
                                            &dct_coeffs[c],
                                            0,
                                        ));
                                        for iy in 0..2 {
                                            for ix in 0..2 {
                                                float_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 2 + ix];
                                                let y_dc = quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)]
                                                    as f32;
                                                quant_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    (dcs[iy * 2 + ix] * inv_factor
                                                        - y_dc * dc_cfl_factor)
                                                        .round()
                                                        as i16;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT32X32 => {
                                        let dcs = dc_from_dct_32x32(as_array_ref::<1024>(
                                            &dct_coeffs[c],
                                            0,
                                        ));
                                        for iy in 0..4 {
                                            for ix in 0..4 {
                                                float_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 4 + ix];
                                                let y_dc = quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)]
                                                    as f32;
                                                quant_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    (dcs[iy * 4 + ix] * inv_factor
                                                        - y_dc * dc_cfl_factor)
                                                        .round()
                                                        as i16;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT32X16 => {
                                        let dcs = dc_from_dct_32x16(as_array_ref::<512>(
                                            &dct_coeffs[c],
                                            0,
                                        ));
                                        for iy in 0..4 {
                                            for ix in 0..2 {
                                                float_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 2 + ix];
                                                let y_dc = quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)]
                                                    as f32;
                                                quant_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    (dcs[iy * 2 + ix] * inv_factor
                                                        - y_dc * dc_cfl_factor)
                                                        .round()
                                                        as i16;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT16X32 => {
                                        let dcs = dc_from_dct_16x32(as_array_ref::<512>(
                                            &dct_coeffs[c],
                                            0,
                                        ));
                                        for iy in 0..2 {
                                            for ix in 0..4 {
                                                float_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 4 + ix];
                                                let y_dc = quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)]
                                                    as f32;
                                                quant_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    (dcs[iy * 4 + ix] * inv_factor
                                                        - y_dc * dc_cfl_factor)
                                                        .round()
                                                        as i16;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT64X64 => {
                                        let dcs = dc_from_dct_64x64(&dct_coeffs[c]);
                                        for iy in 0..8 {
                                            for ix in 0..8 {
                                                float_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 8 + ix];
                                                let y_dc = quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)]
                                                    as f32;
                                                quant_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    (dcs[iy * 8 + ix] * inv_factor
                                                        - y_dc * dc_cfl_factor)
                                                        .round()
                                                        as i16;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT64X32 => {
                                        let dcs = dc_from_dct_64x32(&dct_coeffs[c]);
                                        for iy in 0..8 {
                                            for ix in 0..4 {
                                                float_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 4 + ix];
                                                let y_dc = quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)]
                                                    as f32;
                                                quant_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    (dcs[iy * 4 + ix] * inv_factor
                                                        - y_dc * dc_cfl_factor)
                                                        .round()
                                                        as i16;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT32X64 => {
                                        let dcs = dc_from_dct_32x64(&dct_coeffs[c]);
                                        for iy in 0..4 {
                                            for ix in 0..8 {
                                                float_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    dcs[iy * 8 + ix];
                                                let y_dc = quant_dc[1]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)]
                                                    as f32;
                                                quant_dc[c]
                                                    [(by - yoff + iy) * width + (bx - xoff + ix)] =
                                                    (dcs[iy * 8 + ix] * inv_factor
                                                        - y_dc * dc_cfl_factor)
                                                        .round()
                                                        as i16;
                                            }
                                        }
                                    }
                                    RAW_STRATEGY_DCT4X8 => {
                                        let dc = dc_from_dct_4x8_full(as_array_ref::<64>(
                                            &dct_coeffs[c],
                                            0,
                                        ));
                                        float_dc[c][(by - yoff) * width + (bx - xoff)] = dc;
                                        let y_dc =
                                            quant_dc[1][(by - yoff) * width + (bx - xoff)] as f32;
                                        quant_dc[c][(by - yoff) * width + (bx - xoff)] =
                                            (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                                    }
                                    RAW_STRATEGY_DCT8X4 => {
                                        let dc = dc_from_dct_8x4_full(as_array_ref::<64>(
                                            &dct_coeffs[c],
                                            0,
                                        ));
                                        float_dc[c][(by - yoff) * width + (bx - xoff)] = dc;
                                        let y_dc =
                                            quant_dc[1][(by - yoff) * width + (bx - xoff)] as f32;
                                        quant_dc[c][(by - yoff) * width + (bx - xoff)] =
                                            (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                                    }
                                    RAW_STRATEGY_DCT4X4 => {
                                        let dc = dc_from_dct_4x4_full(as_array_ref::<64>(
                                            &dct_coeffs[c],
                                            0,
                                        ));
                                        float_dc[c][(by - yoff) * width + (bx - xoff)] = dc;
                                        let y_dc =
                                            quant_dc[1][(by - yoff) * width + (bx - xoff)] as f32;
                                        quant_dc[c][(by - yoff) * width + (bx - xoff)] =
                                            (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                                    }
                                    RAW_STRATEGY_AFV0 | RAW_STRATEGY_AFV1 | RAW_STRATEGY_AFV2
                                    | RAW_STRATEGY_AFV3 => {
                                        let dc = dc_from_afv(as_array_ref::<64>(&dct_coeffs[c], 0));
                                        float_dc[c][(by - yoff) * width + (bx - xoff)] = dc;
                                        let y_dc =
                                            quant_dc[1][(by - yoff) * width + (bx - xoff)] as f32;
                                        quant_dc[c][(by - yoff) * width + (bx - xoff)] =
                                            (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                                    }
                                    RAW_STRATEGY_IDENTITY | RAW_STRATEGY_DCT2X2 => {
                                        // IDENTITY/DCT2X2: 1×1 coverage, DC at position [0]
                                        let dc = dct_coeffs[c][0];
                                        float_dc[c][(by - yoff) * width + (bx - xoff)] = dc;
                                        let y_dc =
                                            quant_dc[1][(by - yoff) * width + (bx - xoff)] as f32;
                                        quant_dc[c][(by - yoff) * width + (bx - xoff)] =
                                            (dc * inv_factor - y_dc * dc_cfl_factor).round() as i16;
                                    }
                                    _ => unreachable!(),
                                }
                            }

                            // Quantize AC with thresholding
                            // libjxl uses [0.58, 0.62, 0.62, 0.62] for X/B channels
                            // (different from libjxl-tiny's per-channel adjustments)
                            let thresholds_xb = Self::default_thresholds(c, covered_x, covered_y);
                            let weights = super::quant::quant_weights(raw_strategy as usize, c);
                            let zigzag = if self.error_diffusion {
                                zigzag_cache
                                    .iter()
                                    .find(|(cx2, cy2, _)| *cx2 == cx && *cy2 == cy)
                                    .map(|(_, _, v)| v.as_slice())
                            } else {
                                None
                            };
                            Self::quantize_ac_block(
                                &dct_coeffs[c],
                                weights,
                                qac,
                                qm_multiplier,
                                &thresholds_xb,
                                block_width,
                                block_height,
                                covered_x,
                                covered_y,
                                covered_blocks,
                                size,
                                raw_strategy,
                                bx - xoff,
                                by - yoff,
                                &mut quant_ac[c],
                                width,
                                self.error_diffusion,
                                zigzag,
                                if self.error_diffusion {
                                    Some(error_scratch)
                                } else {
                                    None
                                },
                                quant_flat_scratch,
                            );
                        }

                        // ── Step 8: Count non-zeros for all 3 channels ─────────────
                        let transpose_slots = covered_y > covered_x;
                        for c in 0..3 {
                            if covered_blocks == 1 {
                                num_nonzero_8x8_except_dc(
                                    &quant_ac[c][(by - yoff) * width + (bx - xoff)],
                                    &mut nzeros[c][(by - yoff) * width + (bx - xoff)],
                                );
                                raw_nzeros[c][(by - yoff) * width + (bx - xoff)] =
                                    nzeros[c][(by - yoff) * width + (bx - xoff)] as u16;
                            } else {
                                // Build flat block in cx*8 × cy*8 layout (stride = cx*8).
                                // num_nonzero_except_llf expects block[y * stride + x] for y,x in 0..cy*8, 0..cx*8.
                                // The 8x8 block storage uses quant_ac[(ly * width + lx)][pos_in_8x8].
                                let stride = cx * BLOCK_DIM;
                                let full_block = &mut nz_full_block_scratch[..size];
                                // Nested loops eliminate per-element integer divisions.
                                // Pre-slice full_block rows to eliminate inner bounds checks.
                                for coef_slot_y in 0..cy {
                                    for pos_y in 0..BLOCK_DIM {
                                        let y = coef_slot_y * BLOCK_DIM + pos_y;
                                        let fb_row =
                                            &mut full_block[y * stride..y * stride + stride];
                                        for coef_slot_x in 0..cx {
                                            let (phys_row_off, phys_col_off) = if transpose_slots {
                                                (coef_slot_x, coef_slot_y)
                                            } else {
                                                (coef_slot_y, coef_slot_x)
                                            };
                                            let flat_idx = (by - yoff + phys_row_off) * width
                                                + (bx - xoff + phys_col_off);
                                            let row = &quant_ac[c][flat_idx];
                                            for pos_x in 0..BLOCK_DIM {
                                                let x = coef_slot_x * BLOCK_DIM + pos_x;
                                                fb_row[x] = row[pos_y * BLOCK_DIM + pos_x];
                                            }
                                        }
                                    }
                                }
                                let flat_len = (covered_y - 1) * width + covered_x;
                                let flat_nz = &mut nz_flat_scratch[..flat_len];
                                flat_nz.fill(0);
                                let raw_nz = num_nonzero_except_llf(
                                    cx, cy, full_block, width, flat_nz, covered_x, covered_y,
                                );
                                for dy in 0..covered_y {
                                    for dx in 0..covered_x {
                                        nzeros[c][(by - yoff + dy) * width + (bx - xoff + dx)] =
                                            flat_nz[dx + dy * width];
                                    }
                                }
                                raw_nzeros[c][(by - yoff) * width + (bx - xoff)] = raw_nz;
                            }
                        }
                    }
                }
            },
        );
    }

    /// Perform DCT and quantization on all blocks (parallel over groups).
    ///
    /// Supports all AC strategies. For multi-block transforms, only first blocks
    /// are processed; sub-block slots store their portion of the coefficients.
    ///
    /// Processing order per block matches C++ WriteACGroup:
    /// 1. DCT Y → extract Y DC → quantize Y AC (with thresholding)
    /// 2. Dequantize Y AC back (AdjustQuantBias) → roundtripped Y
    /// 3. DCT X, B → apply CfL using roundtripped Y → extract X/B DC
    /// 4. Quantize X/B AC (with thresholding + x_qm_mul for X)
    ///
    /// Groups are processed in parallel; results are scattered into the output.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(crate) fn transform_and_quantize(
        &self,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        padded_width: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        params: &DistanceParams,
        quant_field: &mut [u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
    ) -> TransformOutput {
        let mut out = TransformOutput::new(xsize_blocks, ysize_blocks);

        if crate::jxl_encode::parallel::sequential_maps_forced() {
            self.transform_and_quantize_into(
                xyb_x,
                xyb_y,
                xyb_b,
                padded_width,
                xsize_blocks,
                ysize_blocks,
                params,
                quant_field,
                cfl_map,
                ac_strategy,
                &mut out,
            );
            return out;
        }

        let xsize_groups = div_ceil(xsize_blocks, GROUP_DIM_IN_BLOCKS);
        let ysize_groups = div_ceil(ysize_blocks, GROUP_DIM_IN_BLOCKS);
        let num_groups = xsize_groups * ysize_groups;

        let group_results = crate::jxl_encode::parallel::parallel_map(num_groups, |group_idx| {
            let gy = group_idx / xsize_groups;
            let gx = group_idx % xsize_groups;
            let start_bx = gx * GROUP_DIM_IN_BLOCKS;
            let start_by = gy * GROUP_DIM_IN_BLOCKS;
            let end_bx = (start_bx + GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
            let end_by = (start_by + GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
            let width = end_bx - start_bx;
            let height = end_by - start_by;

            let mut result = GroupTransformResult::new(start_bx, start_by, width, height);
            self.transform_blocks_into(
                xyb_x,
                xyb_y,
                xyb_b,
                padded_width,
                xsize_blocks,
                params,
                quant_field,
                cfl_map,
                ac_strategy,
                start_by,
                end_by,
                start_bx,
                end_bx,
                &mut result,
            );
            result
        });

        // Scatter group results into TransformOutput.
        // Groups cover non-overlapping block-row/column ranges, so writes to
        // quant_dc/quant_ac/nzeros/raw_nzeros/float_dc and quant_field are
        // disjoint across groups and can be done in parallel.
        #[cfg(feature = "parallel")]
        if !crate::jxl_encode::parallel::sequential_maps_forced() {
            use rayon::prelude::*;
            // SAFETY: each group writes to a disjoint (start_by..end_by, start_bx..end_bx)
            // region of every output array.  Row-indexed arrays (quant_dc, quant_ac, …)
            // are accessed only at row `start_by + ly`, which is unique per group.
            // flat_dc is indexed by `gy * xsize_blocks + bx` — different gy per group.
            // quant_field indices are `by * xsize_blocks + bx` — different (by, bx) per group.
            // No two threads can access the same memory location.
            #[allow(unsafe_code)]
            {
                let out_ptr = &mut out as *mut TransformOutput as usize;
                let qf_ptr = quant_field.as_mut_ptr() as usize;
                let qf_len = quant_field.len();
                group_results.into_par_iter().for_each(|result| {
                    // SAFETY: each group writes to a disjoint (start_by..end_by, start_bx..end_bx)
                    // region of every output array — no two tasks touch the same memory location.
                    // NOTE: creating multiple &mut TransformOutput from the same raw pointer is
                    // formally UB under Rust's aliasing rules even when byte ranges are disjoint.
                    // A sound fix requires UnsafeCell<TransformOutput>, but that is a large
                    // refactor. This pattern works correctly on today's LLVM backend (no
                    // alias-based reordering applied across copy_from_slice boundaries), and
                    // is bounded by the #[allow(unsafe_code)] scope here.
                    let qf = unsafe { std::slice::from_raw_parts_mut(qf_ptr as *mut u8, qf_len) };
                    for &(idx, val) in &result.quant_adjustments {
                        qf[idx] = val;
                    }
                    GroupTransformResult::scatter_into_raw(
                        &result,
                        unsafe { &mut *(out_ptr as *mut TransformOutput) },
                        xsize_blocks,
                    );
                });
            }
        } else {
            for result in group_results {
                for &(idx, val) in &result.quant_adjustments {
                    quant_field[idx] = val;
                }
                result.scatter_into(&mut out, xsize_blocks);
            }
        }
        #[cfg(not(feature = "parallel"))]
        for result in group_results {
            for &(idx, val) in &result.quant_adjustments {
                quant_field[idx] = val;
            }
            result.scatter_into(&mut out, xsize_blocks);
        }

        out
    }

    /// Fill pre-allocated `TransformOutput` buffers (sequential, for butteraugli loop).
    ///
    /// Processes groups sequentially via `transform_blocks_into` and scatters
    /// results into the pre-allocated output. Used by the butteraugli quantization
    /// loop which reuses the same `TransformOutput` across iterations.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn transform_and_quantize_into(
        &self,
        xyb_x: &[f32],
        xyb_y: &[f32],
        xyb_b: &[f32],
        padded_width: usize,
        xsize_blocks: usize,
        ysize_blocks: usize,
        params: &DistanceParams,
        quant_field: &mut [u8],
        cfl_map: &CflMap,
        ac_strategy: &AcStrategyMap,
        out: &mut TransformOutput,
    ) {
        let xsize_groups = div_ceil(xsize_blocks, GROUP_DIM_IN_BLOCKS);
        let ysize_groups = div_ceil(ysize_blocks, GROUP_DIM_IN_BLOCKS);

        for gy in 0..ysize_groups {
            for gx in 0..xsize_groups {
                let start_bx = gx * GROUP_DIM_IN_BLOCKS;
                let start_by = gy * GROUP_DIM_IN_BLOCKS;
                let end_bx = (start_bx + GROUP_DIM_IN_BLOCKS).min(xsize_blocks);
                let end_by = (start_by + GROUP_DIM_IN_BLOCKS).min(ysize_blocks);
                let width = end_bx - start_bx;
                let height = end_by - start_by;

                let mut result = GroupTransformResult::new(start_bx, start_by, width, height);
                self.transform_blocks_into(
                    xyb_x,
                    xyb_y,
                    xyb_b,
                    padded_width,
                    xsize_blocks,
                    params,
                    quant_field,
                    cfl_map,
                    ac_strategy,
                    start_by,
                    end_by,
                    start_bx,
                    end_bx,
                    &mut result,
                );
                // Apply quant adjustments immediately so later groups see them.
                for &(idx, val) in &result.quant_adjustments {
                    quant_field[idx] = val;
                }
                result.scatter_into(out, xsize_blocks);
            }
        }
    }
}
