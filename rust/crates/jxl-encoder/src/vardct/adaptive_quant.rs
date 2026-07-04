// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Adaptive quantization field computation.
//!
//! Ported from full libjxl `enc_adaptive_quantization.cc`.

// Ported float constants from C++ - exact values are intentional for parity.
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
//! Computes per-block quantization values based on perceptual masking.
//!
//! Pipeline (matches full libjxl):
//! 1. `compute_pre_erosion()` — Y channel diffs, gamma ratio, limit clamp, masking sqrt, 4x downsample
//! 2. `fuzzy_erosion()` — 3×3 min-4 distance-weighted sum, 2x downsample
//! 3. `per_block_modulations()` — ComputeMask → GammaModulation → HfModulation → Min(Hf,Blue) → exp2
//! 4. Convert: `raw_quant = clamp(round(quant_field * inv_scale + 0.5), 1, 255)`

use super::common::clamp;

// Fast math helpers and masking sub-functions have been migrated to jxl_simd.
// compute_pre_erosion and per_block_modulations now delegate to jxl_simd SIMD implementations.

/// Insert `v` into the smallest-4 tracking variables if it's smaller than `min3`.
/// Used by `fuzzy_erosion()` (which remains local — operates on small downsampled data).
#[inline(always)]
fn store_min4(v: f32, min0: &mut f32, min1: &mut f32, min2: &mut f32, min3: &mut f32) {
    if v < *min3 {
        if v < *min0 {
            *min3 = *min2;
            *min2 = *min1;
            *min1 = *min0;
            *min0 = v;
        } else if v < *min1 {
            *min3 = *min2;
            *min2 = *min1;
            *min1 = v;
        } else if v < *min2 {
            *min3 = *min2;
            *min2 = v;
        } else {
            *min3 = v;
        }
    }
}

/// Compute pre-erosion map from XYB planes.
///
/// Full libjxl version: Y channel only (no X channel), with limit=0.2 clamp
/// before MaskingSqrt. SIMD-accelerated via jxl_simd.
///
/// Output dimensions: ceil(tile_pixel_w / 4) × ceil(tile_pixel_h / 4).
#[allow(clippy::too_many_arguments)]
fn compute_pre_erosion(
    xyb_y: &[f32],
    width: usize,
    height: usize,
    tile_x0: usize,
    tile_y0: usize,
    tile_x1: usize,
    tile_y1: usize,
) -> (Vec<f32>, usize, usize) {
    jxl_simd::compute_pre_erosion(xyb_y, width, height, tile_x0, tile_y0, tile_x1, tile_y1)
}

/// FuzzyErosion: 3×3 min-4 weighted sum, then 2x downsample.
/// Full libjxl version: distance-dependent weights.
///
/// Output pixels are independent across (ox, oy) pairs, so rows are
/// parallelised with rayon when the `parallel` feature is enabled.
#[allow(clippy::too_many_arguments)]
fn fuzzy_erosion(
    from: &[f32],
    from_w: usize,
    from_h: usize,
    from_x0: usize,
    from_y0: usize,
    region_w: usize,
    region_h: usize,
    butteraugli_target: f32,
) -> (Vec<f32>, usize, usize) {
    let out_w = region_w / 2;
    let out_h = region_h / 2;
    let mut out = vec![0.0_f32; out_w * out_h];

    // Distance-dependent weights (full libjxl)
    const K_MUL_BASE: [f32; 4] = [0.125, 0.1, 0.09, 0.06];
    const K_MUL_ADD: [f32; 4] = [0.0, -0.1, -0.09, -0.06];

    let mul = if butteraugli_target < 2.0 {
        (2.0 - butteraugli_target) * 0.5
    } else {
        0.0
    };

    let mut k_mul = [0.0_f32; 4];
    let mut norm_sum = 0.0_f32;
    for (ii, k) in k_mul.iter_mut().enumerate() {
        *k = K_MUL_BASE[ii] + mul * K_MUL_ADD[ii];
        norm_sum += *k;
    }
    const K_TOTAL: f32 = 0.29959705784054957;
    for k in &mut k_mul {
        *k *= K_TOTAL / norm_sum;
    }

    // Helper: compute the min-4 weighted value for one input pixel at (x, y).
    let compute_pixel = |x: usize, y: usize| -> f32 {
        let ym1 = if y >= 1 { y - 1 } else { y };
        let yp1 = if y + 1 < from_h { y + 1 } else { y };
        let xm1 = if x >= 1 { x - 1 } else { x };
        let xp1 = if x + 1 < from_w { x + 1 } else { x };

        let center = from[y * from_w + x];
        let left = from[y * from_w + xm1];
        let right = from[y * from_w + xp1];
        let top_left = from[ym1 * from_w + xm1];
        let top = from[ym1 * from_w + x];
        let top_right = from[ym1 * from_w + xp1];
        let bot_left = from[yp1 * from_w + xm1];
        let bot = from[yp1 * from_w + x];
        let bot_right = from[yp1 * from_w + xp1];

        let mut min0 = center;
        let mut min1 = left;
        let mut min2 = right;
        let mut min3 = top_left;
        if min0 > min1 {
            core::mem::swap(&mut min0, &mut min1);
        }
        if min0 > min2 {
            core::mem::swap(&mut min0, &mut min2);
        }
        if min0 > min3 {
            core::mem::swap(&mut min0, &mut min3);
        }
        if min1 > min2 {
            core::mem::swap(&mut min1, &mut min2);
        }
        if min1 > min3 {
            core::mem::swap(&mut min1, &mut min3);
        }
        if min2 > min3 {
            core::mem::swap(&mut min2, &mut min3);
        }
        store_min4(top, &mut min0, &mut min1, &mut min2, &mut min3);
        store_min4(top_right, &mut min0, &mut min1, &mut min2, &mut min3);
        store_min4(bot_left, &mut min0, &mut min1, &mut min2, &mut min3);
        store_min4(bot, &mut min0, &mut min1, &mut min2, &mut min3);
        store_min4(bot_right, &mut min0, &mut min1, &mut min2, &mut min3);

        k_mul[0] * min0 + k_mul[1] * min1 + k_mul[2] * min2 + k_mul[3] * min3
    };

    // Each output pixel at (ox, oy) accumulates values from the 2×2 input block
    // at (from_x0 + 2*ox + dx, from_y0 + 2*oy + dy).  Output rows are independent.
    #[cfg(feature = "parallel")]
    if !crate::parallel::sequential_maps_forced() {
        use rayon::prelude::*;
        out.par_chunks_mut(out_w)
            .enumerate()
            .for_each(|(oy, out_row)| {
                for ox in 0..out_w {
                    let mut acc = 0.0f32;
                    for dy in 0..2usize {
                        let fy = oy * 2 + dy;
                        if fy >= region_h {
                            continue;
                        }
                        let y = fy + from_y0;
                        for dx in 0..2usize {
                            let fx = ox * 2 + dx;
                            if fx >= region_w {
                                continue;
                            }
                            let x = fx + from_x0;
                            acc += compute_pixel(x, y);
                        }
                    }
                    out_row[ox] = acc;
                }
            });
        return (out, out_w, out_h);
    }

    // Sequential path (parallel feature disabled or forced sequential).
    for fy in 0..region_h {
        let oy = fy / 2;
        // Guard: when region_h is odd, the last row's oy equals out_h (out of bounds).
        if oy >= out_h {
            continue;
        }
        let y = fy + from_y0;
        for fx in 0..region_w {
            let x = fx + from_x0;
            let v = compute_pixel(x, y);
            let ox = fx / 2;
            if fx % 2 == 0 && fy % 2 == 0 {
                out[oy * out_w + ox] = v;
            } else {
                out[oy * out_w + ox] += v;
            }
        }
    }

    (out, out_w, out_h)
}

/// ComputeMaskForAcStrategyUse: simple masking hack.
fn compute_mask_for_ac_strategy_use(out_val: f32) -> f32 {
    const K_MUL: f32 = 1.0;
    const K_OFFSET: f32 = 0.001;
    K_MUL / (out_val + K_OFFSET)
}

/// Compute per-pixel (1x1) masking field for pixel-domain loss calculation.
///
/// This implements libjxl's 1x1 Laplacian masking from `enc_adaptive_quantization.cc`.
/// The mask is used in `EstimateEntropy` to weight pixel-domain quantization error.
///
/// # Returns
/// Per-pixel mask field of size `width * height`, row-major layout.
/// After computing the raw mask, applies libjxl's Symmetric5 blur.
pub fn compute_mask1x1(xyb_y: &[f32], width: usize, height: usize) -> Vec<f32> {
    let n = width * height;

    // SIMD-accelerated per-pixel masking (neighbor avg → gamma ratio → log1p → reciprocal).
    // Step 1 uses ±1 row neighbors; each strip includes 1 border row top/bottom.
    let mut mask1x1 = vec![0.0_f32; n];
    jxl_simd::compute_mask1x1(xyb_y, width, height, &mut mask1x1);

    // Apply Symmetric5 blur using SIMD gaborish kernel with mask1x1 weights.
    // The gaborish_5x5_channel kernel has the same 5x5 weight pattern:
    //   D  L  R  L  D
    //   L  d  r  d  L
    //   R  r  c  r  R
    //   L  d  r  d  L
    //   D  L  R  L  D
    // libjxl mask1x1 weights from enc_adaptive_quantization.cc:
    const W_R: f32 = 0.364_911_248; // kFilterMask1x1[0] = r (orthogonal dist 1)
    const W_D: f32 = 0.05; // kFilterMask1x1[1] = d (diagonal dist 1)
    const W_R2: f32 = 0.168_888_802_1; // kFilterMask1x1[2] = R (orthogonal dist 2)
    const W_L: f32 = 0.221_069_183; // kFilterMask1x1[3] = L (knight's move)
    const W_D2: f32 = 0.306_563_504; // kFilterMask1x1[4] = D (diagonal dist 2)
    let sum = 1.0 + 4.0 * (W_R + W_D + W_R2 + W_D2 + 2.0 * W_L);
    let inv_sum = 1.0 / sum;
    let wc = inv_sum;
    let wr = inv_sum * W_R;
    let wd = inv_sum * W_D;
    let wr2 = inv_sum * W_R2;
    let wl = inv_sum * W_L;
    let wd2 = inv_sum * W_D2;

    // Parallel strip-based gaborish blur: each strip runs gaborish_5x5_channel
    // independently. We include ±2 border rows so interior pixels of each strip
    // see correct 5-row neighborhoods. Border rows' outputs are discarded.
    //
    // This avoids the 56ms sequential 5×5 convolution cost for large images
    // by exploiting all available rayon threads (each strip is independent).
    //
    // Thread-local buffers (MASK_SNAP, STRIP_BUF, SCRATCH_BUF) avoid fresh
    // allocations and page faults on repeated calls — buffers grow on first use
    // but are reused across all subsequent images on the same rayon thread.
    #[cfg(feature = "parallel")]
    {
        if !crate::parallel::sequential_maps_forced() {
            use rayon::prelude::*;
            const STRIP_ROWS: usize = 128; // ~22 strips for 3000 rows → full core utilisation
            const BORDER: usize = 2; // 5×5 kernel reaches ±2 rows

            // Snapshot the current mask1x1 on the calling thread so rayon workers
            // can read the original (unmodified) data while we write results back.
            // The thread_local grows but never shrinks → no repeated page faults.
            thread_local! {
                static MASK_SNAP: std::cell::RefCell<Vec<f32>> =
                    const { std::cell::RefCell::new(Vec::new()) };
            }
            let snap_ptr: usize = MASK_SNAP.with(|snap| {
                let mut snap = snap.borrow_mut();
                if snap.len() < n {
                    snap.resize(n, 0.0);
                }
                snap[..n].copy_from_slice(&mask1x1);
                snap.as_ptr() as usize
            });

            // Write results directly back to mask1x1 (avoids a separate "blurred" Vec).
            let out_ptr: usize = mask1x1.as_mut_ptr() as usize;
            let num_strips = (height + STRIP_ROWS - 1) / STRIP_ROWS;

            (0..num_strips).into_par_iter().for_each(|si| {
                thread_local! {
                    static STRIP_BUF: std::cell::RefCell<Vec<f32>> =
                        const { std::cell::RefCell::new(Vec::new()) };
                    static SCRATCH_BUF: std::cell::RefCell<Vec<f32>> =
                        const { std::cell::RefCell::new(Vec::new()) };
                }

                let y0 = si * STRIP_ROWS;
                let y1 = (y0 + STRIP_ROWS).min(height);
                let rows = y1 - y0;

                let in_y0 = y0.saturating_sub(BORDER);
                let in_y1 = (y1 + BORDER).min(height);
                let in_h = in_y1 - in_y0;
                let border_top = y0 - in_y0;
                let ext_size = in_h * width;

                STRIP_BUF.with(|strip_cell| {
                    let mut strip = strip_cell.borrow_mut();
                    SCRATCH_BUF.with(|scratch_cell| {
                        let mut scratch = scratch_cell.borrow_mut();

                        if strip.len() < ext_size {
                            strip.resize(ext_size, 0.0);
                        }
                        if scratch.len() < ext_size {
                            scratch.resize(ext_size, 0.0);
                        }

                        // Copy extended input region from snapshot.
                        // SAFETY: snap_ptr points to the calling thread's MASK_SNAP data,
                        // which lives for the duration of this parallel closure (held by
                        // the calling thread's borrow).  Rayon workers only read from it.
                        #[allow(unsafe_code)]
                        {
                            let snap: &[f32] =
                                unsafe { std::slice::from_raw_parts(snap_ptr as *const f32, n) };
                            strip[..ext_size].copy_from_slice(&snap[in_y0 * width..in_y1 * width]);
                        }

                        jxl_simd::gaborish_5x5_channel(
                            &mut strip[..ext_size],
                            &mut scratch[..ext_size],
                            width,
                            in_h,
                            wc,
                            wr,
                            wd,
                            wr2,
                            wl,
                            wd2,
                        );

                        // Write valid rows back to mask1x1.
                        // SAFETY: each si covers a unique non-overlapping row range
                        // y0..y1 of mask1x1; no two strips write to the same memory.
                        let valid_size = rows * width;
                        let src = border_top * width;
                        #[allow(unsafe_code)]
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                strip[src..].as_ptr(),
                                (out_ptr as *mut f32).add(y0 * width),
                                valid_size,
                            );
                        }
                    });
                });
            });
            return mask1x1;
        }
    }

    // Sequential fallback
    let mut scratch = vec![0.0_f32; n];
    jxl_simd::gaborish_5x5_channel(
        &mut mask1x1,
        &mut scratch,
        width,
        height,
        wc,
        wr,
        wd,
        wr2,
        wl,
        wd2,
    );
    mask1x1
}

// symmetric5_blur_mask1x1 replaced by jxl_simd::gaborish_5x5_channel with
// mask1x1-specific weights (same 5x5 kernel structure, ~10x faster via AVX2).

/// PerBlockModulations: apply all modulations and convert exponent to multiplier.
/// SIMD-accelerated via jxl_simd.
///
/// Full libjxl order: ComputeMask → GammaModulation → HfModulation → Min(Hf, BlueModulation) → exp2
///
/// `stride` is the row stride (padded width) of the XYB buffers.
#[allow(clippy::too_many_arguments)]
fn per_block_modulations(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    stride: usize,
    butteraugli_target: f32,
    scale: f32,
    rect_x0_blocks: usize,
    rect_y0_blocks: usize,
    rect_w_blocks: usize,
    rect_h_blocks: usize,
    aq_map: &mut [f32],
    aq_map_w: usize,
) {
    jxl_simd::per_block_modulations(
        xyb_x,
        xyb_y,
        xyb_b,
        stride,
        butteraugli_target,
        scale,
        rect_x0_blocks,
        rect_y0_blocks,
        rect_w_blocks,
        rect_h_blocks,
        aq_map,
        aq_map_w,
    );
}

/// Compute the adaptive quantization field for the entire image.
///
/// Returns `(quant_field_float, masking)`.
#[allow(clippy::too_many_arguments)]
pub fn compute_quant_field_float(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    distance: f32,
    k_ac_quant: f32,
) -> (Vec<f32>, Vec<f32>) {
    let scale = k_ac_quant / distance;

    let tile_x0_pixels = 0;
    let tile_y0_pixels = 0;

    // Step 1: Compute pre-erosion (Y only, limit clamp, 4x downsample)
    let (pre_erosion, pre_erosion_w, pre_erosion_h) = compute_pre_erosion(
        xyb_y,
        width,
        height,
        tile_x0_pixels,
        tile_y0_pixels,
        width,
        height,
    );

    // Step 2: Fuzzy erosion (distance-dependent weights, 2x downsample)
    let from_x0 = if tile_x0_pixels > 0 { 1 } else { 0 };
    let from_y0 = if tile_y0_pixels > 0 { 1 } else { 0 };
    let erosion_region_w = (xsize_blocks * 2).min(pre_erosion_w.saturating_sub(from_x0));
    let erosion_region_h = (ysize_blocks * 2).min(pre_erosion_h.saturating_sub(from_y0));

    let (mut aq_map, aq_map_w, _aq_map_h) = fuzzy_erosion(
        &pre_erosion,
        pre_erosion_w,
        pre_erosion_h,
        from_x0,
        from_y0,
        erosion_region_w,
        erosion_region_h,
        distance,
    );

    // Step 2.5: Compute masking field for AC strategy use
    let mut masking = vec![0.0f32; xsize_blocks * ysize_blocks];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            masking[by * xsize_blocks + bx] =
                compute_mask_for_ac_strategy_use(aq_map[by * aq_map_w + bx]);
        }
    }

    // Step 3: Per-block modulations (full libjxl order).
    // Parallelised by horizontal block-row strips: per_block_modulations uses
    // rect_y0_blocks to compute pixel-domain offsets into xyb_{x,y,b}, and writes
    // only to aq_map[0..chunk_h*aq_map_w], so disjoint chunks are safe.
    {
        const CHUNK_ROWS: usize = 32;
        crate::parallel::parallel_chunks_mut(&mut aq_map, aq_map_w * CHUNK_ROWS, |ci, chunk| {
            let chunk_y0 = ci * CHUNK_ROWS;
            let chunk_h = chunk.len() / aq_map_w;
            per_block_modulations(
                xyb_x,
                xyb_y,
                xyb_b,
                width,
                distance,
                scale,
                0,
                chunk_y0,
                xsize_blocks,
                chunk_h,
                chunk,
                aq_map_w,
            );
        });
    }

    // Step 4: Extract compact float quant field
    let mut quant_field_float = vec![0.0f32; xsize_blocks * ysize_blocks];
    for by in 0..ysize_blocks {
        for bx in 0..xsize_blocks {
            quant_field_float[by * xsize_blocks + bx] = aq_map[by * aq_map_w + bx];
        }
    }

    (quant_field_float, masking)
}

/// Convert float quant field to u8 raw_quant values.
///
/// Matches libjxl's ClampVal: `static_cast<int32_t>(clamp(qf * inv_scale + 0.5, 1.0, 256.0))`
/// which is standard round-to-nearest via add 0.5 then truncate.
pub fn quantize_quant_field(quant_field_float: &[f32], inv_scale: f32) -> Vec<u8> {
    quant_field_float
        .iter()
        .map(|&qf| {
            let val = (qf * inv_scale + 0.5) as i32;
            clamp(val, 1, 255) as u8
        })
        .collect()
}

/// Convenience wrapper that calls `compute_quant_field_float()` then
/// `quantize_quant_field()`.
#[allow(clippy::too_many_arguments, dead_code)]
pub fn compute_adaptive_quant_field(
    xyb_x: &[f32],
    xyb_y: &[f32],
    xyb_b: &[f32],
    width: usize,
    height: usize,
    xsize_blocks: usize,
    ysize_blocks: usize,
    distance: f32,
    inv_scale: f32,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let (quant_field_float, masking) = compute_quant_field_float(
        xyb_x,
        xyb_y,
        xyb_b,
        width,
        height,
        xsize_blocks,
        ysize_blocks,
        distance,
        0.765, // K_AC_QUANT default
    );
    let raw_quant_field = quantize_quant_field(&quant_field_float, inv_scale);
    (raw_quant_field, masking, quant_field_float)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Scalar math unit tests (fast_log2f, fast_pow2f, masking_sqrt, ratio_of_derivatives,
    // compute_mask) migrated to jxl_simd::adaptive_quant::tests.

    #[test]
    fn test_store_min4() {
        let mut min0 = 5.0_f32;
        let mut min1 = 6.0;
        let mut min2 = 7.0;
        let mut min3 = 8.0;

        store_min4(3.0, &mut min0, &mut min1, &mut min2, &mut min3);
        assert_eq!(min0, 3.0);
        assert_eq!(min1, 5.0);
        assert_eq!(min2, 6.0);
        assert_eq!(min3, 7.0);

        store_min4(100.0, &mut min0, &mut min1, &mut min2, &mut min3);
        assert_eq!(min3, 7.0);
    }

    #[test]
    fn test_adaptive_quant_field_uniform() {
        let w = 16;
        let h = 16;
        let n = w * h;
        let xyb_x = vec![0.0_f32; n];
        let xyb_y = vec![0.5_f32; n];
        let xyb_b = vec![0.5_f32; n];

        let xb = w / 8;
        let yb = h / 8;

        let (result, masking, _quant_float) =
            compute_adaptive_quant_field(&xyb_x, &xyb_y, &xyb_b, w, h, xb, yb, 1.0, 8.93);

        assert_eq!(result.len(), xb * yb);
        assert_eq!(masking.len(), xb * yb);
        for &v in &result {
            assert!(v >= 1, "quant value {} out of range", v);
        }
        let first = result[0];
        for &v in &result {
            assert_eq!(v, first, "uniform image should produce uniform quant field");
        }
    }

    #[test]
    fn test_adaptive_quant_field_varying() {
        let w = 32;
        let h = 32;
        let n = w * h;
        let mut xyb_x = vec![0.0_f32; n];
        let mut xyb_y = vec![0.0_f32; n];
        let mut xyb_b = vec![0.0_f32; n];

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if x < w / 2 {
                    xyb_y[idx] = 0.5;
                    xyb_b[idx] = 0.5;
                } else {
                    xyb_y[idx] = if (x + y) % 2 == 0 { 0.8 } else { 0.2 };
                    xyb_b[idx] = xyb_y[idx];
                    xyb_x[idx] = if x % 2 == 0 { 0.1 } else { -0.1 };
                }
            }
        }

        let xb = w / 8;
        let yb = h / 8;

        let (result, _masking, _quant_float) =
            compute_adaptive_quant_field(&xyb_x, &xyb_y, &xyb_b, w, h, xb, yb, 1.0, 8.93);

        assert_eq!(result.len(), xb * yb);
        for &v in &result {
            assert!(v >= 1, "quant value {} out of range", v);
        }
        let left_avg: f32 = (0..yb).map(|by| result[by * xb] as f32).sum::<f32>() / yb as f32;
        let right_avg: f32 = (0..yb)
            .map(|by| result[by * xb + xb - 1] as f32)
            .sum::<f32>()
            / yb as f32;
        assert!(
            (left_avg - right_avg).abs() > 0.01,
            "smooth vs textured should differ: left={}, right={}",
            left_avg,
            right_avg
        );
    }

    #[test]
    fn test_adaptive_quant_field_non_multiple_of_8() {
        for &(w, h) in &[
            (300usize, 300usize),
            (301, 301),
            (100, 100),
            (17, 17),
            (9, 9),
            (15, 33),
            (257, 129),
        ] {
            let xb = w.div_ceil(8);
            let yb = h.div_ceil(8);
            let pw = xb * 8;
            let ph = yb * 8;
            let n = pw * ph;
            let xyb_x = vec![0.0_f32; n];
            let xyb_y = vec![0.5_f32; n];
            let xyb_b = vec![0.5_f32; n];

            let (result, _masking, _quant_float) =
                compute_adaptive_quant_field(&xyb_x, &xyb_y, &xyb_b, pw, ph, xb, yb, 1.0, 8.93);

            assert_eq!(
                result.len(),
                xb * yb,
                "wrong length for {}x{}: got {}, expected {}",
                w,
                h,
                result.len(),
                xb * yb
            );
            for &v in &result {
                assert!(v >= 1, "quant value {} out of range for {}x{}", v, w, h);
            }
        }
    }

    #[test]
    fn test_compute_mask1x1_uniform() {
        let w = 16;
        let h = 16;
        let xyb_y = vec![0.5_f32; w * h];

        let mask = compute_mask1x1(&xyb_y, w, h);

        assert_eq!(mask.len(), w * h);
        for &v in &mask {
            assert!(v > 0.0 && v.is_finite(), "mask value {} invalid", v);
        }
        let first = mask[w + 1];
        assert!(first > 50.0, "uniform mask should be high, got {}", first);
    }

    #[test]
    fn test_compute_mask1x1_edges() {
        let w = 16;
        let h = 16;
        let mut xyb_y = vec![0.2_f32; w * h];

        for y in 0..h {
            for x in 8..w {
                xyb_y[y * w + x] = 0.8;
            }
        }

        let mask = compute_mask1x1(&xyb_y, w, h);

        let interior_left = mask[4 * w + 4];
        let at_edge = mask[8 * w + 8];

        assert!(
            at_edge < interior_left,
            "edge mask {} should be < interior mask {}",
            at_edge,
            interior_left
        );
    }
}
