use std::arch::is_x86_feature_detected;
use std::arch::x86_64::*;

use crate::jxl_decode::jxl_grid::{MutableSubgrid, SharedSubgrid};
use crate::jxl_decode::jxl_modular::ChannelShift;
use crate::jxl_decode::jxl_vardct::{BlockInfo, TransformType};

use crate::jxl_decode::jxl_render::vardct::{
    dct_common::DctDirection, transform_common::transform_varblocks_inner,
};

use super::generic;

#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "sse3")]
unsafe fn transform_dct2_x86_64_sse41(coeff: &mut MutableSubgrid<'_>) {
    generic::aux_idct2_in_place_2(coeff);
    generic::aux_idct2_in_place::<4>(coeff);
    generic::aux_idct2_in_place::<8>(coeff);
}

fn transform_dct4_x86_64_sse2(coeff: &mut MutableSubgrid<'_>) {
    generic::aux_idct2_in_place_2(coeff);

    unsafe {
        let mut scratch_0 = [_mm_setzero_ps(); 4];
        let mut scratch_1 = [_mm_setzero_ps(); 4];
        for y2 in 0..4 {
            let row_ptr = coeff.get_row(y2 * 2).as_ptr();
            let a = _mm_loadu_ps(row_ptr);
            let b = _mm_loadu_ps(row_ptr.add(4));
            scratch_0[y2] = _mm_shuffle_ps::<0b10001000>(a, b);
            scratch_1[y2] = _mm_shuffle_ps::<0b11011101>(a, b);
        }

        super::dct::transpose_lane(&mut scratch_0);
        super::dct::transpose_lane(&mut scratch_1);
        let mut scratch_0 = super::dct::dct4_inverse(scratch_0);
        let mut scratch_1 = super::dct::dct4_inverse(scratch_1);
        for y2 in 0..4 {
            let row_ptr = coeff.get_row_mut(y2).as_mut_ptr();
            _mm_storeu_ps(row_ptr, super::dct::dct4_vec_inverse(scratch_0[y2]));
            _mm_storeu_ps(row_ptr.add(4), super::dct::dct4_vec_inverse(scratch_1[y2]));

            let row_ptr = coeff.get_row(y2 * 2 + 1).as_ptr();
            let a = _mm_loadu_ps(row_ptr);
            let b = _mm_loadu_ps(row_ptr.add(4));
            scratch_0[y2] = _mm_shuffle_ps::<0b10001000>(a, b);
            scratch_1[y2] = _mm_shuffle_ps::<0b11011101>(a, b);
        }

        super::dct::transpose_lane(&mut scratch_0);
        super::dct::transpose_lane(&mut scratch_1);
        let scratch_0 = super::dct::dct4_inverse(scratch_0);
        let scratch_1 = super::dct::dct4_inverse(scratch_1);
        for y in 0..4 {
            let row_ptr = coeff.get_row_mut(y + 4).as_mut_ptr();
            _mm_storeu_ps(row_ptr, super::dct::dct4_vec_inverse(scratch_0[y]));
            _mm_storeu_ps(row_ptr.add(4), super::dct::dct4_vec_inverse(scratch_1[y]));
        }
    }
}

fn transform_dct4x8_x86_64_sse2<const TR: bool>(coeff: &mut MutableSubgrid<'_>) {
    let coeff0 = coeff.get(0, 0);
    let coeff1 = coeff.get(0, 1);
    *coeff.get_mut(0, 0) = coeff0 + coeff1;
    *coeff.get_mut(0, 1) = coeff0 - coeff1;

    unsafe {
        if TR {
            let mut scratch_0 = [_mm_setzero_ps(); 4];
            let mut scratch_1 = [_mm_setzero_ps(); 4];
            for y2 in 0..4 {
                let row_ptr = coeff.get_row(y2 * 2).as_ptr();
                let a = _mm_loadu_ps(row_ptr);
                let b = _mm_loadu_ps(row_ptr.add(4));
                let (l, r) = super::dct::dct8_vec_inverse(a, b);
                scratch_0[y2] = l;
                scratch_1[y2] = r;
            }

            let mut scratch_0 = super::dct::dct4_inverse(scratch_0);
            let mut scratch_1 = super::dct::dct4_inverse(scratch_1);
            super::dct::transpose_lane(&mut scratch_0);
            super::dct::transpose_lane(&mut scratch_1);
            for y2 in 0..4 {
                let y = [1, 5, 3, 7][y2];
                let row_ptr = coeff.get_row(y).as_ptr();
                let a = _mm_loadu_ps(row_ptr);
                let b = _mm_loadu_ps(row_ptr.add(4));
                let (l, r) = super::dct::dct8_vec_inverse(a, b);

                let row_ptr = coeff.get_row_mut(y2).as_mut_ptr();
                _mm_storeu_ps(row_ptr, scratch_0[y2]);
                let row_ptr = coeff.get_row_mut(y2 + 4).as_mut_ptr();
                _mm_storeu_ps(row_ptr, scratch_1[y2]);

                scratch_0[y2] = l;
                scratch_1[y2] = r;
            }
            scratch_0.swap(1, 2);
            scratch_1.swap(1, 2);

            let mut scratch_0 = super::dct::dct4_inverse(scratch_0);
            let mut scratch_1 = super::dct::dct4_inverse(scratch_1);
            super::dct::transpose_lane(&mut scratch_0);
            super::dct::transpose_lane(&mut scratch_1);
            for y in 0..4 {
                let row_ptr = coeff.get_row_mut(y).as_mut_ptr().add(4);
                _mm_storeu_ps(row_ptr, scratch_0[y]);
                let row_ptr = coeff.get_row_mut(y + 4).as_mut_ptr().add(4);
                _mm_storeu_ps(row_ptr, scratch_1[y]);
            }
        } else {
            let mut scratch_0 = [_mm_setzero_ps(); 4];
            let mut scratch_1 = [_mm_setzero_ps(); 4];
            for y2 in 0..4 {
                let row_ptr = coeff.get_row(y2 * 2).as_ptr();
                let a = _mm_loadu_ps(row_ptr);
                let b = _mm_loadu_ps(row_ptr.add(4));
                let (l, r) = super::dct::dct8_vec_inverse(a, b);
                scratch_0[y2] = l;
                scratch_1[y2] = r;
            }

            let mut scratch_0 = super::dct::dct4_inverse(scratch_0);
            let mut scratch_1 = super::dct::dct4_inverse(scratch_1);
            for y2 in 0..4 {
                let row_ptr = coeff.get_row_mut(y2).as_mut_ptr();
                _mm_storeu_ps(row_ptr, scratch_0[y2]);
                _mm_storeu_ps(row_ptr.add(4), scratch_1[y2]);

                let row_ptr = coeff.get_row(y2 * 2 + 1).as_ptr();
                let a = _mm_loadu_ps(row_ptr);
                let b = _mm_loadu_ps(row_ptr.add(4));
                let (l, r) = super::dct::dct8_vec_inverse(a, b);
                scratch_0[y2] = l;
                scratch_1[y2] = r;
            }

            let scratch_0 = super::dct::dct4_inverse(scratch_0);
            let scratch_1 = super::dct::dct4_inverse(scratch_1);
            for y in 0..4 {
                let row_ptr = coeff.get_row_mut(y + 4).as_mut_ptr();
                _mm_storeu_ps(row_ptr, scratch_0[y]);
                _mm_storeu_ps(row_ptr.add(4), scratch_1[y]);
            }
        }
    }
}

fn transform_dct(coeff: &mut MutableSubgrid<'_>) {
    super::dct::dct_2d_x86_64(coeff, DctDirection::Inverse);
}

/// Apply the inverse 8×8 DCT to a compact (stride=8) 64-float block in-place.
///
/// `block` must be exactly 64 elements. The 8×8 compact layout lets
/// `dct_2d_x86_64` dispatch to `dct8x8_avx2` with stride-1 `__m256` access —
/// all 64 floats stay in registers/L1 with no strided cache-line fetches.
pub fn compact_idct_8x8(block: &mut [f32; 64]) {
    // SAFETY: MutableSubgrid::from_buf only requires slice.len() >= width*height.
    let mut grid = MutableSubgrid::from_buf(block.as_mut_slice(), 8, 8, 8);
    super::dct::dct_2d_x86_64(&mut grid, DctDirection::Inverse);
}

/// Apply the inverse DCT to a compact (stride=block_w) block in-place.
/// Uses SSE4.1 when available, falls back to SSE2.
pub fn transform_single_block_compact(
    block: &mut [f32],
    block_w: usize,
    block_h: usize,
    dct_select: TransformType,
) {
    let mut grid = MutableSubgrid::from_buf(block, block_w, block_h, block_w);
    if is_x86_feature_detected!("sse4.1") {
        unsafe { transform_x86_64_sse41(&mut grid, dct_select); }
    } else {
        transform_x86_64_sse2(&mut grid, dct_select);
    }
}

#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "sse3")]
unsafe fn transform_x86_64_sse41(coeff: &mut MutableSubgrid<'_>, dct_select: TransformType) {
    use TransformType::*;

    match dct_select {
        Dct2 => transform_dct2_x86_64_sse41(coeff),
        Dct4 => transform_dct4_x86_64_sse2(coeff),
        Hornuss => generic::transform_hornuss(coeff),
        Dct4x8 => transform_dct4x8_x86_64_sse2::<false>(coeff),
        Dct8x4 => transform_dct4x8_x86_64_sse2::<true>(coeff),
        Afv0 => generic::transform_afv::<0>(coeff),
        Afv1 => generic::transform_afv::<1>(coeff),
        Afv2 => generic::transform_afv::<2>(coeff),
        Afv3 => generic::transform_afv::<3>(coeff),
        _ => transform_dct(coeff),
    }
}

fn transform_x86_64_sse2(coeff: &mut MutableSubgrid<'_>, dct_select: TransformType) {
    use TransformType::*;

    match dct_select {
        Dct2 => generic::transform_dct2(coeff),
        Dct4 => transform_dct4_x86_64_sse2(coeff),
        Hornuss => generic::transform_hornuss(coeff),
        Dct4x8 => transform_dct4x8_x86_64_sse2::<false>(coeff),
        Dct8x4 => transform_dct4x8_x86_64_sse2::<true>(coeff),
        Afv0 => generic::transform_afv::<0>(coeff),
        Afv1 => generic::transform_afv::<1>(coeff),
        Afv2 => generic::transform_afv::<2>(coeff),
        Afv3 => generic::transform_afv::<3>(coeff),
        _ => transform_dct(coeff),
    }
}

#[target_feature(enable = "sse4.1")]
#[target_feature(enable = "sse3")]
unsafe fn transform_varblocks_x86_64_sse41(
    lf: &[SharedSubgrid<f32>; 3],
    coeff_out: &mut [MutableSubgrid<'_, f32>; 3],
    shifts_cbycr: [ChannelShift; 3],
    block_info: &SharedSubgrid<BlockInfo>,
) {
    transform_varblocks_inner(
        lf,
        coeff_out,
        shifts_cbycr,
        block_info,
        super::dct::dct_2d_x86_64,
        transform_x86_64_sse41,
    );
}

pub fn transform_varblocks(
    lf: &[SharedSubgrid<f32>; 3],
    coeff_out: &mut [MutableSubgrid<'_, f32>; 3],
    shifts_cbycr: [ChannelShift; 3],
    block_info: &SharedSubgrid<BlockInfo>,
) {
    if is_x86_feature_detected!("sse4.1") {
        unsafe {
            return transform_varblocks_x86_64_sse41(lf, coeff_out, shifts_cbycr, block_info);
        }
    }

    unsafe {
        transform_varblocks_inner(
            lf,
            coeff_out,
            shifts_cbycr,
            block_info,
            super::dct::dct_2d_x86_64,
            transform_x86_64_sse2,
        );
    }
}
