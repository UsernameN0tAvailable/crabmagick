#![allow(unsafe_op_in_unsafe_fn)]
//! AVX2 (8-wide) port of the SSE2 2D DCT.
//!
//! This mirrors the `super::dct_2d_x86_64_sse2` lane algorithm exactly, but with
//! `LANE_SIZE = 8` / `Lane = __m256`, processing eight columns per SIMD vector
//! instead of four. The scalar butterflies are element-wise and therefore widen
//! without any change; only the in-register transpose becomes an 8x8 transpose.
//!
//! For the dominant 8×8 block size a specialized `dct8x8_avx2` fast path is used.
//! It keeps the full 512-byte working set in registers between the two DCT passes
//! and avoids the per-block heap allocation that `dct_2d_lane` requires.

use std::arch::x86_64::*;

use crate::jxl_decode::jxl_grid::{MutableSubgrid, SimdVector};
use crate::jxl_decode::jxl_render::vardct::dct_common::{self, DctDirection};
use crate::jxl_decode::jxl_render::vardct::generic;

const LANE_SIZE: usize = 8;
type Lane = __m256;

#[target_feature(enable = "avx2,fma")]
unsafe fn transpose_lane(lanes: &mut [Lane]) {
    let [r0, r1, r2, r3, r4, r5, r6, r7] = lanes else {
        panic!()
    };
    let (a0, a1, a2, a3, a4, a5, a6, a7) = (*r0, *r1, *r2, *r3, *r4, *r5, *r6, *r7);
    let t0 = _mm256_unpacklo_ps(a0, a1);
    let t1 = _mm256_unpackhi_ps(a0, a1);
    let t2 = _mm256_unpacklo_ps(a2, a3);
    let t3 = _mm256_unpackhi_ps(a2, a3);
    let t4 = _mm256_unpacklo_ps(a4, a5);
    let t5 = _mm256_unpackhi_ps(a4, a5);
    let t6 = _mm256_unpacklo_ps(a6, a7);
    let t7 = _mm256_unpackhi_ps(a6, a7);
    let u0 = _mm256_shuffle_ps::<0x44>(t0, t2);
    let u1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
    let u2 = _mm256_shuffle_ps::<0x44>(t1, t3);
    let u3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
    let u4 = _mm256_shuffle_ps::<0x44>(t4, t6);
    let u5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
    let u6 = _mm256_shuffle_ps::<0x44>(t5, t7);
    let u7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
    *r0 = _mm256_permute2f128_ps::<0x20>(u0, u4);
    *r1 = _mm256_permute2f128_ps::<0x20>(u1, u5);
    *r2 = _mm256_permute2f128_ps::<0x20>(u2, u6);
    *r3 = _mm256_permute2f128_ps::<0x20>(u3, u7);
    *r4 = _mm256_permute2f128_ps::<0x31>(u0, u4);
    *r5 = _mm256_permute2f128_ps::<0x31>(u1, u5);
    *r6 = _mm256_permute2f128_ps::<0x31>(u2, u6);
    *r7 = _mm256_permute2f128_ps::<0x31>(u3, u7);
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dct_2d_avx2(io: &mut MutableSubgrid<'_>, direction: DctDirection) {
    if io.width() % LANE_SIZE != 0 || io.height() % LANE_SIZE != 0 {
        return generic::dct_2d(io, direction);
    }

    let Some(mut io) = io.as_vectored::<Lane>() else {
        tracing::trace!("Input buffer is not aligned");
        return generic::dct_2d(io, direction);
    };

    // Fast path for 8×8 (the dominant block size in VarDCT): avoids a heap
    // allocation for scratch space and keeps data in registers between passes.
    if io.width() == 1 && io.height() == LANE_SIZE {
        return dct8x8_avx2(&mut io, direction);
    }

    dct_2d_lane(&mut io, direction);
}

/// Specialised 2-D DCT for 8×8 blocks (represented as a 1×8 `MutableSubgrid<__m256>`).
///
/// Compared to the general `dct_2d_lane` path this function:
/// - Avoids the per-block `vec!` heap allocation for scratch space.
/// - Keeps the 512-byte working set in AVX2 registers across both passes.
#[target_feature(enable = "avx2,fma")]
unsafe fn dct8x8_avx2(io: &mut MutableSubgrid<'_, Lane>, direction: DctDirection) {
    let mut data: [Lane; 8] = [_mm256_setzero_ps(); 8];

    // Load all 8 rows into registers.
    for y in 0..8usize {
        data[y] = io.get(0, y);
    }

    // Pass 1: column-direction DCT (element-wise across 8 parallel columns).
    if direction == DctDirection::Forward {
        dct8_forward(&mut MutableSubgrid::from_buf(&mut data, 1, 8, 1));
    } else {
        dct8_inverse(&mut MutableSubgrid::from_buf(&mut data, 1, 8, 1));
    }
    // Transpose so that the second pass operates on the row direction.
    transpose_lane(&mut data);

    // Pass 2: row-direction DCT (same butterfly, data already transposed).
    if direction == DctDirection::Forward {
        dct8_forward(&mut MutableSubgrid::from_buf(&mut data, 1, 8, 1));
    } else {
        dct8_inverse(&mut MutableSubgrid::from_buf(&mut data, 1, 8, 1));
    }
    // Transpose back to restore canonical layout.
    transpose_lane(&mut data);

    // Store results.
    for y in 0..8usize {
        *io.get_mut(0, y) = data[y];
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dct_2d_lane(io: &mut MutableSubgrid<'_, Lane>, direction: DctDirection) {
    let scratch_size = io.height().max(io.width() * LANE_SIZE) * 2;
    let mut scratch_lanes = vec![_mm256_setzero_ps(); scratch_size];
    column_dct_lane(io, &mut scratch_lanes, direction);
    row_dct_lane(io, &mut scratch_lanes, direction);
}

#[target_feature(enable = "avx2,fma")]
unsafe fn column_dct_lane(
    io: &mut MutableSubgrid<'_, Lane>,
    scratch: &mut [Lane],
    direction: DctDirection,
) {
    let width = io.width();
    let height = io.height();
    let (io_lanes, scratch_lanes) = scratch[..height * 2].split_at_mut(height);
    for x in 0..width {
        for (y, input) in io_lanes.iter_mut().enumerate() {
            *input = io.get(x, y);
        }
        dct(io_lanes, scratch_lanes, direction);
        for (y, output) in io_lanes.chunks_exact_mut(LANE_SIZE).enumerate() {
            transpose_lane(output);
            for (dy, output) in output.iter_mut().enumerate() {
                *io.get_mut(x, y * LANE_SIZE + dy) = *output;
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn row_dct_lane(
    io: &mut MutableSubgrid<'_, Lane>,
    scratch: &mut [Lane],
    direction: DctDirection,
) {
    let width = io.width() * LANE_SIZE;
    let height = io.height();
    let (io_lanes, scratch_lanes) = scratch[..width * 2].split_at_mut(width);
    for y in (0..height).step_by(LANE_SIZE) {
        for (x, input) in io_lanes.chunks_exact_mut(LANE_SIZE).enumerate() {
            for (dy, input) in input.iter_mut().enumerate() {
                *input = io.get(x, y + dy);
            }
        }
        dct(io_lanes, scratch_lanes, direction);
        for (x, output) in io_lanes.chunks_exact_mut(LANE_SIZE).enumerate() {
            transpose_lane(output);
            for (dy, output) in output.iter_mut().enumerate() {
                *io.get_mut(x, y + dy) = *output;
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dct4_forward(input: [Lane; 4]) -> [Lane; 4] {
    let sec0 = Lane::splat_f32(0.5411961 / 4.0);
    let sec1 = Lane::splat_f32(1.306563 / 4.0);
    let quarter = Lane::splat_f32(0.25);
    let sqrt2 = Lane::splat_f32(std::f32::consts::SQRT_2);

    let sum03 = input[0].add(input[3]);
    let sum12 = input[1].add(input[2]);
    let tmp0 = input[0].sub(input[3]).mul(sec0);
    let tmp1 = input[1].sub(input[2]).mul(sec1);
    let out0 = tmp0.add(tmp1);
    let out1 = tmp0.sub(tmp1);

    [
        sum03.add(sum12).mul(quarter),
        out0.muladd(sqrt2, out1),
        sum03.sub(sum12).mul(quarter),
        out1,
    ]
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dct4_inverse(input: [Lane; 4]) -> [Lane; 4] {
    let sec0 = Lane::splat_f32(0.5411961);
    let sec1 = Lane::splat_f32(1.306563);
    let sqrt2 = Lane::splat_f32(std::f32::consts::SQRT_2);

    let tmp0 = input[1].mul(sqrt2);
    let tmp1 = input[1].add(input[3]);
    let out0 = tmp0.add(tmp1).mul(sec0);
    let out1 = tmp0.sub(tmp1).mul(sec1);
    let sum02 = input[0].add(input[2]);
    let sub02 = input[0].sub(input[2]);

    [
        sum02.add(out0),
        sub02.add(out1),
        sub02.sub(out1),
        sum02.sub(out0),
    ]
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dct8_forward(io: &mut MutableSubgrid<'_, Lane>) {
    assert!(io.height() == 8);
    let half = Lane::splat_f32(0.5);
    let sqrt2 = Lane::splat_f32(std::f32::consts::SQRT_2);
    let sec = dct_common::sec_half_small(8);

    let input0 = [
        io.get(0, 0).add(io.get(0, 7)).mul(half),
        io.get(0, 1).add(io.get(0, 6)).mul(half),
        io.get(0, 2).add(io.get(0, 5)).mul(half),
        io.get(0, 3).add(io.get(0, 4)).mul(half),
    ];
    let input1 = [
        io.get(0, 0)
            .sub(io.get(0, 7))
            .mul(Lane::splat_f32(sec[0] / 2.0)),
        io.get(0, 1)
            .sub(io.get(0, 6))
            .mul(Lane::splat_f32(sec[1] / 2.0)),
        io.get(0, 2)
            .sub(io.get(0, 5))
            .mul(Lane::splat_f32(sec[2] / 2.0)),
        io.get(0, 3)
            .sub(io.get(0, 4))
            .mul(Lane::splat_f32(sec[3] / 2.0)),
    ];
    let output0 = dct4_forward(input0);
    for (idx, v) in output0.into_iter().enumerate() {
        *io.get_mut(0, idx * 2) = v;
    }
    let mut output1 = dct4_forward(input1);
    output1[0] = output1[0].mul(sqrt2);
    for idx in 0..3 {
        *io.get_mut(0, idx * 2 + 1) = output1[idx].add(output1[idx + 1]);
    }
    *io.get_mut(0, 7) = output1[3];
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dct8_inverse(io: &mut MutableSubgrid<'_, Lane>) {
    assert!(io.height() == 8);
    let sqrt2 = Lane::splat_f32(std::f32::consts::SQRT_2);
    let sec = dct_common::sec_half_small(8);

    let input0 = [io.get(0, 0), io.get(0, 2), io.get(0, 4), io.get(0, 6)];
    let input1 = [
        io.get(0, 1).mul(sqrt2),
        io.get(0, 3).add(io.get(0, 1)),
        io.get(0, 5).add(io.get(0, 3)),
        io.get(0, 7).add(io.get(0, 5)),
    ];
    let output0 = dct4_inverse(input0);
    let output1 = dct4_inverse(input1);
    for (idx, &sec) in sec.iter().enumerate() {
        let r = output1[idx].mul(Lane::splat_f32(sec));
        *io.get_mut(0, idx) = output0[idx].add(r);
        *io.get_mut(0, 7 - idx) = output0[idx].sub(r);
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dct(io: &mut [Lane], scratch: &mut [Lane], direction: DctDirection) {
    let n = io.len();
    assert!(scratch.len() == n);

    if n == 0 {
        return;
    }
    if n == 1 {
        return;
    }

    let half = Lane::splat_f32(0.5);
    if n == 2 {
        let tmp0 = io[0].add(io[1]);
        let tmp1 = io[0].sub(io[1]);
        if direction == DctDirection::Forward {
            io[0] = tmp0.mul(half);
            io[1] = tmp1.mul(half);
        } else {
            io[0] = tmp0;
            io[1] = tmp1;
        }
        return;
    }

    if n == 4 {
        if direction == DctDirection::Forward {
            io.copy_from_slice(&dct4_forward([io[0], io[1], io[2], io[3]]));
        } else {
            io.copy_from_slice(&dct4_inverse([io[0], io[1], io[2], io[3]]));
        }
        return;
    }

    let sqrt2 = Lane::splat_f32(std::f32::consts::SQRT_2);
    if n == 8 {
        if direction == DctDirection::Forward {
            dct8_forward(&mut MutableSubgrid::from_buf(io, 1, 8, 1));
        } else {
            dct8_inverse(&mut MutableSubgrid::from_buf(io, 1, 8, 1));
        }
        return;
    }

    assert!(n.is_power_of_two());

    if direction == DctDirection::Forward {
        let (input0, input1) = scratch.split_at_mut(n / 2);
        for (idx, &sec) in dct_common::sec_half(n).iter().enumerate() {
            input0[idx] = io[idx].add(io[n - idx - 1]).mul(half);
            input1[idx] = io[idx].sub(io[n - idx - 1]).mul(Lane::splat_f32(sec / 2.0));
        }
        let (output0, output1) = io.split_at_mut(n / 2);
        dct(input0, output0, DctDirection::Forward);
        dct(input1, output1, DctDirection::Forward);
        for (idx, v) in input0.iter().enumerate() {
            io[idx * 2] = *v;
        }
        input1[0] = input1[0].mul(sqrt2);
        for idx in 0..(n / 2 - 1) {
            io[idx * 2 + 1] = input1[idx].add(input1[idx + 1]);
        }
        io[n - 1] = input1[n / 2 - 1];
    } else {
        let (input0, input1) = scratch.split_at_mut(n / 2);
        for idx in 1..(n / 2) {
            let idx = n / 2 - idx;
            input0[idx] = io[idx * 2];
            input1[idx] = io[idx * 2 + 1].add(io[idx * 2 - 1]);
        }
        input0[0] = io[0];
        input1[0] = io[1].mul(sqrt2);
        let (output0, output1) = io.split_at_mut(n / 2);
        dct(input0, output0, DctDirection::Inverse);
        dct(input1, output1, DctDirection::Inverse);
        for (idx, &sec) in dct_common::sec_half(n).iter().enumerate() {
            let r = input1[idx].mul(Lane::splat_f32(sec));
            output0[idx] = input0[idx].add(r);
            output1[n / 2 - idx - 1] = input0[idx].sub(r);
        }
    }
}
