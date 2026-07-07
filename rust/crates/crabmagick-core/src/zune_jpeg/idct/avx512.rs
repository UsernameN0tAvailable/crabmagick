/*
 * Copyright (c) 2023.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! AVX-512 optimised IDCT.
//!
//! This is a faithful widening of the AVX2 integer IDCT ([`super::avx2`]).
//!
//! The AVX2 routine keeps each of the eight rows of the 8x8 block in a separate
//! 256-bit register (8 x `i32`) and carries out the well known
//! `jpeg`/`stb_image` two-pass 1-D IDCT with a transpose in between.
//!
//! Here the heavy butterfly arithmetic (the `dct_pass` macro, executed twice)
//! runs on 512-bit registers ([`ZmmRegister`]).  The block is loaded into the
//! low 256 bits of every register with `_mm512_zextsi256_si512` (the upper 256
//! bits are zeroed and never observed), so the low lane results are *bit
//! identical* to the AVX2 (and therefore scalar) implementation.  The data
//! movement steps (transpose / pack+store) reuse the already proven AVX2
//! helpers on the low 256 bits, which keeps this implementation trivially
//! correct while still exercising genuine AVX-512F arithmetic instructions.

#![cfg(target_arch = "x86_64")]
#![allow(dead_code)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::ops::{Add, AddAssign, Mul, Sub};

use crate::zune_jpeg::unsafe_utils::{transpose, YmmRegister};

const SCALE_BITS: i32 = 512 + 65536 + (128 << 17);

/// A thin abstraction over a 512-bit register interpreted as 16 x `i32`.
///
/// Only the low 256 bits (8 x `i32`, one JPEG block row) carry meaningful data;
/// the upper lanes are computed but discarded.
#[derive(Clone, Copy)]
struct ZmmRegister {
    zmm: __m512i
}

impl Add for ZmmRegister {
    type Output = ZmmRegister;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            ZmmRegister {
                zmm: _mm512_add_epi32(self.zmm, rhs.zmm)
            }
        }
    }
}

impl Add<i32> for ZmmRegister {
    type Output = ZmmRegister;

    #[inline]
    fn add(self, rhs: i32) -> Self::Output {
        unsafe {
            ZmmRegister {
                zmm: _mm512_add_epi32(self.zmm, _mm512_set1_epi32(rhs))
            }
        }
    }
}

impl Sub for ZmmRegister {
    type Output = ZmmRegister;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe {
            ZmmRegister {
                zmm: _mm512_sub_epi32(self.zmm, rhs.zmm)
            }
        }
    }
}

impl Mul<i32> for ZmmRegister {
    type Output = ZmmRegister;

    #[inline]
    fn mul(self, rhs: i32) -> Self::Output {
        unsafe {
            ZmmRegister {
                zmm: _mm512_mullo_epi32(self.zmm, _mm512_set1_epi32(rhs))
            }
        }
    }
}

impl AddAssign for ZmmRegister {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        unsafe {
            self.zmm = _mm512_add_epi32(self.zmm, rhs.zmm);
        }
    }
}

impl AddAssign<i32> for ZmmRegister {
    #[inline]
    fn add_assign(&mut self, rhs: i32) {
        unsafe {
            self.zmm = _mm512_add_epi32(self.zmm, _mm512_set1_epi32(rhs));
        }
    }
}

#[inline]
const fn shuffle(z: i32, y: i32, x: i32, w: i32) -> i32 {
    (z << 6) | (y << 4) | (x << 2) | w
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx2")]
unsafe fn clamp_avx(reg: __m256i) -> __m256i {
    let min_s = _mm256_set1_epi16(0);
    let max_s = _mm256_set1_epi16(255);

    let max_v = _mm256_max_epi16(reg, min_s);
    _mm256_min_epi16(max_v, max_s)
}

/// Transpose the low-256-bit lanes of the eight registers using the proven AVX2
/// 8x8 transpose, then zero-extend back into 512-bit registers.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx2")]
unsafe fn transpose_z(rows: &mut [ZmmRegister; 8]) {
    let mut r0 = YmmRegister { mm256: _mm512_castsi512_si256(rows[0].zmm) };
    let mut r1 = YmmRegister { mm256: _mm512_castsi512_si256(rows[1].zmm) };
    let mut r2 = YmmRegister { mm256: _mm512_castsi512_si256(rows[2].zmm) };
    let mut r3 = YmmRegister { mm256: _mm512_castsi512_si256(rows[3].zmm) };
    let mut r4 = YmmRegister { mm256: _mm512_castsi512_si256(rows[4].zmm) };
    let mut r5 = YmmRegister { mm256: _mm512_castsi512_si256(rows[5].zmm) };
    let mut r6 = YmmRegister { mm256: _mm512_castsi512_si256(rows[6].zmm) };
    let mut r7 = YmmRegister { mm256: _mm512_castsi512_si256(rows[7].zmm) };

    transpose(
        &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7
    );

    rows[0].zmm = _mm512_zextsi256_si512(r0.mm256);
    rows[1].zmm = _mm512_zextsi256_si512(r1.mm256);
    rows[2].zmm = _mm512_zextsi256_si512(r2.mm256);
    rows[3].zmm = _mm512_zextsi256_si512(r3.mm256);
    rows[4].zmm = _mm512_zextsi256_si512(r4.mm256);
    rows[5].zmm = _mm512_zextsi256_si512(r5.mm256);
    rows[6].zmm = _mm512_zextsi256_si512(r6.mm256);
    rows[7].zmm = _mm512_zextsi256_si512(r7.mm256);
}

// Pack i32 to i16's, clamp them between 0-255, undo shuffling and store back to
// the output array.  Operates on the low 256 bits of the supplied registers.
macro_rules! permute_store {
    ($x:expr,$y:expr,$index:tt,$out:tt,$stride:tt) => {
        let a = _mm256_packs_epi32(_mm512_castsi512_si256($x), _mm512_castsi512_si256($y));

        let b = clamp_avx(a);
        let mut tmp = [0; 8];
        let c = _mm256_permute4x64_epi64(b, shuffle(3, 1, 2, 0));

        _mm_storeu_si128(
            ($out)
                .get_mut($index..$index + 8)
                .unwrap_or(&mut tmp)
                .as_mut_ptr()
                .cast(),
            _mm256_extractf128_si256::<0>(c)
        );
        $index += $stride;
        _mm_storeu_si128(
            ($out)
                .get_mut($index..$index + 8)
                .unwrap()
                .as_mut_ptr()
                .cast(),
            _mm256_extractf128_si256::<1>(c)
        );
        $index += $stride;
    };
}

#[target_feature(enable = "avx512f,avx512bw,avx2")]
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::op_ref,
    unused_assignments,
    clippy::zero_prefixed_literal
)]
pub unsafe fn idct_avx512(in_vector: &mut [i32; 64], out_vector: &mut [i16], stride: usize) {
    let mut pos = 0;

    let rw0 = _mm256_loadu_si256(in_vector[00..].as_ptr().cast());
    let rw1 = _mm256_loadu_si256(in_vector[08..].as_ptr().cast());
    let rw2 = _mm256_loadu_si256(in_vector[16..].as_ptr().cast());
    let rw3 = _mm256_loadu_si256(in_vector[24..].as_ptr().cast());
    let rw4 = _mm256_loadu_si256(in_vector[32..].as_ptr().cast());
    let rw5 = _mm256_loadu_si256(in_vector[40..].as_ptr().cast());
    let rw6 = _mm256_loadu_si256(in_vector[48..].as_ptr().cast());
    let rw7 = _mm256_loadu_si256(in_vector[56..].as_ptr().cast());

    // AC-term all-zero short circuit, identical to the AVX2 path.
    let rw8 = _mm256_loadu_si256(in_vector[1..].as_ptr().cast());

    let mut bitmap = _mm256_or_si256(rw1, rw2);
    bitmap = _mm256_or_si256(bitmap, rw3);
    bitmap = _mm256_or_si256(bitmap, rw4);
    bitmap = _mm256_or_si256(bitmap, rw5);
    bitmap = _mm256_or_si256(bitmap, rw6);
    bitmap = _mm256_or_si256(bitmap, rw7);
    bitmap = _mm256_or_si256(bitmap, rw8);

    if _mm256_testz_si256(bitmap, bitmap) == 1 {
        let coeff = ((in_vector[0] + 4 + 1024) >> 3).clamp(0, 255) as i16;
        let idct_value = _mm_set1_epi16(coeff);

        macro_rules! store {
            ($pos:tt,$value:tt) => {
                _mm_storeu_si128(
                    out_vector
                        .get_mut($pos..$pos + 8)
                        .unwrap()
                        .as_mut_ptr()
                        .cast(),
                    $value
                );
                $pos += stride;
            };
        }
        store!(pos, idct_value);
        store!(pos, idct_value);
        store!(pos, idct_value);
        store!(pos, idct_value);

        store!(pos, idct_value);
        store!(pos, idct_value);
        store!(pos, idct_value);
        store!(pos, idct_value);

        return;
    }

    let mut rows = [
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw0) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw1) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw2) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw3) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw4) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw5) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw6) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw7) }
    ];

    macro_rules! dct_pass {
        ($SCALE_BITS:expr,$scale:tt) => {
            let row0 = rows[0];
            let row1 = rows[1];
            let row2 = rows[2];
            let row3 = rows[3];
            let row4 = rows[4];
            let row5 = rows[5];
            let row6 = rows[6];
            let row7 = rows[7];

            // even part
            let p1 = (row2 + row6) * 2217;

            let mut t2 = p1 + row6 * -7567;
            let mut t3 = p1 + row2 * 3135;

            let mut t0 = ZmmRegister {
                zmm: _mm512_slli_epi32::<12>((row0 + row4).zmm)
            };
            let mut t1 = ZmmRegister {
                zmm: _mm512_slli_epi32::<12>((row0 - row4).zmm)
            };

            let x0 = t0 + t3 + $SCALE_BITS;
            let x3 = t0 - t3 + $SCALE_BITS;
            let x1 = t1 + t2 + $SCALE_BITS;
            let x2 = t1 - t2 + $SCALE_BITS;

            let p3 = row7 + row3;
            let p4 = row5 + row1;
            let p1 = row7 + row1;
            let p2 = row5 + row3;
            let p5 = (p3 + p4) * 4816;

            t0 = row7 * 1223;
            t1 = row5 * 8410;
            t2 = row3 * 12586;
            t3 = row1 * 6149;

            let p1 = p5 + p1 * -3685;
            let p2 = p5 + (p2 * -10497);
            let p3 = p3 * -8034;
            let p4 = p4 * -1597;

            t3 += p1 + p4;
            t2 += p2 + p3;
            t1 += p2 + p4;
            t0 += p1 + p3;

            rows[0].zmm = _mm512_srai_epi32::<$scale>((x0 + t3).zmm);
            rows[1].zmm = _mm512_srai_epi32::<$scale>((x1 + t2).zmm);
            rows[2].zmm = _mm512_srai_epi32::<$scale>((x2 + t1).zmm);
            rows[3].zmm = _mm512_srai_epi32::<$scale>((x3 + t0).zmm);

            rows[4].zmm = _mm512_srai_epi32::<$scale>((x3 - t0).zmm);
            rows[5].zmm = _mm512_srai_epi32::<$scale>((x2 - t1).zmm);
            rows[6].zmm = _mm512_srai_epi32::<$scale>((x1 - t2).zmm);
            rows[7].zmm = _mm512_srai_epi32::<$scale>((x0 - t3).zmm);
        };
    }

    // Process rows
    dct_pass!(512, 10);
    transpose_z(&mut rows);

    // process columns
    dct_pass!(SCALE_BITS, 17);
    transpose_z(&mut rows);

    permute_store!(rows[0].zmm, rows[1].zmm, pos, out_vector, stride);
    permute_store!(rows[2].zmm, rows[3].zmm, pos, out_vector, stride);
    permute_store!(rows[4].zmm, rows[5].zmm, pos, out_vector, stride);
    permute_store!(rows[6].zmm, rows[7].zmm, pos, out_vector, stride);
}

#[target_feature(enable = "avx512f,avx512bw,avx2")]
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::op_ref,
    unused_assignments,
    clippy::zero_prefixed_literal
)]
pub unsafe fn idct_avx512_4x4(in_vector: &mut [i32; 64], out_vector: &mut [i16], stride: usize) {
    let rw0 = _mm256_loadu_si256(in_vector[00..].as_ptr().cast());
    let rw1 = _mm256_loadu_si256(in_vector[08..].as_ptr().cast());
    let rw2 = _mm256_loadu_si256(in_vector[16..].as_ptr().cast());
    let rw3 = _mm256_loadu_si256(in_vector[24..].as_ptr().cast());

    let mut rows = [
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw0) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw1) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw2) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw3) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw0) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw0) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw0) },
        ZmmRegister { zmm: _mm512_zextsi256_si512(rw0) }
    ];

    {
        rows[0].zmm = _mm512_slli_epi32::<12>(rows[0].zmm);
        rows[0] += 512;

        let row0 = rows[0];
        let i2 = rows[2];

        let p1 = i2 * 2217;
        let p3 = i2 * 5352;

        let x0 = row0 + p3;
        let x1 = row0 + p1;
        let x2 = row0 - p1;
        let x3 = row0 - p3;

        let i4 = rows[3];
        let i3 = rows[1];

        let p5 = (i4 + i3) * 4816;

        let p1 = p5 + i3 * -3685;
        let p2 = p5 + i4 * -10497;

        let t3 = p5 + i3 * 867;
        let t2 = p5 + i4 * -5945;

        let t1 = p2 + i3 * -1597;
        let t0 = p1 + i4 * -8034;

        rows[0].zmm = _mm512_srai_epi32::<10>((x0 + t3).zmm);
        rows[1].zmm = _mm512_srai_epi32::<10>((x1 + t2).zmm);
        rows[2].zmm = _mm512_srai_epi32::<10>((x2 + t1).zmm);
        rows[3].zmm = _mm512_srai_epi32::<10>((x3 + t0).zmm);

        rows[4].zmm = _mm512_srai_epi32::<10>((x3 - t0).zmm);
        rows[5].zmm = _mm512_srai_epi32::<10>((x2 - t1).zmm);
        rows[6].zmm = _mm512_srai_epi32::<10>((x1 - t2).zmm);
        rows[7].zmm = _mm512_srai_epi32::<10>((x0 - t3).zmm);
    }

    transpose_z(&mut rows);

    {
        let i2 = rows[2];
        let i0 = rows[0];

        rows[0].zmm = _mm512_slli_epi32::<12>(i0.zmm);
        let t0 = rows[0] + SCALE_BITS;

        let t2 = i2 * 2217;
        let t3 = i2 * 5352;

        let x0 = t0 + t3;
        let x3 = t0 - t3;
        let x1 = t0 + t2;
        let x2 = t0 - t2;

        let i3 = rows[3];
        let i1 = rows[1];

        let p5 = (i3 + i1) * 4816;

        let p1 = p5 + i1 * -3685;
        let p2 = p5 + i3 * -10497;

        let t3 = p5 + i1 * 867;
        let t2 = p5 + i3 * -5945;

        let t1 = p2 + i1 * -1597;
        let t0 = p1 + i3 * -8034;

        rows[0].zmm = _mm512_srai_epi32::<17>((x0 + t3).zmm);
        rows[1].zmm = _mm512_srai_epi32::<17>((x1 + t2).zmm);
        rows[2].zmm = _mm512_srai_epi32::<17>((x2 + t1).zmm);
        rows[3].zmm = _mm512_srai_epi32::<17>((x3 + t0).zmm);
        rows[4].zmm = _mm512_srai_epi32::<17>((x3 - t0).zmm);
        rows[5].zmm = _mm512_srai_epi32::<17>((x2 - t1).zmm);
        rows[6].zmm = _mm512_srai_epi32::<17>((x1 - t2).zmm);
        rows[7].zmm = _mm512_srai_epi32::<17>((x0 - t3).zmm);
    }

    transpose_z(&mut rows);

    let mut pos = 0;

    permute_store!(rows[0].zmm, rows[1].zmm, pos, out_vector, stride);
    permute_store!(rows[2].zmm, rows[3].zmm, pos, out_vector, stride);
    permute_store!(rows[4].zmm, rows[5].zmm, pos, out_vector, stride);
    permute_store!(rows[6].zmm, rows[7].zmm, pos, out_vector, stride);
}
