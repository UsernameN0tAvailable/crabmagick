/*
 * Copyright (c) 2023.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! AVX-512 YCbCr -> RGB / RGBA color conversion.
//!
//! Where the AVX2 routines split the 16 pixels into a low/high `i16` half, the
//! AVX-512 path widens every pixel to a full `i32` lane and processes all 16
//! pixels of an MCU row in a single `__m512i`.  The fixed-point arithmetic uses
//! exactly the same 14-bit BT.601 full-range coefficients as
//! [`super::scalar`], so the produced bytes are bit-identical to the scalar
//! reference implementation.

#![cfg(target_arch = "x86_64")]
#![allow(
    clippy::wildcard_imports,
    clippy::cast_possible_truncation,
    clippy::too_many_arguments,
    clippy::inline_always,
    clippy::doc_markdown,
    dead_code
)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::jpeg_decode::color_convert::scalar::{
    CB_CF, CR_CF, C_G_CB_COEF_2, C_G_CR_COEF_1, YUV_PREC, YUV_RND, Y_CF
};

// The `_mm512_srai_epi32::<14>` below hard-codes the shift; make sure the shared
// precision constant agrees so the two never drift apart.
const _: () = assert!(YUV_PREC == 14);

/// Widen 16 `i16` values behind `ptr` into 16 sign-extended `i32` lanes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx2")]
unsafe fn load_i16x16_to_i32(ptr: *const i16) -> __m512i {
    _mm512_cvtepi16_epi32(_mm256_loadu_si256(ptr.cast()))
}

/// Compute the three (unclamped, then clamped to `0..=255`) RGB planes for 16
/// pixels, each returned as a `__m128i` of 16 `u8` in natural pixel order.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx2")]
unsafe fn ycbcr_to_rgb_planes(
    y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16]
) -> (__m128i, __m128i, __m128i) {
    let y_i = load_i16x16_to_i32(y.as_ptr());
    let cb_i = load_i16x16_to_i32(cb.as_ptr());
    let cr_i = load_i16x16_to_i32(cr.as_ptr());

    let bias = _mm512_set1_epi32(128);
    let crm = _mm512_sub_epi32(cr_i, bias);
    let cbm = _mm512_sub_epi32(cb_i, bias);

    // y0 = y * Y_CF + YUV_RND
    let y0 = _mm512_add_epi32(
        _mm512_mullo_epi32(y_i, _mm512_set1_epi32(i32::from(Y_CF))),
        _mm512_set1_epi32(i32::from(YUV_RND))
    );

    let r = _mm512_add_epi32(y0, _mm512_mullo_epi32(crm, _mm512_set1_epi32(i32::from(CR_CF))));
    let g = _mm512_add_epi32(
        _mm512_add_epi32(y0, _mm512_mullo_epi32(crm, _mm512_set1_epi32(i32::from(C_G_CR_COEF_1)))),
        _mm512_mullo_epi32(cbm, _mm512_set1_epi32(i32::from(C_G_CB_COEF_2)))
    );
    let b = _mm512_add_epi32(y0, _mm512_mullo_epi32(cbm, _mm512_set1_epi32(i32::from(CB_CF))));

    let r = _mm512_srai_epi32::<14>(r);
    let g = _mm512_srai_epi32::<14>(g);
    let b = _mm512_srai_epi32::<14>(b);

    let zero = _mm512_setzero_si512();
    let max = _mm512_set1_epi32(255);

    let clamp = |v| _mm512_min_epi32(_mm512_max_epi32(v, zero), max);
    let r = clamp(r);
    let g = clamp(g);
    let b = clamp(b);

    // Values are within 0..=255 so a plain truncating narrow yields the exact byte.
    (
        _mm512_cvtepi32_epi8(r),
        _mm512_cvtepi32_epi8(g),
        _mm512_cvtepi32_epi8(b)
    )
}

/// YCbCr -> RGBA (alpha forced to 255), 16 pixels, 64 bytes written.
#[inline(always)]
pub fn ycbcr_to_rgba_avx512(
    y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], out: &mut [u8], offset: &mut usize
) {
    unsafe {
        ycbcr_to_rgba_avx512_inner(y, cb, cr, out, offset);
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx2")]
unsafe fn ycbcr_to_rgba_avx512_inner(
    y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], out: &mut [u8], offset: &mut usize
) {
    let tmp: &mut [u8; 64] = out
        .get_mut(*offset..*offset + 64)
        .expect("Slice too small cannot write")
        .try_into()
        .unwrap();

    let (r8, g8, b8) = ycbcr_to_rgb_planes(y, cb, cr);
    let a8 = _mm_set1_epi8(-1); // 255

    // Interleave to AoS: (r,g) and (b,a) as byte pairs, then combine as 16-bit units.
    let rg_lo = _mm_unpacklo_epi8(r8, g8);
    let rg_hi = _mm_unpackhi_epi8(r8, g8);
    let ba_lo = _mm_unpacklo_epi8(b8, a8);
    let ba_hi = _mm_unpackhi_epi8(b8, a8);

    let out0 = _mm_unpacklo_epi16(rg_lo, ba_lo); // pixels 0..=3
    let out1 = _mm_unpackhi_epi16(rg_lo, ba_lo); // pixels 4..=7
    let out2 = _mm_unpacklo_epi16(rg_hi, ba_hi); // pixels 8..=11
    let out3 = _mm_unpackhi_epi16(rg_hi, ba_hi); // pixels 12..=15

    _mm_storeu_si128(tmp[0..].as_mut_ptr().cast(), out0);
    _mm_storeu_si128(tmp[16..].as_mut_ptr().cast(), out1);
    _mm_storeu_si128(tmp[32..].as_mut_ptr().cast(), out2);
    _mm_storeu_si128(tmp[48..].as_mut_ptr().cast(), out3);

    *offset += 64;
}

/// Build the `_mm_shuffle_epi8` control mask that scatters a single channel's
/// bytes into their positions inside one 16-byte slice of the interleaved RGB
/// output.  `chunk` selects the output slice (0,1,2 -> bytes 0..16, 16..32,
/// 32..48) and `channel` selects R=0, G=1, B=2.
const fn rgb_shuffle_mask(chunk: usize, channel: usize) -> [i8; 16] {
    let mut mask = [-128i8; 16]; // high bit set -> writes a zero byte
    let mut local = 0;
    while local < 16 {
        let global = chunk * 16 + local;
        if global % 3 == channel {
            mask[local] = (global / 3) as i8;
        }
        local += 1;
    }
    mask
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx2")]
unsafe fn load_mask(mask: &[i8; 16]) -> __m128i {
    _mm_loadu_si128(mask.as_ptr().cast())
}

/// YCbCr -> RGB, 16 pixels, 48 bytes written.
#[inline(always)]
pub fn ycbcr_to_rgb_avx512(
    y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], out: &mut [u8], offset: &mut usize
) {
    unsafe {
        ycbcr_to_rgb_avx512_inner(y, cb, cr, out, offset);
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx2")]
unsafe fn ycbcr_to_rgb_avx512_inner(
    y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], out: &mut [u8], offset: &mut usize
) {
    let tmp: &mut [u8; 48] = out
        .get_mut(*offset..*offset + 48)
        .expect("Slice too small cannot write")
        .try_into()
        .unwrap();

    let (r8, g8, b8) = ycbcr_to_rgb_planes(y, cb, cr);

    const M00: [i8; 16] = rgb_shuffle_mask(0, 0);
    const M01: [i8; 16] = rgb_shuffle_mask(0, 1);
    const M02: [i8; 16] = rgb_shuffle_mask(0, 2);
    const M10: [i8; 16] = rgb_shuffle_mask(1, 0);
    const M11: [i8; 16] = rgb_shuffle_mask(1, 1);
    const M12: [i8; 16] = rgb_shuffle_mask(1, 2);
    const M20: [i8; 16] = rgb_shuffle_mask(2, 0);
    const M21: [i8; 16] = rgb_shuffle_mask(2, 1);
    const M22: [i8; 16] = rgb_shuffle_mask(2, 2);

    let chunk0 = _mm_or_si128(
        _mm_or_si128(
            _mm_shuffle_epi8(r8, load_mask(&M00)),
            _mm_shuffle_epi8(g8, load_mask(&M01))
        ),
        _mm_shuffle_epi8(b8, load_mask(&M02))
    );
    let chunk1 = _mm_or_si128(
        _mm_or_si128(
            _mm_shuffle_epi8(r8, load_mask(&M10)),
            _mm_shuffle_epi8(g8, load_mask(&M11))
        ),
        _mm_shuffle_epi8(b8, load_mask(&M12))
    );
    let chunk2 = _mm_or_si128(
        _mm_or_si128(
            _mm_shuffle_epi8(r8, load_mask(&M20)),
            _mm_shuffle_epi8(g8, load_mask(&M21))
        ),
        _mm_shuffle_epi8(b8, load_mask(&M22))
    );

    _mm_storeu_si128(tmp[0..].as_mut_ptr().cast(), chunk0);
    _mm_storeu_si128(tmp[16..].as_mut_ptr().cast(), chunk1);
    _mm_storeu_si128(tmp[32..].as_mut_ptr().cast(), chunk2);

    *offset += 48;
}
