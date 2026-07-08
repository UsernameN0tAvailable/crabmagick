// AVX2 fast path for the fancy-upsampling YUV -> RGB conversion (RGB / BPP = 3 only).
//
// This file is `include!`d into `yuv.rs` (guarded by `#[cfg(target_arch = "x86_64")]`)
// so it shares the parent module and can use the scalar helpers
// `get_fancy_chroma_value` and `set_pixel` directly.
//
// It reproduces `fill_row_fancy_with_2_uv_rows_scalar::<3>` bit-for-bit while
// processing 8 output pixels (4 chroma "windows") per iteration.

use std::arch::x86_64::*;

/// Load 8 bytes starting at `ptr`, zero-extend to 8 x i32, then build the
/// "main" fancy-chroma vector `[c[k], c[k+1], c[k+1], c[k+2], c[k+2], c[k+3], c[k+3], c[k+4]]`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_perm(ptr: *const u8, idx: __m256i) -> __m256i {
    let v = _mm_loadl_epi64(ptr as *const __m128i);
    let ext = _mm256_cvtepu8_epi32(v);
    _mm256_permutevar8x32_epi32(ext, idx)
}

/// `(9*a + 3*b + 3*c + d + 8) >> 4` lane-wise on 8 x i32 (all lanes non-negative).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn fancy_chroma(
    a: __m256i,
    b: __m256i,
    c: __m256i,
    d: __m256i,
    c9: __m256i,
    c3: __m256i,
    c8: __m256i,
) -> __m256i {
    let t = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(a, c9), _mm256_mullo_epi32(b, c3)),
        _mm256_add_epi32(_mm256_mullo_epi32(c, c3), d),
    );
    _mm256_srli_epi32(_mm256_add_epi32(t, c8), 4)
}

/// Clamp 8 x i32 to `[0, 255]` and pack into the low 8 bytes of a `__m128i`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn pack_to_u8(v: __m256i) -> __m128i {
    let v16 = _mm256_packs_epi32(v, v);
    let v16 = _mm256_permute4x64_epi64(v16, 0xD8);
    let v16_lo = _mm256_castsi256_si128(v16);
    _mm_packus_epi16(v16_lo, _mm_setzero_si128())
}

/// AVX2 implementation of `fill_row_fancy_with_2_uv_rows` for RGB (BPP = 3).
///
/// # Safety
/// Requires the `avx2` target feature to be available at runtime.
#[target_feature(enable = "avx2")]
unsafe fn fill_row_fancy_rgb_avx2(
    row_buffer: &mut [u8],
    y_row: &[u8],
    u_row_1: &[u8],
    u_row_2: &[u8],
    v_row_1: &[u8],
    v_row_2: &[u8],
) {
    const BPP: usize = 3;

    // First pixel: only one chroma column available, mirror it.
    {
        let y_value = y_row[0];
        let u_value = get_fancy_chroma_value(u_row_1[0], u_row_1[0], u_row_2[0], u_row_2[0]);
        let v_value = get_fancy_chroma_value(v_row_1[0], v_row_1[0], v_row_2[0], v_row_2[0]);
        set_pixel(&mut row_buffer[0..3], y_value, u_value, v_value);
    }

    let width = y_row.len();
    // Number of 2-pixel "windows" after the first pixel.
    let num_windows = (width - 1) / 2;

    // Process windows in groups of 4 (8 pixels). Cap `n4` so that:
    //  - the 8-byte chroma loads never read past `chroma_width` (= num_windows + 1),
    //    which requires the last group start `k = n4 - 4` to satisfy `k + 8 <= num_windows + 1`,
    //    i.e. `n4 <= num_windows - 3`;
    //  - the 4-byte store overshoot of the final group stays inside `row_buffer`.
    let n4 = if num_windows >= 3 {
        ((num_windows - 3) / 4) * 4
    } else {
        0
    };

    let rest_row = &mut row_buffer[BPP..];
    let rest_y = &y_row[1..];

    if n4 > 0 {
        let permute_idx = _mm256_set_epi32(4, 3, 3, 2, 2, 1, 1, 0);
        let c9 = _mm256_set1_epi32(9);
        let c3 = _mm256_set1_epi32(3);
        let c8 = _mm256_set1_epi32(8);

        let c19077 = _mm256_set1_epi32(19077);
        let c26149 = _mm256_set1_epi32(26149);
        let c6419 = _mm256_set1_epi32(6419);
        let c13320 = _mm256_set1_epi32(13320);
        let c33050 = _mm256_set1_epi32(33050);
        let r_bias = _mm256_set1_epi32(14234);
        let g_bias = _mm256_set1_epi32(8708);
        let b_bias = _mm256_set1_epi32(17685);

        let mask_r_lo = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 3, -1, -1, 2, -1, -1, 1, -1, -1, 0);
        let mask_g_lo = _mm_set_epi8(-1, -1, -1, -1, -1, 3, -1, -1, 2, -1, -1, 1, -1, -1, 0, -1);
        let mask_b_lo = _mm_set_epi8(-1, -1, -1, -1, 3, -1, -1, 2, -1, -1, 1, -1, -1, 0, -1, -1);
        let mask_r_hi = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 7, -1, -1, 6, -1, -1, 5, -1, -1, 4);
        let mask_g_hi = _mm_set_epi8(-1, -1, -1, -1, -1, 7, -1, -1, 6, -1, -1, 5, -1, -1, 4, -1);
        let mask_b_hi = _mm_set_epi8(-1, -1, -1, -1, 7, -1, -1, 6, -1, -1, 5, -1, -1, 4, -1, -1);

        let rest_row_ptr = rest_row.as_mut_ptr();

        let mut k = 0usize;
        while k < n4 {
            // Fancy chroma upsampling for U.
            let ua = load_perm(u_row_1.as_ptr().add(k), permute_idx);
            let uc = load_perm(u_row_2.as_ptr().add(k), permute_idx);
            let ub = _mm256_shuffle_epi32(ua, 0b10_11_00_01);
            let ud = _mm256_shuffle_epi32(uc, 0b10_11_00_01);
            let u = fancy_chroma(ua, ub, uc, ud, c9, c3, c8);

            // Fancy chroma upsampling for V.
            let va = load_perm(v_row_1.as_ptr().add(k), permute_idx);
            let vc = load_perm(v_row_2.as_ptr().add(k), permute_idx);
            let vb = _mm256_shuffle_epi32(va, 0b10_11_00_01);
            let vd = _mm256_shuffle_epi32(vc, 0b10_11_00_01);
            let v = fancy_chroma(va, vb, vc, vd, c9, c3, c8);

            // 8 luma values for these 8 pixels.
            let y128 = _mm_loadl_epi64(rest_y.as_ptr().add(2 * k) as *const __m128i);
            let y = _mm256_cvtepu8_epi32(y128);

            let ky = _mm256_srli_epi32(_mm256_mullo_epi32(y, c19077), 8);
            let kv_r = _mm256_srli_epi32(_mm256_mullo_epi32(v, c26149), 8);
            let ku_g = _mm256_srli_epi32(_mm256_mullo_epi32(u, c6419), 8);
            let kv_g = _mm256_srli_epi32(_mm256_mullo_epi32(v, c13320), 8);
            let ku_b = _mm256_srli_epi32(_mm256_mullo_epi32(u, c33050), 8);

            let r = _mm256_srai_epi32(_mm256_sub_epi32(_mm256_add_epi32(ky, kv_r), r_bias), 6);
            let g = _mm256_srai_epi32(
                _mm256_add_epi32(
                    _mm256_sub_epi32(_mm256_sub_epi32(ky, ku_g), kv_g),
                    g_bias,
                ),
                6,
            );
            let b = _mm256_srai_epi32(_mm256_sub_epi32(_mm256_add_epi32(ky, ku_b), b_bias), 6);

            let r8 = pack_to_u8(r);
            let g8 = pack_to_u8(g);
            let b8 = pack_to_u8(b);

            let lo = _mm_or_si128(
                _mm_or_si128(
                    _mm_shuffle_epi8(r8, mask_r_lo),
                    _mm_shuffle_epi8(g8, mask_g_lo),
                ),
                _mm_shuffle_epi8(b8, mask_b_lo),
            );
            let hi = _mm_or_si128(
                _mm_or_si128(
                    _mm_shuffle_epi8(r8, mask_r_hi),
                    _mm_shuffle_epi8(g8, mask_g_hi),
                ),
                _mm_shuffle_epi8(b8, mask_b_hi),
            );

            let dst = rest_row_ptr.add(k * BPP * 2);
            _mm_storeu_si128(dst as *mut __m128i, lo);
            _mm_storeu_si128(dst.add(12) as *mut __m128i, hi);

            k += 4;
        }
    }

    // Scalar tail for the remaining windows.
    for i in n4..num_windows {
        let base = i * BPP * 2;
        let y0 = rest_y[2 * i];
        let y1 = rest_y[2 * i + 1];

        let u_value_0 = get_fancy_chroma_value(u_row_1[i], u_row_1[i + 1], u_row_2[i], u_row_2[i + 1]);
        let v_value_0 = get_fancy_chroma_value(v_row_1[i], v_row_1[i + 1], v_row_2[i], v_row_2[i + 1]);
        set_pixel(&mut rest_row[base..base + 3], y0, u_value_0, v_value_0);

        let u_value_1 = get_fancy_chroma_value(u_row_1[i + 1], u_row_1[i], u_row_2[i + 1], u_row_2[i]);
        let v_value_1 = get_fancy_chroma_value(v_row_1[i + 1], v_row_1[i], v_row_2[i + 1], v_row_2[i]);
        set_pixel(&mut rest_row[base + BPP..base + BPP + 3], y1, u_value_1, v_value_1);
    }

    // Final leftover pixel (only present when `width` is even), mirrors the last chroma column.
    if (width - 1) % 2 == 1 {
        let base = num_windows * BPP * 2;
        let y_value = *rest_y.last().unwrap();
        let final_u_1 = *u_row_1.last().unwrap();
        let final_u_2 = *u_row_2.last().unwrap();
        let final_v_1 = *v_row_1.last().unwrap();
        let final_v_2 = *v_row_2.last().unwrap();
        let u_value = get_fancy_chroma_value(final_u_1, final_u_1, final_u_2, final_u_2);
        let v_value = get_fancy_chroma_value(final_v_1, final_v_1, final_v_2, final_v_2);
        set_pixel(&mut rest_row[base..base + 3], y_value, u_value, v_value);
    }
}
