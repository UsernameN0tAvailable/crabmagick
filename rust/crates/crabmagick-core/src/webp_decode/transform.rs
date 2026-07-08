static CONST1: i64 = 20091;
static CONST2: i64 = 35468;

pub(crate) fn idct4x4(block: &mut [i32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse4.1") {
            // SAFETY: guarded by runtime feature detection above.
            unsafe {
                sse41::idct4x4_sse41(block);
            }
            return;
        }
    }
    idct4x4_scalar(block);
}

pub(crate) fn idct4x4_scalar(block: &mut [i32]) {
    // The intermediate results may overflow the types, so we stretch the type.
    fn fetch(block: &[i32], idx: usize) -> i64 {
        i64::from(block[idx])
    }

    // Perform one lenght check up front to avoid subsequent bounds checks in this function
    assert!(block.len() >= 16);

    for i in 0usize..4 {
        let a1 = fetch(block, i) + fetch(block, 8 + i);
        let b1 = fetch(block, i) - fetch(block, 8 + i);

        let t1 = (fetch(block, 4 + i) * CONST2) >> 16;
        let t2 = fetch(block, 12 + i) + ((fetch(block, 12 + i) * CONST1) >> 16);
        let c1 = t1 - t2;

        let t1 = fetch(block, 4 + i) + ((fetch(block, 4 + i) * CONST1) >> 16);
        let t2 = (fetch(block, 12 + i) * CONST2) >> 16;
        let d1 = t1 + t2;

        block[i] = (a1 + d1) as i32;
        block[4 + i] = (b1 + c1) as i32;
        block[4 * 3 + i] = (a1 - d1) as i32;
        block[4 * 2 + i] = (b1 - c1) as i32;
    }

    for i in 0usize..4 {
        let a1 = fetch(block, 4 * i) + fetch(block, 4 * i + 2);
        let b1 = fetch(block, 4 * i) - fetch(block, 4 * i + 2);

        let t1 = (fetch(block, 4 * i + 1) * CONST2) >> 16;
        let t2 = fetch(block, 4 * i + 3) + ((fetch(block, 4 * i + 3) * CONST1) >> 16);
        let c1 = t1 - t2;

        let t1 = fetch(block, 4 * i + 1) + ((fetch(block, 4 * i + 1) * CONST1) >> 16);
        let t2 = (fetch(block, 4 * i + 3) * CONST2) >> 16;
        let d1 = t1 + t2;

        block[4 * i] = ((a1 + d1 + 4) >> 3) as i32;
        block[4 * i + 3] = ((a1 - d1 + 4) >> 3) as i32;
        block[4 * i + 1] = ((b1 + c1 + 4) >> 3) as i32;
        block[4 * i + 2] = ((b1 - c1 + 4) >> 3) as i32;
    }
}

// 14.3
pub(crate) fn iwht4x4(block: &mut [i32]) {
    // Perform one length check up front to avoid subsequent bounds checks in this function
    assert!(block.len() >= 16);

    for i in 0usize..4 {
        let a1 = block[i] + block[12 + i];
        let b1 = block[4 + i] + block[8 + i];
        let c1 = block[4 + i] - block[8 + i];
        let d1 = block[i] - block[12 + i];

        block[i] = a1 + b1;
        block[4 + i] = c1 + d1;
        block[8 + i] = a1 - b1;
        block[12 + i] = d1 - c1;
    }

    for block in block.chunks_exact_mut(4) {
        let a1 = block[0] + block[3];
        let b1 = block[1] + block[2];
        let c1 = block[1] - block[2];
        let d1 = block[0] - block[3];

        let a2 = a1 + b1;
        let b2 = c1 + d1;
        let c2 = a1 - b1;
        let d2 = d1 - c1;

        block[0] = (a2 + 3) >> 3;
        block[1] = (b2 + 3) >> 3;
        block[2] = (c2 + 3) >> 3;
        block[3] = (d2 + 3) >> 3;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse41 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const CONST1: i32 = 20091;
    const CONST2: i32 = 35468;

    // Performs `(x * C) >> 16` for all 4 i32 lanes, using full 64-bit
    // intermediate products so the result matches the scalar i64 path exactly
    // (a 32-bit `mullo` would overflow on the pass-2 intermediates).
    //
    // `_mm_mul_epi32` multiplies the signed low 32 bits of lanes 0 and 2 into
    // two 64-bit products. Shifting the operands right by 32 first brings lanes
    // 1 and 3 into that position, so two multiplies cover all four lanes.
    //
    // Because the true `(x*C)>>16` result fits in i32, the low 32 bits of a
    // *logical* 64-bit `>> 16` equal the low 32 bits of an arithmetic shift, so
    // `_mm_srli_epi64` is sufficient before repacking the four results.
    #[inline(always)]
    unsafe fn mulhi(x: __m128i, c: __m128i) -> __m128i {
        let pe = _mm_srli_epi64(_mm_mul_epi32(x, c), 16);
        let po = _mm_srli_epi64(
            _mm_mul_epi32(_mm_srli_epi64(x, 32), _mm_srli_epi64(c, 32)),
            16,
        );
        // pe -> [r0, r2, r0, r2], po -> [r1, r3, r1, r3]
        let pe = _mm_shuffle_epi32::<0b_10_00_10_00>(pe);
        let po = _mm_shuffle_epi32::<0b_10_00_10_00>(po);
        // interleave low -> [r0, r1, r2, r3]
        _mm_unpacklo_epi32(pe, po)
    }

    // Performs `x + ((x * C) >> 16)` for all 4 i32 lanes.
    #[inline(always)]
    unsafe fn mulhi_add(x: __m128i, c: __m128i) -> __m128i {
        _mm_add_epi32(x, mulhi(x, c))
    }

    // Pass 1 butterfly: operates on the four rows as SSE registers (all 4
    // columns at once). No bias/shift. Returns (r0, r1, r2, r3) where the
    // outputs correspond to (a1+d1, b1+c1, b1-c1, a1-d1).
    #[inline(always)]
    unsafe fn butterfly(
        r0: __m128i,
        r1: __m128i,
        r2: __m128i,
        r3: __m128i,
    ) -> (__m128i, __m128i, __m128i, __m128i) {
        let c1 = _mm_set1_epi32(CONST1);
        let c2 = _mm_set1_epi32(CONST2);
        let a1 = _mm_add_epi32(r0, r2);
        let b1 = _mm_sub_epi32(r0, r2);
        let t1 = mulhi(r1, c2);
        let t2 = mulhi_add(r3, c1);
        let cval = _mm_sub_epi32(t1, t2);
        let t1p = mulhi_add(r1, c1);
        let t2p = mulhi(r3, c2);
        let d1 = _mm_add_epi32(t1p, t2p);
        (
            _mm_add_epi32(a1, d1),
            _mm_add_epi32(b1, cval),
            _mm_sub_epi32(b1, cval),
            _mm_sub_epi32(a1, d1),
        )
    }

    // Pass 2 butterfly: same structure but with `+4` bias and `>>3` shift.
    #[inline(always)]
    unsafe fn butterfly_bias(
        r0: __m128i,
        r1: __m128i,
        r2: __m128i,
        r3: __m128i,
    ) -> (__m128i, __m128i, __m128i, __m128i) {
        let c1 = _mm_set1_epi32(CONST1);
        let c2 = _mm_set1_epi32(CONST2);
        let bias = _mm_set1_epi32(4);
        let a1 = _mm_add_epi32(r0, r2);
        let b1 = _mm_sub_epi32(r0, r2);
        let t1 = mulhi(r1, c2);
        let t2 = mulhi_add(r3, c1);
        let cval = _mm_sub_epi32(t1, t2);
        let t1p = mulhi_add(r1, c1);
        let t2p = mulhi(r3, c2);
        let d1 = _mm_add_epi32(t1p, t2p);
        let ad1 = _mm_add_epi32(a1, d1);
        let ad0 = _mm_sub_epi32(a1, d1);
        let bc1 = _mm_add_epi32(b1, cval);
        let bc0 = _mm_sub_epi32(b1, cval);
        (
            _mm_srai_epi32(_mm_add_epi32(ad1, bias), 3),
            _mm_srai_epi32(_mm_add_epi32(bc1, bias), 3),
            _mm_srai_epi32(_mm_add_epi32(bc0, bias), 3),
            _mm_srai_epi32(_mm_add_epi32(ad0, bias), 3),
        )
    }

    // Transpose a 4x4 i32 matrix held in four SSE registers (one row each).
    #[inline(always)]
    unsafe fn transpose4x4(
        r0: __m128i,
        r1: __m128i,
        r2: __m128i,
        r3: __m128i,
    ) -> (__m128i, __m128i, __m128i, __m128i) {
        let t0 = _mm_unpacklo_epi32(r0, r1);
        let t1 = _mm_unpackhi_epi32(r0, r1);
        let t2 = _mm_unpacklo_epi32(r2, r3);
        let t3 = _mm_unpackhi_epi32(r2, r3);
        (
            _mm_unpacklo_epi64(t0, t2),
            _mm_unpackhi_epi64(t0, t2),
            _mm_unpacklo_epi64(t1, t3),
            _mm_unpackhi_epi64(t1, t3),
        )
    }

    #[target_feature(enable = "sse4.1")]
    pub unsafe fn idct4x4_sse41(block: &mut [i32]) {
        assert!(block.len() >= 16);

        let r0 = _mm_loadu_si128(block.as_ptr().add(0) as *const __m128i);
        let r1 = _mm_loadu_si128(block.as_ptr().add(4) as *const __m128i);
        let r2 = _mm_loadu_si128(block.as_ptr().add(8) as *const __m128i);
        let r3 = _mm_loadu_si128(block.as_ptr().add(12) as *const __m128i);

        // Pass 1: column-wise butterfly (operating on rows as vectors).
        let (nr0, nr1, nr2, nr3) = butterfly(r0, r1, r2, r3);

        // Transpose so columns become rows for pass 2.
        let (t0, t1, t2, t3) = transpose4x4(nr0, nr1, nr2, nr3);

        // Pass 2: row-wise butterfly with `+4` bias and `>>3` shift.
        let (out0, out1, out2, out3) = butterfly_bias(t0, t1, t2, t3);

        // Transpose back to row-major order for storing.
        let (f0, f1, f2, f3) = transpose4x4(out0, out1, out2, out3);

        _mm_storeu_si128(block.as_mut_ptr().add(0) as *mut __m128i, f0);
        _mm_storeu_si128(block.as_mut_ptr().add(4) as *mut __m128i, f1);
        _mm_storeu_si128(block.as_mut_ptr().add(8) as *mut __m128i, f2);
        _mm_storeu_si128(block.as_mut_ptr().add(12) as *mut __m128i, f3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn idct4x4_sse41_matches_scalar() {
        if !std::is_x86_feature_detected!("sse4.1") {
            return;
        }

        // Simple deterministic LCG so the test is reproducible without deps.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as i64
        };

        for _ in 0..100_000 {
            // Coefficients within the realistic VP8 dequantized range (i16).
            let mut block: Vec<i32> = (0..16).map(|_| (next() % 65535 - 32767) as i32).collect();
            let mut expected = block.clone();

            idct4x4_scalar(&mut expected);
            // SAFETY: guarded by the feature check above.
            unsafe {
                sse41::idct4x4_sse41(&mut block);
            }

            assert_eq!(block, expected, "SSE4.1 IDCT mismatch");
        }
    }
}
