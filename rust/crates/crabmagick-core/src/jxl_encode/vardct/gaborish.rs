// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Gaborish inverse pre-filter for the encoder.
//!
//! Applies a 5x5 symmetric sharpening kernel to XYB channels before DCT.
//! The decoder applies a 3x3 Gabor-like blur; this encoder-side inverse
//! compensates, reducing blocking artifacts and improving rate-distortion.
//!
//! Ported from libjxl `lib/jxl/enc_gaborish.cc`.

/// Butteraugli-optimized 5x5 symmetric kernel weights.
///
/// These are NOT the mathematical inverse of the decoder's 3x3 blur — they
/// were optimized by butteraugli for favorable rate-distortion tradeoffs.
///
/// Kernel layout (lower-right quadrant):
/// ```text
///   c  r  R
///   r  d  L
///   R  L  D
/// ```
/// where:
///   r = kGaborish[0] (orthogonal distance 1)
///   d = kGaborish[1] (diagonal distance sqrt(2))
///   R = kGaborish[2] (orthogonal distance 2)
///   L = kGaborish[3] (knight's move distance)
///   D = kGaborish[4] (corner distance 2*sqrt(2))
const K_GABORISH: [f64; 5] = [
    -0.09495815671340026,   // [0] r: orthogonal dist 1
    -0.041031725066768575,  // [1] d: diagonal dist sqrt(2)
    0.013710004822696948,   // [2] R: orthogonal dist 2
    0.006510206083837737,   // [3] L: knight's move
    -0.0014789063378272242, // [4] D: corner dist 2*sqrt(2)
];

/// Compute normalized weights for one channel.
///
/// Returns `(center_weight, r, d, big_r, l, big_d)` all as f32.
fn compute_weights(mul: f64) -> (f32, f32, f32, f32, f32, f32) {
    let sum = 1.0
        + mul
            * 4.0
            * (K_GABORISH[0] + K_GABORISH[1] + K_GABORISH[2] + K_GABORISH[4] + 2.0 * K_GABORISH[3]);
    let sum = if sum < 1e-5 { 1e-5 } else { sum };
    let normalize = 1.0 / sum;
    let normalize_mul = mul * normalize;

    (
        normalize as f32,                       // center
        (normalize_mul * K_GABORISH[0]) as f32, // r
        (normalize_mul * K_GABORISH[1]) as f32, // d
        (normalize_mul * K_GABORISH[2]) as f32, // R
        (normalize_mul * K_GABORISH[3]) as f32, // L
        (normalize_mul * K_GABORISH[4]) as f32, // D
    )
}

/// Apply the gaborish inverse (5x5 sharpening) to one channel in-place.
///
/// Uses a thread-local scratch buffer to avoid per-call mmap/page-fault overhead.
/// On the first call per thread the buffer is allocated and all pages mapped via
/// `resize(n, 0.0)`. Subsequent calls on the same thread reuse the mapped pages,
/// eliminating the ~12ms/channel page-fault cost for large (48 MB) images.
fn apply_channel(data: &mut [f32], width: usize, height: usize, mul: f64) {
    let n = width * height;
    let (wc, wr, wd, w_big_r, wl, w_big_d) = compute_weights(mul);

    thread_local! {
        // Grows on first use (or when a larger image arrives), never shrinks.
        // resize(n, 0.0) zero-fills new pages → they are mapped → no future faults.
        static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        if s.len() < n {
            s.resize(n, 0.0);
        }
        crate::jxl_encode_simd::gaborish_5x5_channel(
            data,
            &mut s[..n],
            width,
            height,
            wc,
            wr,
            wd,
            w_big_r,
            wl,
            w_big_d,
        );
    });
}

/// Apply gaborish inverse sharpening to all three XYB channels.
///
/// This should be called AFTER noise estimation/denoising and BEFORE
/// adaptive quantization, matching the libjxl pipeline order.
///
/// Uses `mul=[1.0, 1.0, 1.0]` for all channels (libjxl VarDCT default).
///
/// With the `parallel` feature the three independent channels run concurrently
/// via `rayon::join`.  Each channel uses a per-thread scratch buffer (thread_local)
/// so large-image page faults happen at most once per rayon thread rather than
/// on every encode call.
pub fn gaborish_inverse(
    xyb_x: &mut [f32],
    xyb_y: &mut [f32],
    xyb_b: &mut [f32],
    width: usize,
    height: usize,
) {
    #[cfg(feature = "parallel")]
    {
        if crate::jxl_encode::parallel::sequential_maps_forced() {
            apply_channel(xyb_x, width, height, 1.0);
            apply_channel(xyb_y, width, height, 1.0);
            apply_channel(xyb_b, width, height, 1.0);
        } else {
            let (((), ()), ()) = rayon::join(
                || {
                    rayon::join(
                        || apply_channel(xyb_x, width, height, 1.0),
                        || apply_channel(xyb_y, width, height, 1.0),
                    )
                },
                || apply_channel(xyb_b, width, height, 1.0),
            );
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        apply_channel(xyb_x, width, height, 1.0);
        apply_channel(xyb_y, width, height, 1.0);
        apply_channel(xyb_b, width, height, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_normalization() {
        // With mul=1.0, the weights should sum to 1.0
        let (wc, wr, wd, w_big_r, wl, w_big_d) = compute_weights(1.0);
        let sum = wc + 4.0 * wr + 4.0 * wd + 4.0 * w_big_r + 8.0 * wl + 4.0 * w_big_d;
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Kernel weights should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_uniform_image_preserved() {
        let width = 16;
        let height = 16;
        let value = 0.5f32;
        let mut x = vec![value; width * height];
        let mut y = x.clone();
        let mut b = x.clone();
        gaborish_inverse(&mut x, &mut y, &mut b, width, height);
        for (i, &v) in x.iter().enumerate() {
            assert!(
                (v - value).abs() < 1e-5,
                "Pixel {} changed: {} -> {}",
                i,
                value,
                v
            );
        }
    }

    #[test]
    fn test_sharpening_effect() {
        let width = 8;
        let height = 8;
        let mut data = vec![0.0f32; width * height];
        data[4 * width + 4] = 1.0;
        let original_center = data[4 * width + 4];
        let mut dummy = vec![0.0f32; width * height];
        gaborish_inverse(&mut data, &mut dummy.clone(), &mut dummy, width, height);
        assert!(
            data[4 * width + 4] > original_center,
            "Sharpening should increase isolated bright pixel"
        );
        assert!(
            data[4 * width + 3] < 0.0,
            "Sharpening should create negative ringing at neighbors"
        );
    }
}
