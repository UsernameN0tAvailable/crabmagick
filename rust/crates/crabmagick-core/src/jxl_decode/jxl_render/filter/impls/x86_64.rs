use crate::jxl_decode::jxl_frame::{FrameHeader, filter::EpfParams};
use crate::jxl_decode::jxl_grid::{AlignedGrid, MutableSubgrid};
use crate::jxl_decode::jxl_threadpool::JxlThreadPool;

use crate::jxl_decode::jxl_render::{
    Region,
    filter::{
        epf::run_epf_rows,
        gabor::{run_gabor_row_generic, run_gabor_rows, run_gabor_3ch_rows_unsafe},
    },
};

mod epf_sse41;
mod epf_avx2;
mod gabor_avx2;

pub fn epf<const STEP: usize>(
    input: &mut [MutableSubgrid<f32>; 3],
    output: &mut [MutableSubgrid<f32>; 3],
    color_padded_region: Region,
    frame_header: &FrameHeader,
    sigma_grid_map: &[Option<&AlignedGrid<f32>>],
    epf_params: &EpfParams,
    pool: &JxlThreadPool,
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Features are checked above.
        unsafe {
            return run_epf_rows(
                input,
                output,
                color_padded_region,
                frame_header,
                sigma_grid_map,
                epf_params,
                pool,
                Some(epf_avx2::epf_row_x86_64_avx2::<STEP>),
                super::generic::epf::epf_row::<STEP>,
            );
        }
    }

    if is_x86_feature_detected!("sse4.1") {
        // SAFETY: Features are checked above.
        unsafe {
            return run_epf_rows(
                input,
                output,
                color_padded_region,
                frame_header,
                sigma_grid_map,
                epf_params,
                pool,
                Some(epf_sse41::epf_row_x86_64_sse41::<STEP>),
                super::generic::epf::epf_row::<STEP>,
            );
        }
    }

    unsafe {
        run_epf_rows(
            input,
            output,
            color_padded_region,
            frame_header,
            sigma_grid_map,
            epf_params,
            pool,
            None,
            super::generic::epf::epf_row::<STEP>,
        )
    }
}

pub fn apply_gabor_like(
    fb: [MutableSubgrid<f32>; 3],
    fb_scratch: &mut [AlignedGrid<f32>; 3],
    weights: [[f32; 2]; 3],
    pool: &crate::jxl_decode::jxl_threadpool::JxlThreadPool,
) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Features are checked above. run_gabor_3ch_rows_unsafe processes all
        // 3 channels in a single pool::for_each_vec, eliminating 2 sequential barriers.
        unsafe {
            run_gabor_3ch_rows_unsafe(
                fb,
                fb_scratch,
                weights,
                pool,
                gabor_avx2::run_gabor_row_x86_64_avx2,
            );
        }
        return;
    }

    for ((input, output), weights) in fb.into_iter().zip(fb_scratch).zip(weights) {
        run_gabor_rows(input, output, weights, pool, run_gabor_row_generic);
    }
}

#[cfg(test)]
mod avx2_epf_tests {
    use crate::jxl_decode::jxl_frame::filter::EpfParams;
    use crate::jxl_decode::jxl_grid::SharedSubgrid;
    use crate::jxl_decode::jxl_render::filter::epf::EpfRow;

    const WIDTH: usize = 48;
    const ROWS: usize = 7;

    // Deterministic smooth-ish data in a narrow range so EPF weights are non-degenerate
    // (not universally clamped to zero), exercising the full arithmetic path.
    fn make_channel(seed: u64) -> Vec<f32> {
        let mut s = seed ^ 0x9e3779b97f4a7c15;
        let mut out = Vec::with_capacity(ROWS * WIDTH);
        for _ in 0..ROWS * WIDTH {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let noise = ((s >> 40) as f32) / ((1u64 << 24) as f32); // 0..1
            out.push(0.5 + 0.02 * (noise - 0.5));
        }
        out
    }

    fn rows_of<'a>(buf: &'a [f32]) -> [&'a [f32]; ROWS] {
        std::array::from_fn(|r| &buf[r * WIDTH..r * WIDTH + WIDTH])
    }

    fn run_case<const STEP: usize>(y: usize, sigma_row: &[f32]) {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }

        let params = EpfParams::default();
        let chans: [Vec<f32>; 3] = std::array::from_fn(|c| make_channel(c as u64 + 1));

        // Reference: whole row via generic scalar kernel (skip_inner = false).
        let mut ref_out: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; WIDTH]);
        {
            let input_rows: [[&[f32]; ROWS]; 3] = std::array::from_fn(|c| rows_of(&chans[c]));
            let [r0, r1, r2] = ref_out.each_mut();
            let output_rows = [r0.as_mut_slice(), r1.as_mut_slice(), r2.as_mut_slice()];
            let row = EpfRow {
                input_rows,
                merged_input_rows: None,
                output_rows,
                width: WIDTH,
                y,
                sigma_row,
                epf_params: &params,
                skip_inner: false,
            };
            super::super::generic::epf::epf_row::<STEP>(row);
        }

        // Test: AVX2 fills inner region, then generic fills borders (skip_inner = true).
        let mut test_out: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; WIDTH]);
        {
            let merged: [SharedSubgrid<f32>; 3] =
                std::array::from_fn(|c| SharedSubgrid::from_buf(&chans[c], WIDTH, ROWS, WIDTH));
            let input_rows: [[&[f32]; ROWS]; 3] = std::array::from_fn(|c| rows_of(&chans[c]));
            let [t0, t1, t2] = test_out.each_mut();
            let output_rows = [t0.as_mut_slice(), t1.as_mut_slice(), t2.as_mut_slice()];
            let row = EpfRow {
                input_rows,
                merged_input_rows: Some(merged),
                output_rows,
                width: WIDTH,
                y,
                sigma_row,
                epf_params: &params,
                skip_inner: true,
            };
            // SAFETY: AVX2 + FMA checked above.
            unsafe { super::epf_avx2::epf_row_x86_64_avx2::<STEP>(row) };
        }
        {
            let input_rows: [[&[f32]; ROWS]; 3] = std::array::from_fn(|c| rows_of(&chans[c]));
            let [t0, t1, t2] = test_out.each_mut();
            let output_rows = [t0.as_mut_slice(), t1.as_mut_slice(), t2.as_mut_slice()];
            let row = EpfRow {
                input_rows,
                merged_input_rows: None,
                output_rows,
                width: WIDTH,
                y,
                sigma_row,
                epf_params: &params,
                skip_inner: true,
            };
            super::super::generic::epf::epf_row::<STEP>(row);
        }

        for c in 0..3 {
            for x in 0..WIDTH {
                let a = ref_out[c][x];
                let b = test_out[c][x];
                assert!(
                    (a - b).abs() <= 1e-4,
                    "STEP={STEP} y={y} c={c} x={x}: generic={a} avx2={b} diff={}",
                    (a - b).abs()
                );
            }
        }
    }

    fn sigma_all(v: f32) -> Vec<f32> {
        vec![v; WIDTH.div_ceil(8)]
    }

    fn sigma_mixed() -> Vec<f32> {
        // Alternate active / early-out (< 0.3) blocks.
        (0..WIDTH.div_ceil(8))
            .map(|i| if i % 2 == 0 { 1.7 } else { 0.1 })
            .collect()
    }

    #[test]
    fn avx2_matches_generic_step0() {
        run_case::<0>(4, &sigma_all(1.7));
        run_case::<0>(7, &sigma_all(1.7)); // y-border case
        run_case::<0>(5, &sigma_mixed());
    }

    #[test]
    fn avx2_matches_generic_step1() {
        run_case::<1>(4, &sigma_all(1.7));
        run_case::<1>(7, &sigma_all(1.7));
        run_case::<1>(5, &sigma_mixed());
    }

    #[test]
    fn avx2_matches_generic_step2() {
        run_case::<2>(4, &sigma_all(1.7));
        run_case::<2>(7, &sigma_all(1.7));
        run_case::<2>(5, &sigma_mixed());
    }
}
