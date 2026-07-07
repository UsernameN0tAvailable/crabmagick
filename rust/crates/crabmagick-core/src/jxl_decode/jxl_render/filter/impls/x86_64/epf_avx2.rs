use std::arch::x86_64::*;

use crate::jxl_decode::jxl_grid::SimdVector;

use crate::jxl_decode::jxl_render::filter::epf::*;

type Vector = __m256;

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn weight_avx2(scaled_distance: Vector, sigma: f32, step_multiplier: Vector) -> Vector {
    let neg_inv_sigma = Vector::splat_f32(6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / sigma)
        .mul(step_multiplier);
    let result = scaled_distance.mul(neg_inv_sigma).add(_mm256_set1_ps(1.0));
    _mm256_max_ps(result, Vector::zero())
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn epf_row_x86_64_avx2<const STEP: usize>(epf_row: EpfRow) {
    let EpfRow {
        merged_input_rows,
        output_rows,
        width,
        y,
        sigma_row,
        epf_params,
        ..
    } = epf_row;
    let merged_input_rows = merged_input_rows.unwrap();
    let kernel_offsets = epf_kernel_offsets::<STEP>();

    let step_multiplier = if STEP == 0 {
        epf_params.sigma.pass0_sigma_scale
    } else if STEP == 2 {
        epf_params.sigma.pass2_sigma_scale
    } else {
        1.0
    };
    let border_sad_mul = epf_params.sigma.border_sad_mul;
    let channel_scale = epf_params.channel_scale;

    let padding = 3 - STEP;
    if width < padding * 2 {
        return;
    }

    let simd_range = {
        let start = 8;
        let end = (width - padding) & !7;
        if start > end {
            start..start
        } else {
            start..end
        }
    };

    // Each 8-wide vector spans exactly one 8-pixel sigma block, so border_sad_mul applies to both
    // the first (lane 0) and last (lane 7) pixel of the block, matching the generic implementation.
    let is_y_border = (y + 1) & 0b110 == 0;
    let sm = if is_y_border {
        _mm256_set1_ps(step_multiplier * border_sad_mul)
    } else {
        _mm256_set_ps(
            step_multiplier * border_sad_mul,
            step_multiplier,
            step_multiplier,
            step_multiplier,
            step_multiplier,
            step_multiplier,
            step_multiplier,
            step_multiplier * border_sad_mul,
        )
    };

    for dx in simd_range.step_by(8) {
        let sigma_x = dx / 8;

        let sigma_val = sigma_row[sigma_x];

        let originals: [_; 3] = std::array::from_fn(|c| unsafe {
            _mm256_loadu_ps(merged_input_rows[c].get_row(3).as_ptr().add(dx))
        });

        if sigma_val < 0.3 {
            for (c, val) in originals.into_iter().enumerate() {
                _mm256_storeu_ps(output_rows[c].as_mut_ptr().add(dx), val);
            }
            continue;
        }

        let mut sum_weights = _mm256_set1_ps(1.0);
        let mut sum_channels = originals;

        if STEP == 1 {
            // (0, -1), (0, 1)
            {
                let mut dist0 = _mm256_setzero_ps();
                let mut dist1 = _mm256_setzero_ps();
                for c in 0..3 {
                    let scale = _mm256_set1_ps(channel_scale[c]);
                    let input_rows = &merged_input_rows[c];

                    let v0 = _mm256_loadu_ps(input_rows.get_row(1).as_ptr().add(dx));
                    let v1 = _mm256_loadu_ps(input_rows.get_row(2).as_ptr().add(dx));
                    let v2 = _mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx));
                    let v3 = _mm256_loadu_ps(input_rows.get_row(4).as_ptr().add(dx));
                    let v4 = _mm256_loadu_ps(input_rows.get_row(5).as_ptr().add(dx));
                    let tmp = v1.sub(v2).abs().add(v3.sub(v2).abs());
                    let mut acc0 = tmp.add(v1.sub(v0).abs());
                    let mut acc1 = tmp.add(v3.sub(v4).abs());

                    let v1_left = _mm256_loadu_ps(input_rows.get_row(2).as_ptr().add(dx - 1));
                    let v2_left = _mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx - 1));
                    let v3_left = _mm256_loadu_ps(input_rows.get_row(4).as_ptr().add(dx - 1));
                    acc0 = acc0.add(v1_left.sub(v2_left).abs());
                    acc1 = acc1.add(v3_left.sub(v2_left).abs());

                    let v1_right = _mm256_loadu_ps(input_rows.get_row(2).as_ptr().add(dx + 1));
                    let v2_right = _mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx + 1));
                    let v3_right = _mm256_loadu_ps(input_rows.get_row(4).as_ptr().add(dx + 1));
                    acc0 = acc0.add(v1_right.sub(v2_right).abs());
                    acc1 = acc1.add(v3_right.sub(v2_right).abs());

                    dist0 = scale.mul(acc0).add(dist0);
                    dist1 = scale.mul(acc1).add(dist1);
                }

                let weight0 = weight_avx2(dist0, sigma_val, sm);
                let weight1 = weight_avx2(dist1, sigma_val, sm);
                sum_weights = sum_weights.add(weight0).add(weight1);

                for (c, sum) in sum_channels.iter_mut().enumerate() {
                    let input_rows = &merged_input_rows[c];
                    let weighted0 =
                        weight0.mul(_mm256_loadu_ps(input_rows.get_row(2).as_ptr().add(dx)));
                    let weighted1 =
                        weight1.mul(_mm256_loadu_ps(input_rows.get_row(4).as_ptr().add(dx)));
                    *sum = sum.add(weighted0).add(weighted1);
                }
            }

            // (-1, 0), (1, 0)
            {
                let mut dist0 = _mm256_setzero_ps();
                let mut dist1 = _mm256_setzero_ps();
                for c in 0..3 {
                    let scale = _mm256_set1_ps(channel_scale[c]);
                    let input_rows = &merged_input_rows[c];

                    let v0r = _mm256_loadu_ps(input_rows.get_row(2).as_ptr().add(dx + 1));
                    let v0 = _mm256_loadu_ps(input_rows.get_row(2).as_ptr().add(dx));
                    let v0l = _mm256_loadu_ps(input_rows.get_row(2).as_ptr().add(dx - 1));
                    let mut acc0 = v0l.sub(v0).abs();
                    let mut acc1 = v0r.sub(v0).abs();

                    let v1rr = _mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx + 2));
                    let v1r = _mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx + 1));
                    let v1 = _mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx));
                    let v1l = _mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx - 1));
                    let v1ll = _mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx - 2));
                    acc0 = acc0.add(v1ll.sub(v1l).abs());
                    acc0 = acc0.add(v1.sub(v1l).abs());
                    acc0 = acc0.add(v1.sub(v1r).abs());
                    acc1 = acc1.add(v1.sub(v1l).abs());
                    acc1 = acc1.add(v1.sub(v1r).abs());
                    acc1 = acc1.add(v1rr.sub(v1r).abs());

                    let v2r = _mm256_loadu_ps(input_rows.get_row(4).as_ptr().add(dx + 1));
                    let v2 = _mm256_loadu_ps(input_rows.get_row(4).as_ptr().add(dx));
                    let v2l = _mm256_loadu_ps(input_rows.get_row(4).as_ptr().add(dx - 1));
                    acc0 = acc0.add(v2l.sub(v2).abs());
                    acc1 = acc1.add(v2r.sub(v2).abs());

                    dist0 = scale.mul(acc0).add(dist0);
                    dist1 = scale.mul(acc1).add(dist1);
                }

                let weight0 = weight_avx2(dist0, sigma_val, sm);
                let weight1 = weight_avx2(dist1, sigma_val, sm);
                sum_weights = sum_weights.add(weight0).add(weight1);

                for (c, sum) in sum_channels.iter_mut().enumerate() {
                    let input_rows = &merged_input_rows[c];
                    let weighted0 =
                        weight0.mul(_mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx - 1)));
                    let weighted1 =
                        weight1.mul(_mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx + 1)));
                    *sum = sum.add(weighted0).add(weighted1);
                }
            }
        } else {
            for &(kx, ky) in kernel_offsets {
                let ky = 3usize.wrapping_add_signed(ky);
                let kx = dx.wrapping_add_signed(kx);
                let mut dist = _mm256_setzero_ps();
                for c in 0..3 {
                    let scale = _mm256_set1_ps(channel_scale[c]);
                    let input_rows = &merged_input_rows[c];
                    if STEP == 0 {
                        let vk0 = _mm256_loadu_ps(input_rows.get_row(ky - 1).as_ptr().add(kx));
                        let vb0 = _mm256_loadu_ps(input_rows.get_row(2).as_ptr().add(dx));
                        let mut acc = vk0.sub(vb0).abs();

                        let vk1r = _mm256_loadu_ps(input_rows.get_row(ky).as_ptr().add(kx + 1));
                        let vb1r = _mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx + 1));
                        acc = acc.add(vk1r.sub(vb1r).abs());

                        let vk1 = _mm256_loadu_ps(input_rows.get_row(ky).as_ptr().add(kx));
                        let vb1 = _mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx));
                        acc = acc.add(vk1.sub(vb1).abs());

                        let vk1l = _mm256_loadu_ps(input_rows.get_row(ky).as_ptr().add(kx - 1));
                        let vb1l = _mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx - 1));
                        acc = acc.add(vk1l.sub(vb1l).abs());

                        let vk2 = _mm256_loadu_ps(input_rows.get_row(ky + 1).as_ptr().add(kx));
                        let vb2 = _mm256_loadu_ps(input_rows.get_row(4).as_ptr().add(dx));
                        acc = acc.add(vk2.sub(vb2).abs());

                        dist = scale.mul(acc).add(dist);
                    } else {
                        let v0 = _mm256_loadu_ps(input_rows.get_row(ky).as_ptr().add(kx));
                        let v1 = _mm256_loadu_ps(input_rows.get_row(3).as_ptr().add(dx));
                        dist = scale.mul(v0.sub(v1).abs()).add(dist);
                    }
                }

                let weight = weight_avx2(dist, sigma_val, sm);
                sum_weights = sum_weights.add(weight);

                for (c, sum) in sum_channels.iter_mut().enumerate() {
                    *sum = weight
                        .mul(_mm256_loadu_ps(
                            merged_input_rows[c].get_row(ky).as_ptr().add(kx),
                        ))
                        .add(*sum);
                }
            }
        }

        for (c, sum) in sum_channels.into_iter().enumerate() {
            _mm256_storeu_ps(output_rows[c].as_mut_ptr().add(dx), sum.div(sum_weights));
        }
    }
}
