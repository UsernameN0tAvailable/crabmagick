use std::arch::x86_64::*;

use crate::jxl_decode::jxl_render::filter::gabor::GaborRow;

// 3×3 stencil convolution, 8-wide AVX2+FMA:
//   out[x] = (cc + (tc + cl + cr + bc)*w0 + (tl + tr + bl + br)*w1) * global_weight
// where t/c/b = top/center/bottom row, l/c/r = left/center/right column.
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
pub(super) unsafe fn run_gabor_row_x86_64_avx2(row: GaborRow) {
    let GaborRow {
        input_rows: [input_row_t, input_row_c, input_row_b],
        output_row,
        weights,
    } = row;
    let width = output_row.len();
    assert_eq!(input_row_t.len(), width);
    assert_eq!(input_row_c.len(), width);
    assert_eq!(input_row_b.len(), width);

    if width == 0 {
        return;
    }

    let [w0, w1] = weights;
    let global_weight = (1.0 + w0 * 4.0 + w1 * 4.0).recip();

    if width == 1 {
        let t = input_row_t[0];
        let c = input_row_c[0];
        let b = input_row_b[0];
        let sum_side = t + 2.0 * c + b;
        let sum_diag = 2.0 * (t + b);
        output_row[0] = (c + sum_side * w0 + sum_diag * w1) * global_weight;
        return;
    }

    let pt = input_row_t.as_ptr();
    let pc = input_row_c.as_ptr();
    let pb = input_row_b.as_ptr();
    let po = output_row.as_mut_ptr();

    // Left edge (scalar, x=0)
    {
        let t1 = *pt; let c1 = *pc; let b1 = *pb;
        let t0 = *pt.add(1); let c0 = *pc.add(1); let b0 = *pb.add(1);
        let sum_side = t1 + c0 + c1 + b1;
        let sum_diag = t0 + t1 + b0 + b1;
        *po = (c1 + sum_side * w0 + sum_diag * w1) * global_weight;
    }

    // Inner region: x = 1..width-1. Process 8 at a time.
    // inner_len = width - 2 pixels at x=1..width-2 (inclusive).
    // Guard: ix+8 <= inner_len ensures top[x+1+7] = top[ix+9] <= top[width-1].
    let vw0 = _mm256_set1_ps(w0);
    let vw1 = _mm256_set1_ps(w1);
    let vgw = _mm256_set1_ps(global_weight);
    let inner_len = width.saturating_sub(2);
    let mut ix = 0usize;
    while ix + 8 <= inner_len {
        let x = ix + 1;
        let tl = _mm256_loadu_ps(pt.add(x - 1));
        let tc = _mm256_loadu_ps(pt.add(x));
        let tr = _mm256_loadu_ps(pt.add(x + 1));
        let cl = _mm256_loadu_ps(pc.add(x - 1));
        let cc = _mm256_loadu_ps(pc.add(x));
        let cr = _mm256_loadu_ps(pc.add(x + 1));
        let bl = _mm256_loadu_ps(pb.add(x - 1));
        let bc = _mm256_loadu_ps(pb.add(x));
        let br = _mm256_loadu_ps(pb.add(x + 1));
        // sum_side = tc + cl + cr + bc
        let sum_side = _mm256_add_ps(_mm256_add_ps(tc, cl), _mm256_add_ps(cr, bc));
        // sum_diag = tl + tr + bl + br
        let sum_diag = _mm256_add_ps(_mm256_add_ps(tl, tr), _mm256_add_ps(bl, br));
        // out = (cc + sum_side*w0 + sum_diag*w1) * gw
        let unweighted = _mm256_fmadd_ps(sum_diag, vw1, _mm256_fmadd_ps(sum_side, vw0, cc));
        _mm256_storeu_ps(po.add(x), _mm256_mul_ps(unweighted, vgw));
        ix += 8;
    }
    // Scalar remainder of inner region
    while ix < inner_len {
        let x = ix + 1;
        let t0 = *pt.add(x - 1); let t1 = *pt.add(x); let t2 = *pt.add(x + 1);
        let c0 = *pc.add(x - 1); let c1 = *pc.add(x); let c2 = *pc.add(x + 1);
        let b0 = *pb.add(x - 1); let b1 = *pb.add(x); let b2 = *pb.add(x + 1);
        let sum_side = t1 + c0 + c2 + b1;
        let sum_diag = t0 + t2 + b0 + b2;
        *po.add(x) = (c1 + sum_side * w0 + sum_diag * w1) * global_weight;
        ix += 1;
    }

    // Right edge (scalar, x=width-1)
    if width >= 2 {
        let x = width - 1;
        let t1 = *pt.add(x); let c1 = *pc.add(x); let b1 = *pb.add(x);
        let t0 = *pt.add(x - 1); let c0 = *pc.add(x - 1); let b0 = *pb.add(x - 1);
        let sum_side = t1 + c0 + c1 + b1;
        let sum_diag = t0 + t1 + b0 + b1;
        *po.add(x) = (c1 + sum_side * w0 + sum_diag * w1) * global_weight;
    }
}
