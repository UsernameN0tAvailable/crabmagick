pub(crate) fn run(xyb: [&mut [f32]; 3], ob: [f32; 3], intensity_target: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
            // SAFETY: Feature set is checked above.
            return unsafe { run_x86_64_avx2(xyb, ob, intensity_target) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: Feature set is checked above.
            return unsafe { run_aarch64_neon(xyb, ob, intensity_target) };
        }
    }

    run_generic(xyb, ob, intensity_target)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn run_x86_64_avx2(xyb: [&mut [f32]; 3], ob: [f32; 3], intensity_target: f32) {
    use std::arch::x86_64::*;

    let itscale = 255.0 / intensity_target;
    let [xc, yc, bc] = xyb;
    let len = xc.len();
    if len != yc.len() || len != bc.len() {
        panic!("Grid size mismatch");
    }

    let cbrt_ob = ob.map(|v| v.cbrt());

    // Pre-broadcast constants
    let vcbrt0 = _mm256_set1_ps(cbrt_ob[0]);
    let vcbrt1 = _mm256_set1_ps(cbrt_ob[1]);
    let vcbrt2 = _mm256_set1_ps(cbrt_ob[2]);
    // (v^3 + ob) * itscale = v^3 * itscale + ob*itscale → use FMA: fmadd(v^3, itscale, ob_scaled)
    let vob0 = _mm256_set1_ps(ob[0] * itscale);
    let vob1 = _mm256_set1_ps(ob[1] * itscale);
    let vob2 = _mm256_set1_ps(ob[2] * itscale);
    let vits = _mm256_set1_ps(itscale);

    let px = xc.as_mut_ptr();
    let py = yc.as_mut_ptr();
    let pb = bc.as_mut_ptr();

    let mut i = 0usize;
    while i + 8 <= len {
        let xv = _mm256_loadu_ps(px.add(i));
        let yv = _mm256_loadu_ps(py.add(i));
        let bv = _mm256_loadu_ps(pb.add(i));

        // matrix: g_l = y+x, g_m = y-x, g_s = b
        let g_l = _mm256_sub_ps(_mm256_add_ps(yv, xv), vcbrt0);
        let g_m = _mm256_sub_ps(_mm256_sub_ps(yv, xv), vcbrt1);
        let g_s = _mm256_sub_ps(bv, vcbrt2);

        // cube: v^3 = v*v*v, then (v^3)*itscale + ob*itscale
        let gl2 = _mm256_mul_ps(g_l, g_l);
        let gm2 = _mm256_mul_ps(g_m, g_m);
        let gs2 = _mm256_mul_ps(g_s, g_s);
        let gl3 = _mm256_mul_ps(gl2, g_l);
        let gm3 = _mm256_mul_ps(gm2, g_m);
        let gs3 = _mm256_mul_ps(gs2, g_s);

        let rx = _mm256_fmadd_ps(gl3, vits, vob0);
        let ry = _mm256_fmadd_ps(gm3, vits, vob1);
        let rb = _mm256_fmadd_ps(gs3, vits, vob2);

        _mm256_storeu_ps(px.add(i), rx);
        _mm256_storeu_ps(py.add(i), ry);
        _mm256_storeu_ps(pb.add(i), rb);
        i += 8;
    }

    // Scalar tail
    while i < len {
        let xi = *px.add(i);
        let yi = *py.add(i);
        let bi = *pb.add(i);
        let g_l = yi + xi - cbrt_ob[0];
        let g_m = yi - xi - cbrt_ob[1];
        let g_s = bi - cbrt_ob[2];
        *px.add(i) = (g_l * g_l).mul_add(g_l, ob[0]) * itscale;
        *py.add(i) = (g_m * g_m).mul_add(g_m, ob[1]) * itscale;
        *pb.add(i) = (g_s * g_s).mul_add(g_s, ob[2]) * itscale;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn run_aarch64_neon(xyb: [&mut [f32]; 3], ob: [f32; 3], intensity_target: f32) {
    run_generic(xyb, ob, intensity_target)
}

#[inline(always)]
fn run_generic(xyb: [&mut [f32]; 3], ob: [f32; 3], intensity_target: f32) {
    let itscale = 255.0 / intensity_target;

    let [x, y, b] = xyb;
    if x.len() != y.len() || y.len() != b.len() {
        panic!("Grid size mismatch");
    }
    let cbrt_ob = ob.map(|v| v.cbrt());

    for ((x, y), b) in x.iter_mut().zip(&mut *y).zip(&mut *b) {
        // matrix: [1, 1, 0, -1, 1, 0, 0, 0, 1]
        let g_l = *y + *x;
        let g_m = *y - *x;
        let g_s = *b;

        // bias: -cbrt_ob
        let g_l = g_l - cbrt_ob[0];
        let g_m = g_m - cbrt_ob[1];
        let g_s = g_s - cbrt_ob[2];

        // inverse tf: gamma3, bias: ob, matrix: id * itscale
        *x = (g_l * g_l).mul_add(g_l, ob[0]) * itscale;
        *y = (g_m * g_m).mul_add(g_m, ob[1]) * itscale;
        *b = (g_s * g_s).mul_add(g_s, ob[2]) * itscale;
    }
}
