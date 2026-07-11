#![allow(unsafe_op_in_unsafe_fn)]

use std::{cell::RefCell, collections::HashMap};

#[cfg(feature = "__profile")]
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use crate::jxl_decode::jxl_frame::{
    FrameHeader,
    data::{
        HfGlobal, LfGlobal, LfGroup, PassGroupParams, PassGroupParamsCompact,
        PassGroupParamsVardct, PassGroupParamsVardctCompact, decode_hf_coeff_direct,
        decode_pass_group_compact,
    },
};
use crate::jxl_decode::jxl_grid::{AlignedGrid, MutableSubgrid, SharedSubgrid};
use crate::jxl_decode::jxl_image::ImageHeader;
use crate::jxl_decode::jxl_modular::{ChannelShift, Sample};
use crate::jxl_decode::jxl_threadpool::JxlThreadPool;
use crate::jxl_decode::jxl_vardct::{
    BlockInfo, CompactHfCoeffStore, DequantMatrixSet, LfChannelCorrelation,
    LfChannelDequantization, Quantizer, TransformType,
};

use crate::jxl_decode::jxl_render::{
    Error, ImageWithRegion, IndexedFrame, Reference, Region, RenderCache, Result,
    image::ImageBuffer, modular, util,
};

mod dct_common;
mod transform_common;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
use x86_64 as impls;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
use aarch64 as impls;

#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
mod wasm32;
#[cfg(all(target_family = "wasm", target_feature = "simd128"))]
use wasm32 as impls;

mod generic;
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_family = "wasm", target_feature = "simd128")
)))]
use generic as impls;

// `perf` is not usable in several deployment targets. This compiles out of normal builds and
// records parallel worker phases when the internal `__profile` feature is explicitly enabled.
#[cfg(feature = "__profile")]
#[derive(Debug)]
struct VardctProfile {
    started: Instant,
    load_lf_ns: AtomicU64,
    prepare_groups_ns: AtomicU64,
    task_setup_ns: AtomicU64,
    workers_wall_ns: AtomicU64,
    task_ns_sum: AtomicU64,
    task_ns_max: AtomicU64,
    task_count: AtomicU64,
    compact_alloc_ns: AtomicU64,
    hf_decode_ns: AtomicU64,
    dct8_transform_ns: AtomicU64,
    generic_transform_ns: AtomicU64,
    generic_dequant_ns: AtomicU64,
    generic_idct_ns: AtomicU64,
    generic_scatter_ns: AtomicU64,
    fallback_transform_ns: AtomicU64,
}

#[cfg(feature = "__profile")]
const TRANSFORM_TYPE_NAMES: [&str; 27] = [
    "Dct8",
    "Hornuss",
    "Dct2",
    "Dct4",
    "Dct16",
    "Dct32",
    "Dct16x8",
    "Dct8x16",
    "Dct32x8",
    "Dct8x32",
    "Dct32x16",
    "Dct16x32",
    "Dct4x8",
    "Dct8x4",
    "Afv0",
    "Afv1",
    "Afv2",
    "Afv3",
    "Dct64",
    "Dct64x32",
    "Dct32x64",
    "Dct128",
    "Dct128x64",
    "Dct64x128",
    "Dct256",
    "Dct256x128",
    "Dct128x256",
];

#[cfg(feature = "__profile")]
static PROFILE_TRANSFORM_COUNTS: [AtomicU64; TRANSFORM_TYPE_NAMES.len()] =
    [const { AtomicU64::new(0) }; TRANSFORM_TYPE_NAMES.len()];

#[cfg(feature = "__profile")]
type VardctProfileHandle = Option<Arc<VardctProfile>>;

#[cfg(not(feature = "__profile"))]
type VardctProfileHandle = ();

#[cfg(feature = "__profile")]
fn start_vardct_profile() -> VardctProfileHandle {
    std::env::var_os("CRABMAGICK_JXL_PROFILE").map(|_| {
        for counter in &PROFILE_TRANSFORM_COUNTS {
            counter.store(0, Ordering::Relaxed);
        }
        Arc::new(VardctProfile {
            started: Instant::now(),
            load_lf_ns: AtomicU64::new(0),
            prepare_groups_ns: AtomicU64::new(0),
            task_setup_ns: AtomicU64::new(0),
            workers_wall_ns: AtomicU64::new(0),
            task_ns_sum: AtomicU64::new(0),
            task_ns_max: AtomicU64::new(0),
            task_count: AtomicU64::new(0),
            compact_alloc_ns: AtomicU64::new(0),
            hf_decode_ns: AtomicU64::new(0),
            dct8_transform_ns: AtomicU64::new(0),
            generic_transform_ns: AtomicU64::new(0),
            generic_dequant_ns: AtomicU64::new(0),
            generic_idct_ns: AtomicU64::new(0),
            generic_scatter_ns: AtomicU64::new(0),
            fallback_transform_ns: AtomicU64::new(0),
        })
    })
}

#[cfg(not(feature = "__profile"))]
#[inline(always)]
fn start_vardct_profile() -> VardctProfileHandle {}

#[cfg(feature = "__profile")]
fn finish_vardct_profile(profile: &VardctProfileHandle) {
    let Some(profile) = profile else {
        return;
    };
    let ms = |value: &AtomicU64| value.load(Ordering::Relaxed) as f64 / 1_000_000.0;
    let transforms = TRANSFORM_TYPE_NAMES
        .iter()
        .zip(&PROFILE_TRANSFORM_COUNTS)
        .filter_map(|(name, count)| {
            let count = count.load(Ordering::Relaxed);
            (count != 0).then_some(format!("{name}={count}"))
        })
        .collect::<Vec<_>>()
        .join(",");
    let task_count = profile.task_count.load(Ordering::Relaxed);
    let task_avg = if task_count == 0 {
        0.0
    } else {
        ms(&profile.task_ns_sum) / task_count as f64
    };
    eprintln!(
        "CRABMAGICK_JXL_VARDCT wall={:.2}ms load_lf={:.2}ms prepare={:.2}ms task_setup={:.2}ms workers_wall={:.2}ms tasks={task_count} task_avg={task_avg:.2}ms task_max={:.2}ms compact_alloc_sum={:.2}ms hf_decode_sum={:.2}ms dct8_sum={:.2}ms generic_sum={:.2}ms generic_dequant_sum={:.2}ms generic_idct_sum={:.2}ms generic_scatter_sum={:.2}ms fallback_sum={:.2}ms transforms={transforms}",
        profile.started.elapsed().as_secs_f64() * 1_000.0,
        ms(&profile.load_lf_ns),
        ms(&profile.prepare_groups_ns),
        ms(&profile.task_setup_ns),
        ms(&profile.workers_wall_ns),
        ms(&profile.task_ns_max),
        ms(&profile.compact_alloc_ns),
        ms(&profile.hf_decode_ns),
        ms(&profile.dct8_transform_ns),
        ms(&profile.generic_transform_ns),
        ms(&profile.generic_dequant_ns),
        ms(&profile.generic_idct_ns),
        ms(&profile.generic_scatter_ns),
        ms(&profile.fallback_transform_ns),
    );
}

#[cfg(not(feature = "__profile"))]
#[inline(always)]
fn finish_vardct_profile(_: &VardctProfileHandle) {}

#[cfg(feature = "__profile")]
struct VardctTaskTimer {
    profile: Option<Arc<VardctProfile>>,
    started: Instant,
}

#[cfg(feature = "__profile")]
impl Drop for VardctTaskTimer {
    fn drop(&mut self) {
        let Some(profile) = &self.profile else {
            return;
        };
        let elapsed = self.started.elapsed().as_nanos() as u64;
        profile.task_ns_sum.fetch_add(elapsed, Ordering::Relaxed);
        profile.task_ns_max.fetch_max(elapsed, Ordering::Relaxed);
        profile.task_count.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(feature = "__profile")]
fn start_vardct_task_timer(profile: &VardctProfileHandle) -> VardctTaskTimer {
    VardctTaskTimer {
        profile: profile.clone(),
        started: Instant::now(),
    }
}

#[cfg(not(feature = "__profile"))]
#[inline(always)]
fn start_vardct_task_timer(_: &VardctProfileHandle) {}

#[cfg(feature = "__profile")]
macro_rules! profile_stage {
    ($profile:expr, $field:ident, $body:expr) => {{
        if let Some(profile) = &$profile {
            let start = Instant::now();
            let value = $body;
            profile
                .$field
                .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            value
        } else {
            $body
        }
    }};
}

#[cfg(not(feature = "__profile"))]
macro_rules! profile_stage {
    ($profile:expr, $field:ident, $body:expr) => {{ $body }};
}

pub(crate) fn render_vardct<S: Sample>(
    frame: &IndexedFrame,
    lf_frame: Option<&Reference<S>>,
    cache: &mut RenderCache<S>,
    region: Region,
    pool: &JxlThreadPool,
) -> Result<ImageWithRegion> {
    let span = tracing::span!(tracing::Level::TRACE, "Render VarDCT");
    let _guard = span.enter();
    let profile = start_vardct_profile();

    let image_header = frame.image_header();
    let frame_header = frame.header();
    let tracker = frame.alloc_tracker();

    let jpeg_upsampling = frame_header.jpeg_upsampling;
    let subsampled = jpeg_upsampling.into_iter().any(|x| x != 0);

    let lf_global = match &cache.lf_global {
        Some(x) if !x.gmodular.is_partial() => x,
        _ => {
            let lf_global = frame
                .try_parse_lf_global()
                .ok_or(Error::IncompleteFrame)??;
            cache.lf_global = Some(lf_global);
            cache.lf_global.as_ref().unwrap()
        }
    };
    if lf_frame.is_some() && lf_global.gmodular.is_partial() {
        return Err(Error::IncompleteFrame);
    }

    let mut gmodular = lf_global.gmodular.try_clone()?;
    let lf_global_vardct = lf_global.vardct.as_ref().unwrap();

    let width = frame_header.color_sample_width() as usize;
    let height = frame_header.color_sample_height() as usize;
    let (width_rounded, height_rounded) = {
        let mut bw = width.div_ceil(8);
        let mut bh = height.div_ceil(8);
        let h_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 2);
        let v_upsample = jpeg_upsampling.into_iter().any(|j| j == 1 || j == 3);
        if h_upsample {
            bw = bw.div_ceil(2) * 2;
        }
        if v_upsample {
            bh = bh.div_ceil(2) * 2;
        }
        (bw * 8, bh * 8)
    };

    let aligned_region = region.container_aligned(frame_header.group_dim());
    let aligned_lf_region = {
        // group_dim is multiple of 8
        let aligned_region_div8 = Region {
            left: aligned_region.left / 8,
            top: aligned_region.top / 8,
            width: aligned_region.width / 8,
            height: aligned_region.height / 8,
        };
        if frame_header.flags.skip_adaptive_lf_smoothing() {
            aligned_region_div8
        } else {
            aligned_region_div8.pad(1)
        }
        .container_aligned(frame_header.group_dim())
    };

    let aligned_region = aligned_region.intersection(Region::with_size(
        width_rounded as u32,
        height_rounded as u32,
    ));
    let aligned_lf_region = aligned_lf_region.intersection(Region::with_size(
        width_rounded as u32 / 8,
        height_rounded as u32 / 8,
    ));
    let modular_region =
        modular::compute_modular_region(frame_header, &gmodular, aligned_region, false);
    let modular_lf_region =
        modular::compute_modular_region(frame_header, &gmodular, aligned_lf_region, true)
            .intersection(Region::with_size(
                width_rounded as u32 / 8,
                height_rounded as u32 / 8,
            ));

    let mut modular_image = gmodular.modular.image_mut();
    let groups = modular_image
        .as_mut()
        .map(|x| x.prepare_groups(frame.pass_shifts()))
        .transpose()?;
    let (lf_group_image, pass_group_image) = groups.map(|x| (x.lf_groups, x.pass_groups)).unzip();
    let lf_group_image = lf_group_image.unwrap_or_else(Vec::new);
    let pass_group_image = pass_group_image.unwrap_or_else(|| {
        let passes = frame_header.passes.num_passes as usize;
        let mut ret = Vec::with_capacity(passes);
        ret.resize_with(passes, Vec::new);
        ret
    });

    let hf_global = &mut cache.hf_global;
    let lf_groups = &mut cache.lf_groups;
    let group_dim = frame_header.group_dim();

    let result = std::sync::RwLock::new(Result::Ok(()));
    let (mut fb, lf_xyb) = pool.scope(|scope| -> Result<_> {
        if hf_global.is_none() {
            scope.spawn(|_| {
                let ret = tracing::trace_span!("Parse HfGlobal").in_scope(|| -> Result<_> {
                    *hf_global = frame.try_parse_hf_global(Some(lf_global)).transpose()?;
                    Ok(())
                });
                if let Err(e) = ret {
                    *result.write().unwrap() = Err(e);
                }
            });
        }

        let lf_xyb = profile_stage!(
            profile,
            load_lf_ns,
            tracing::trace_span!("Load LF groups").in_scope(|| {
                util::load_lf_groups(
                    frame,
                    lf_global,
                    lf_groups,
                    lf_group_image,
                    modular_lf_region,
                    pool,
                )
            })
        )?;

        let lf_xyb = if let Some(x) = lf_frame {
            tracing::trace_span!("Copy LFQuant").in_scope(|| -> Result<_> {
                let lf_frame = std::sync::Arc::clone(&x.image).run_with_image()?;
                let lf_frame = lf_frame.blend(None, pool)?.try_clone()?;
                Ok(lf_frame)
            })?
        } else {
            let mut lf_xyb = lf_xyb.unwrap();

            if !subsampled {
                tracing::trace_span!("LF CfL").in_scope(|| {
                    chroma_from_luma_lf(
                        lf_xyb.as_color_floats_mut(),
                        &lf_global_vardct.lf_chan_corr,
                    );
                });
            }

            if !frame_header.flags.skip_adaptive_lf_smoothing() {
                tracing::trace_span!("Adaptive LF smoothing").in_scope(|| {
                    adaptive_lf_smoothing(
                        lf_xyb.as_color_floats_mut(),
                        &lf_global.lf_dequant,
                        &lf_global_vardct.quantizer,
                    )
                })?;
            }

            lf_xyb
        };

        let fb = {
            let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
                ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
            });
            let Region { width, height, .. } = modular_region;

            let mut fb = ImageWithRegion::new(3, tracker);
            for shift in shifts_cbycr {
                let (w8, h8) = shift.shift_size((width.div_ceil(8), height.div_ceil(8)));
                let width = w8 * 8;
                let height = h8 * 8;
                // SAFETY: transform_with_lf_grouped writes every pixel before it is read.
                // Every group in the frame is processed by pool.for_each_vec below, covering
                // the entire [0..width) × [0..height) region without gaps.
                let buffer = unsafe {
                    AlignedGrid::<f32>::uninit(width as usize, height as usize, tracker)?
                };
                fb.append_channel_shifted(ImageBuffer::F32(buffer), modular_region, shift);
            }
            fb
        };

        Ok((fb, lf_xyb))
    })?;
    result.into_inner().unwrap()?;

    let hf_global = cache.hf_global.as_ref();
    let lf_groups = &mut cache.lf_groups;

    let mut it = profile_stage!(
        profile,
        prepare_groups_ns,
        tracing::trace_span!("Prepare PassGroup").in_scope(|| {
            fb.color_groups_with_group_id(frame_header)
                .into_iter()
                .filter_map(|(group_idx, grid_xyb)| {
                    let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
                    let lf_group = lf_groups.get(&lf_group_idx)?;

                    Some((group_idx, grid_xyb, lf_group))
                })
                .collect::<Vec<_>>()
        })
    );
    // Per-group "Decode + Transform": each scope task decodes all passes for one group and then
    // immediately dequantizes / transforms that group's coefficients. Non-subsampled groups use a
    // compact per-block HF coefficient store for decode, then dequantize directly into the strided
    // f32 output grids before the existing CfL + IDCT passes.
    tracing::trace_span!("Decode and transform").in_scope(|| -> Result<()> {
        let groups_per_row = frame_header.groups_per_row();
        let lf_xyb_ref = &lf_xyb;
        let lf_groups_ref: &HashMap<u32, LfGroup<S>> = lf_groups;

        let Some(hf_global) = hf_global else {
            // Modular / LF-only frame: nothing to decode; transform with LF for all groups.
            pool.for_each_vec(it, |job| {
                let (group_idx, mut grid_xyb, _) = job;
                transform_with_lf_grouped(
                    lf_xyb_ref,
                    &mut grid_xyb,
                    group_idx,
                    frame_header,
                    lf_groups_ref,
                );
            });
            return Ok(());
        };

        let global_ma_config = gmodular.ma_config.as_ref();
        let result = std::sync::RwLock::new(Result::Ok(()));
        let num_passes = frame_header.passes.num_passes as usize;

        // Per-pass modular subimage iterators (indexed by group_idx).
        let mut pass_modular_vecs: Vec<Vec<Option<_>>> = pass_group_image
            .into_iter()
            .map(|pass_image| pass_image.into_iter().map(Some).collect())
            .collect();

        // Per-group task: owns the output grid and all per-pass bitstreams.
        // Constructed sequentially on the main thread before the parallel scope.
        struct GroupTask<'g, 'frame, S: Sample> {
            group_idx: u32,
            grid_xyb: [MutableSubgrid<'g, f32>; 3],
            lf_group: &'frame LfGroup<S>,
            // Per-pass: Some(bitstream, allow_partial, optional modular) or None when unavailable.
            pass_data: Vec<
                Option<(
                    crate::jxl_decode::jxl_bitstream::Bitstream<'frame>,
                    bool,
                    Option<
                        crate::jxl_decode::jxl_modular::image::TransformedModularSubimage<'g, S>,
                    >,
                )>,
            >,
        }
        // SAFETY: all fields are Send (MutableSubgrid<f32>: Send, &LfGroup: Send if LfGroup: Sync,
        // Bitstream<'_>: Send, TransformedModularSubimage: Send).
        unsafe impl<'g, 'frame, S: Sample + Send> Send for GroupTask<'g, 'frame, S> {}

        let group_tasks: Vec<GroupTask<'_, '_, S>> = profile_stage!(
            profile,
            task_setup_ns,
            it.into_iter()
                .map(|(group_idx, grid_xyb, lf_group)| {
                    let group_x = group_idx % groups_per_row;
                    let group_y = group_idx / groups_per_row;
                    let transform_hf = {
                        let left = group_x * group_dim;
                        let top = group_y * group_dim;
                        let gr = Region {
                            left: left as i32,
                            top: top as i32,
                            width: group_dim,
                            height: group_dim,
                        };
                        !gr.intersection(aligned_region).is_empty()
                    };
                    let pass_data = if lf_group.hf_meta.is_some() && transform_hf {
                        (0..num_passes)
                            .map(|pass_idx| {
                                let modular = pass_modular_vecs
                                    .get_mut(pass_idx)
                                    .and_then(|v| v.get_mut(group_idx as usize))
                                    .and_then(Option::take);
                                match frame.pass_group_bitstream(pass_idx as u32, group_idx) {
                                    Some(Ok(bs)) => Some((bs.bitstream, bs.partial, modular)),
                                    Some(Err(e)) => {
                                        *result.write().unwrap() = Err(e.into());
                                        None
                                    }
                                    None => None,
                                }
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    GroupTask {
                        group_idx,
                        grid_xyb,
                        lf_group,
                        pass_data,
                    }
                })
                .collect()
        );

        let run_group_task = |task| {
            let result_ref = &result;
            let GroupTask {
                group_idx,
                mut grid_xyb,
                lf_group,
                pass_data,
            } = task;
            let profile_for_task = profile.clone();
            let group_x = group_idx % groups_per_row;
            let group_y = group_idx / groups_per_row;
            let transform_hf = {
                let left = group_x * group_dim;
                let top = group_y * group_dim;
                let gr = Region {
                    left: left as i32,
                    top: top as i32,
                    width: group_dim,
                    height: group_dim,
                };
                !gr.intersection(aligned_region).is_empty()
            };

            {
                let _task_timer = start_vardct_task_timer(&profile_for_task);
                let has_hf = lf_group.hf_meta.is_some() && transform_hf;
                let use_compact_hf = has_hf && !subsampled;
                let use_direct_hf = use_compact_hf
                    && num_passes == 1
                    && matches!(pass_data.first(), Some(Some((_, false, None))));
                if use_direct_hf {
                    let hf_meta = lf_group.hf_meta.as_ref().unwrap();
                    let group_dim_u = group_dim as usize;
                    let left_in_lf = ((group_x % 8) * (group_dim / 8)) as usize;
                    let top_in_lf = ((group_y % 8) * (group_dim / 8)) as usize;
                    let lf_width = (hf_meta.block_info.width() - left_in_lf).min(group_dim_u / 8);
                    let lf_height = (hf_meta.block_info.height() - top_in_lf).min(group_dim_u / 8);
                    let block_info_sub = hf_meta.block_info.as_subgrid().subgrid(
                        left_in_lf..(left_in_lf + lf_width),
                        top_in_lf..(top_in_lf + lf_height),
                    );
                    if !all_blocks_are_dct8(&block_info_sub) {
                        let cfl_base_x = ((group_x % 8) * group_dim / 64) as usize;
                        let cfl_base_y = ((group_y % 8) * group_dim / 64) as usize;
                        let gw = grid_xyb[0].width().div_ceil(64);
                        let gh = grid_xyb[0].height().div_ceil(64);
                        let x_from_y = hf_meta
                            .x_from_y
                            .as_subgrid()
                            .subgrid(cfl_base_x..(cfl_base_x + gw), cfl_base_y..(cfl_base_y + gh));
                        let b_from_y = hf_meta
                            .b_from_y
                            .as_subgrid()
                            .subgrid(cfl_base_x..(cfl_base_x + gw), cfl_base_y..(cfl_base_y + gh));
                        let lf = compact_hf_lf_subgrids(lf_xyb_ref, group_idx, frame_header);
                        let lf_global_vardct = lf_global.vardct.as_ref().unwrap();
                        let oim = &image_header.metadata.opsin_inverse_matrix;
                        let quantizer = &lf_global_vardct.quantizer;
                        let qm_scale = [
                            0.8f32.powi(frame_header.x_qm_scale as i32 - 2),
                            1.0f32,
                            0.8f32.powi(frame_header.b_qm_scale as i32 - 2),
                        ];
                        let quant_bias = [oim.quant_bias[0], oim.quant_bias[1], oim.quant_bias[2]];
                        let quant_bias_numerator = oim.quant_bias_numerator;
                        #[cfg(target_arch = "x86_64")]
                        let use_avx2 = std::is_x86_feature_detected!("avx2")
                            && std::is_x86_feature_detected!("fma");
                        #[cfg(not(target_arch = "x86_64"))]
                        let use_avx2 = false;
                        let (mut bitstream, _, _) = pass_data.into_iter().next().unwrap().unwrap();

                        HF_COMPACT_SCRATCH.with(|f32_cell| {
                            let mut f32_scratch = f32_cell.borrow_mut();
                            let f32_scratch = f32_scratch.buf_mut();
                            HF_COEFF_DIRECT_SCRATCH.with(|coeff_cell| {
                                let mut coeff_scratch = coeff_cell.borrow_mut();
                                let r = profile_stage!(profile_for_task, hf_decode_ns, {
                                    decode_hf_coeff_direct(
                                        &mut bitstream,
                                        frame_header,
                                        lf_group,
                                        0,
                                        group_idx,
                                        lf_global_vardct,
                                        hf_global,
                                        tracker,
                                        &mut coeff_scratch,
                                        |bx, by, dct_select, hf_mul, compact| {
                                            profile_stage!(
                                                profile_for_task,
                                                generic_transform_ns,
                                                {
                                                    dequant_cfl_direct_transform_block(
                                                        compact,
                                                        &mut grid_xyb,
                                                        &lf,
                                                        bx,
                                                        by,
                                                        dct_select,
                                                        hf_mul,
                                                        &hf_global.dequant_matrices,
                                                        quantizer,
                                                        qm_scale,
                                                        quant_bias,
                                                        quant_bias_numerator,
                                                        &lf_global_vardct.lf_chan_corr,
                                                        &x_from_y,
                                                        &b_from_y,
                                                        use_avx2,
                                                        f32_scratch,
                                                        profile_for_task.clone(),
                                                    );
                                                }
                                            );
                                        },
                                    )
                                });
                                if let Err(error) = r {
                                    *result_ref.write().unwrap() = Err(error.into());
                                }
                            });
                        });
                        return;
                    }
                }
                let mut decode_ok = true;
                let mut compact_store = profile_stage!(profile_for_task, compact_alloc_ns, {
                    if use_compact_hf {
                        let hf_meta = lf_group.hf_meta.as_ref().unwrap();
                        let left_in_lf = ((group_x % 8) * (group_dim / 8)) as usize;
                        let top_in_lf = ((group_y % 8) * (group_dim / 8)) as usize;
                        let lf_width =
                            (hf_meta.block_info.width() - left_in_lf).min(group_dim as usize / 8);
                        let lf_height =
                            (hf_meta.block_info.height() - top_in_lf).min(group_dim as usize / 8);
                        let block_info_sub = hf_meta.block_info.as_subgrid().subgrid(
                            left_in_lf..(left_in_lf + lf_width),
                            top_in_lf..(top_in_lf + lf_height),
                        );
                        Some(CompactHfCoeffStore::new(&block_info_sub))
                    } else {
                        None
                    }
                });

                profile_stage!(profile_for_task, hf_decode_ns, {
                    if has_hf {
                        for (pass_idx, entry) in pass_data.into_iter().enumerate() {
                            let Some((mut bitstream, allow_partial, modular)) = entry else {
                                continue;
                            };
                            if let Some(compact_store) = compact_store.as_mut() {
                                let r = decode_pass_group_compact(
                                    &mut bitstream,
                                    PassGroupParamsCompact {
                                        frame_header,
                                        lf_group,
                                        pass_idx: pass_idx as u32,
                                        group_idx,
                                        global_ma_config,
                                        modular,
                                        vardct: Some(PassGroupParamsVardctCompact {
                                            lf_vardct: lf_global_vardct,
                                            hf_global,
                                            hf_coeff_compact: compact_store,
                                        }),
                                        allow_partial,
                                        tracker,
                                        pool,
                                    },
                                );
                                if !allow_partial && r.is_err() {
                                    *result_ref.write().unwrap() = r.map_err(From::from);
                                    decode_ok = false;
                                }
                            } else {
                                let [x, y, b] = &mut grid_xyb;
                                let mut gi32 = [
                                    x.borrow_mut().into_i32(),
                                    y.borrow_mut().into_i32(),
                                    b.borrow_mut().into_i32(),
                                ];
                                let r = crate::jxl_decode::jxl_frame::data::decode_pass_group(
                                    &mut bitstream,
                                    PassGroupParams {
                                        frame_header,
                                        lf_group,
                                        pass_idx: pass_idx as u32,
                                        group_idx,
                                        global_ma_config,
                                        modular,
                                        vardct: Some(PassGroupParamsVardct {
                                            lf_vardct: lf_global_vardct,
                                            hf_global,
                                            hf_coeff_output: &mut gi32,
                                        }),
                                        allow_partial,
                                        tracker,
                                        pool,
                                    },
                                );
                                if !allow_partial && r.is_err() {
                                    *result_ref.write().unwrap() = r.map_err(From::from);
                                    decode_ok = false;
                                }
                            }
                        }
                    }
                });
                if let Some(compact_store) = compact_store.as_ref() {
                    if decode_ok {
                        let hf_meta = lf_group.hf_meta.as_ref().unwrap();
                        let group_dim_u = group_dim as usize;
                        let left_in_lf = ((group_x % 8) * (group_dim / 8)) as usize;
                        let top_in_lf = ((group_y % 8) * (group_dim / 8)) as usize;
                        let lf_width =
                            (hf_meta.block_info.width() - left_in_lf).min(group_dim_u / 8);
                        let lf_height =
                            (hf_meta.block_info.height() - top_in_lf).min(group_dim_u / 8);
                        let block_info_sub = hf_meta.block_info.as_subgrid().subgrid(
                            left_in_lf..(left_in_lf + lf_width),
                            top_in_lf..(top_in_lf + lf_height),
                        );
                        let cfl_base_x = ((group_x % 8) * group_dim / 64) as usize;
                        let cfl_base_y = ((group_y % 8) * group_dim / 64) as usize;
                        let gw = grid_xyb[0].width().div_ceil(64);
                        let gh = grid_xyb[0].height().div_ceil(64);
                        let x_from_y = hf_meta
                            .x_from_y
                            .as_subgrid()
                            .subgrid(cfl_base_x..(cfl_base_x + gw), cfl_base_y..(cfl_base_y + gh));
                        let b_from_y = hf_meta
                            .b_from_y
                            .as_subgrid()
                            .subgrid(cfl_base_x..(cfl_base_x + gw), cfl_base_y..(cfl_base_y + gh));
                        if all_blocks_are_dct8(&block_info_sub) {
                            profile_stage!(profile_for_task, dct8_transform_ns, {
                                dequant_cfl_transform_compact_dct8_grouped(
                                    compact_store,
                                    &mut grid_xyb,
                                    lf_xyb_ref,
                                    group_idx,
                                    image_header,
                                    frame_header,
                                    lf_global,
                                    lf_groups_ref,
                                    hf_global,
                                    &x_from_y,
                                    &b_from_y,
                                );
                            });
                            return;
                        }
                        profile_stage!(profile_for_task, generic_transform_ns, {
                            dequant_cfl_compact_transform_grouped(
                                compact_store,
                                &mut grid_xyb,
                                lf_xyb_ref,
                                group_idx,
                                image_header,
                                frame_header,
                                lf_global,
                                lf_groups_ref,
                                hf_global,
                                &lf_global_vardct.lf_chan_corr,
                                &x_from_y,
                                &b_from_y,
                                profile_for_task.clone(),
                            );
                        });
                        return;
                    }
                } else if has_hf && decode_ok {
                    if !subsampled {
                        dequant_hf_varblock_grouped(
                            &mut grid_xyb,
                            group_idx,
                            image_header,
                            frame_header,
                            lf_global,
                            lf_groups_ref,
                            hf_global,
                        );
                        let hf_meta = lf_group.hf_meta.as_ref().unwrap();
                        let cfl_base_x = ((group_x % 8) * group_dim / 64) as usize;
                        let cfl_base_y = ((group_y % 8) * group_dim / 64) as usize;
                        let gw = grid_xyb[0].width().div_ceil(64);
                        let gh = grid_xyb[0].height().div_ceil(64);
                        let x_from_y = hf_meta
                            .x_from_y
                            .as_subgrid()
                            .subgrid(cfl_base_x..(cfl_base_x + gw), cfl_base_y..(cfl_base_y + gh));
                        let b_from_y = hf_meta
                            .b_from_y
                            .as_subgrid()
                            .subgrid(cfl_base_x..(cfl_base_x + gw), cfl_base_y..(cfl_base_y + gh));
                        chroma_from_luma_hf_grouped(
                            &mut grid_xyb,
                            &x_from_y,
                            &b_from_y,
                            &lf_global_vardct.lf_chan_corr,
                        );
                    } else {
                        dequant_hf_varblock_grouped(
                            &mut grid_xyb,
                            group_idx,
                            image_header,
                            frame_header,
                            lf_global,
                            lf_groups_ref,
                            hf_global,
                        );
                    }
                }

                profile_stage!(profile_for_task, fallback_transform_ns, {
                    transform_with_lf_grouped(
                        lf_xyb_ref,
                        &mut grid_xyb,
                        group_idx,
                        frame_header,
                        lf_groups_ref,
                    );
                });
            }
        };

        profile_stage!(profile, workers_wall_ns, {
            if group_tasks.len() <= 128 {
                pool.for_each_vec(group_tasks, &run_group_task);
            } else {
                pool.scope(|scope| {
                    for task in group_tasks {
                        let run_group_task = &run_group_task;
                        scope.spawn(move |_| run_group_task(task));
                    }
                });
            }
        });

        result.into_inner().unwrap()
    })?;

    if let Some(modular_image) = modular_image {
        tracing::trace_span!("Extra channel inverse transform").in_scope(|| {
            modular_image.prepare_subimage().unwrap().finish(pool);
        });
        fb.extend_from_gmodular(gmodular);
    }

    finish_vardct_profile(&profile);
    Ok(fb)
}

pub fn copy_lf_dequant<S: Sample>(
    grid: &mut MutableSubgrid<f32>,
    quantizer: &Quantizer,
    m_lf: f32,
    channel_data: &AlignedGrid<S>,
    extra_precision: u8,
) {
    debug_assert!(extra_precision < 4);
    assert!(grid.width() >= channel_data.width());
    assert!(grid.height() >= channel_data.height());

    let precision_scale = 1i32 << (9 - extra_precision);
    let scale_inv = quantizer.global_scale as u64 * quantizer.quant_lf as u64;
    let scale = (m_lf as f64 * precision_scale as f64 / scale_inv as f64) as f32;

    let width = channel_data.width();
    let height = channel_data.height();
    let buf = channel_data.buf();
    for y in 0..height {
        let row = grid.get_row_mut(y);
        let quant = &buf[y * width..][..width];
        for (out, &q) in row.iter_mut().zip(quant) {
            *out = q.to_i32() as f32 * scale;
        }
    }
}

pub fn adaptive_lf_smoothing(
    lf_image: [&mut AlignedGrid<f32>; 3],
    lf_dequant: &LfChannelDequantization,
    quantizer: &Quantizer,
) -> Result<()> {
    let scale_inv = quantizer.global_scale as u64 * quantizer.quant_lf as u64;
    let lf_x = (512.0 * lf_dequant.m_x_lf as f64 / scale_inv as f64) as f32;
    let lf_y = (512.0 * lf_dequant.m_y_lf as f64 / scale_inv as f64) as f32;
    let lf_b = (512.0 * lf_dequant.m_b_lf as f64 / scale_inv as f64) as f32;

    let [in_x, in_y, in_b] = lf_image;
    let tracker = in_x.tracker();
    let width = in_x.width();
    let height = in_x.height();

    let in_x = in_x.buf_mut();
    let in_y = in_y.buf_mut();
    let in_b = in_b.buf_mut();

    impls::adaptive_lf_smoothing_impl(
        width,
        height,
        [in_x, in_y, in_b],
        [lf_x, lf_y, lf_b],
        tracker.as_ref(),
    )
}

pub fn dequant_cfl_compact_to_strided<S: Sample>(
    compact_store: &CompactHfCoeffStore,
    out: &mut [MutableSubgrid<'_, f32>; 3],
    group_idx: u32,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    lf_global: &LfGlobal<S>,
    lf_groups: &HashMap<u32, LfGroup<S>>,
    hf_global: &HfGlobal,
    lf_chan_corr: &LfChannelCorrelation,
    x_from_y: &SharedSubgrid<i32>,
    b_from_y: &SharedSubgrid<i32>,
) {
    let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });
    let oim = &image_header.metadata.opsin_inverse_matrix;
    let quantizer = &lf_global.vardct.as_ref().unwrap().quantizer;
    let dequant_matrices = &hf_global.dequant_matrices;

    let qm_scale = [
        0.8f32.powi(frame_header.x_qm_scale as i32 - 2),
        1.0f32,
        0.8f32.powi(frame_header.b_qm_scale as i32 - 2),
    ];
    let quant_bias = [oim.quant_bias[0], oim.quant_bias[1], oim.quant_bias[2]];
    let quant_bias_numerator = oim.quant_bias_numerator;

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();
    let group_x = group_idx % groups_per_row;
    let group_y = group_idx / groups_per_row;

    let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
    let Some(lf_group) = lf_groups.get(&lf_group_idx) else {
        return;
    };
    let left_in_lf = ((group_x % 8) * (group_dim / 8)) as usize;
    let top_in_lf = ((group_y % 8) * (group_dim / 8)) as usize;

    let Some(hf_meta) = &lf_group.hf_meta else {
        return;
    };

    let block_info = {
        let lf_width = (hf_meta.block_info.width() - left_in_lf).min(group_dim as usize / 8);
        let lf_height = (hf_meta.block_info.height() - top_in_lf).min(group_dim as usize / 8);
        hf_meta.block_info.as_subgrid().subgrid(
            left_in_lf..(left_in_lf + lf_width),
            top_in_lf..(top_in_lf + lf_height),
        )
    };
    let [coeff_x, coeff_y, coeff_b] = out;
    let quant_bias = [oim.quant_bias[0], oim.quant_bias[1], oim.quant_bias[2]];
    let shift = shifts_cbycr[0];
    for_each_varblocks(
        &block_info,
        shift,
        |VarblockInfo {
             shifted_bx,
             shifted_by,
             dct_select,
             hf_mul,
         }| {
            let (bw, bh) = dct_select.dct_select_size();
            let left = shifted_bx * 8;
            let top = shifted_by * 8;

            let bw = bw as usize;
            let bh = bh as usize;
            let width = bw * 8;
            let height = bh * 8;

            let need_transpose = dct_select.need_transpose();
            let matrix_x = if need_transpose {
                dequant_matrices.get_transposed(0, dct_select)
            } else {
                dequant_matrices.get(0, dct_select)
            };
            let matrix_y = if need_transpose {
                dequant_matrices.get_transposed(1, dct_select)
            } else {
                dequant_matrices.get(1, dct_select)
            };
            let matrix_b = if need_transpose {
                dequant_matrices.get_transposed(2, dct_select)
            } else {
                dequant_matrices.get(2, dct_select)
            };
            let mul = [
                65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[0],
                65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[1],
                65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[2],
            ];
            let compact_x = compact_store.get_channel(shifted_bx, shifted_by, 0);
            let compact_y = compact_store.get_channel(shifted_bx, shifted_by, 1);
            let compact_b = compact_store.get_channel(shifted_bx, shifted_by, 2);

            for y in 0..height {
                let abs_y = top + y;
                let cfl_y = abs_y / 64;
                let row_x = coeff_x.get_row_mut(abs_y);
                let row_y = coeff_y.get_row_mut(abs_y);
                let row_b = coeff_b.get_row_mut(abs_y);

                let input_base = y * width;
                let matrix_row_x = &matrix_x[input_base..][..width];
                let matrix_row_y = &matrix_y[input_base..][..width];
                let matrix_row_b = &matrix_b[input_base..][..width];

                let mut x = 0usize;
                while x < width {
                    let abs_x = left + x;
                    let cfl_x = abs_x / 64;
                    let next = ((cfl_x + 1) * 64).saturating_sub(left).min(width);
                    let len = next - x;
                    let kx = lf_chan_corr.base_correlation_x
                        + (x_from_y.get(cfl_x, cfl_y) as f32 / lf_chan_corr.colour_factor as f32);
                    let kb = lf_chan_corr.base_correlation_b
                        + (b_from_y.get(cfl_x, cfl_y) as f32 / lf_chan_corr.colour_factor as f32);

                    let input_x = &compact_x[input_base + x..input_base + x + len];
                    let input_y = &compact_y[input_base + x..input_base + x + len];
                    let input_b = &compact_b[input_base + x..input_base + x + len];
                    let out_x = &mut row_x[left + x..left + x + len];
                    let out_y = &mut row_y[left + x..left + x + len];
                    let out_b = &mut row_b[left + x..left + x + len];
                    let mat_x = &matrix_row_x[x..x + len];
                    let mat_y = &matrix_row_y[x..x + len];
                    let mat_b = &matrix_row_b[x..x + len];

                    if use_avx2 {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            dequant_cfl_row_avx2_typed(
                                input_x,
                                input_y,
                                input_b,
                                out_x,
                                out_y,
                                out_b,
                                mat_x,
                                mat_y,
                                mat_b,
                                quant_bias,
                                quant_bias_numerator,
                                mul,
                                kx,
                                kb,
                            );
                        }
                    } else {
                        dequant_cfl_row_scalar_typed(
                            input_x,
                            input_y,
                            input_b,
                            out_x,
                            out_y,
                            out_b,
                            mat_x,
                            mat_y,
                            mat_b,
                            quant_bias,
                            quant_bias_numerator,
                            mul,
                            kx,
                            kb,
                        );
                    }
                    x = next;
                }
            }
        },
    );
}

thread_local! {
    // 32-byte aligned (AlignedGrid uses 64-byte alignment), so as_vectored::<__m256>()
    // succeeds for the compact IDCT. Pre-sized to 3× the JXL max block (64×64).
    static HF_COMPACT_SCRATCH: RefCell<AlignedGrid<f32>> = RefCell::new(
        AlignedGrid::with_alloc_tracker(4096 * 3, 1, None)
            .expect("failed to allocate compact HF scratch"),
    );
    static HF_COEFF_DIRECT_SCRATCH: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
}

/// Fully fused per-block pipeline: compact i32 → dequant+CfL → compact f32
/// → LF insertion → IDCT → scatter to strided f32.
/// Skips the separate `transform_with_lf_grouped` pass entirely.
pub fn dequant_cfl_compact_transform_grouped<S: Sample>(
    compact_store: &CompactHfCoeffStore,
    out: &mut [MutableSubgrid<'_, f32>; 3],
    lf_image: &ImageWithRegion,
    group_idx: u32,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    lf_global: &LfGlobal<S>,
    lf_groups: &HashMap<u32, LfGroup<S>>,
    hf_global: &HfGlobal,
    lf_chan_corr: &LfChannelCorrelation,
    x_from_y: &SharedSubgrid<i32>,
    b_from_y: &SharedSubgrid<i32>,
    profile: VardctProfileHandle,
) {
    let lf_regions = <[_; 3]>::try_from(&lf_image.regions_and_shifts()[..3]).unwrap();
    let [lf_x, lf_y, lf_b] = lf_image.as_color_floats();

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();
    let (group_width, group_height) = frame_header.group_size_for(group_idx);
    let group_x = group_idx % groups_per_row;
    let group_y = group_idx / groups_per_row;
    let lf_base_left = group_x * group_dim / 8;
    let lf_base_top = group_y * group_dim / 8;

    let lf = [
        (lf_regions[0], lf_x),
        (lf_regions[1], lf_y),
        (lf_regions[2], lf_b),
    ]
    .map(|((lf_region, shift), lf)| {
        let lf_base_left = lf_base_left.checked_add_signed(-lf_region.left).unwrap();
        let lf_base_top = lf_base_top.checked_add_signed(-lf_region.top).unwrap();
        let lf_width = (lf_region.width - lf_base_left).min(group_width.div_ceil(8));
        let lf_height = (lf_region.height - lf_base_top).min(group_height.div_ceil(8));
        let lf_base_left = (lf_base_left as usize) >> shift.hshift();
        let lf_base_top = (lf_base_top as usize) >> shift.vshift();
        let (lf_width, lf_height) = shift.shift_size((lf_width, lf_height));
        lf.as_subgrid().subgrid(
            lf_base_left..(lf_base_left + lf_width as usize),
            lf_base_top..(lf_base_top + lf_height as usize),
        )
    });

    let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
    let Some(lf_group) = lf_groups.get(&lf_group_idx) else {
        return;
    };
    let Some(hf_meta) = &lf_group.hf_meta else {
        return;
    };
    let left_in_lf = ((group_x % 8) * (group_dim / 8)) as usize;
    let top_in_lf = ((group_y % 8) * (group_dim / 8)) as usize;
    let lf_width = (hf_meta.block_info.width() - left_in_lf).min(group_dim as usize / 8);
    let lf_height = (hf_meta.block_info.height() - top_in_lf).min(group_dim as usize / 8);
    let block_info = hf_meta.block_info.as_subgrid().subgrid(
        left_in_lf..(left_in_lf + lf_width),
        top_in_lf..(top_in_lf + lf_height),
    );

    let lf_global_vardct = lf_global.vardct.as_ref().unwrap();
    let oim = &image_header.metadata.opsin_inverse_matrix;
    let quantizer = &lf_global_vardct.quantizer;
    let dequant_matrices = &hf_global.dequant_matrices;
    let qm_scale = [
        0.8f32.powi(frame_header.x_qm_scale as i32 - 2),
        1.0f32,
        0.8f32.powi(frame_header.b_qm_scale as i32 - 2),
    ];
    let quant_bias = [oim.quant_bias[0], oim.quant_bias[1], oim.quant_bias[2]];
    let quant_bias_numerator = oim.quant_bias_numerator;

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    let shift = ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, 0);

    let max_block_size = 4096usize; // pre-allocated to JXL maximum (64×64)

    HF_COMPACT_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let scratch = scratch.buf_mut(); // &mut [f32], 64-byte aligned

        let [grid_x, grid_y, grid_b] = out;
        for_each_varblocks(
            &block_info,
            shift,
            |VarblockInfo {
                 shifted_bx,
                 shifted_by,
                 dct_select,
                 hf_mul,
             }| {
                #[cfg(feature = "__profile")]
                PROFILE_TRANSFORM_COUNTS[dct_select as usize].fetch_add(1, Ordering::Relaxed);
                let (bw8, bh8) = dct_select.dct_select_size();
                let bw8 = bw8 as usize;
                let bh8 = bh8 as usize;
                let block_w = bw8 * 8;
                let block_h = bh8 * 8;
                let block_size = block_w * block_h;
                let left = shifted_bx * 8;
                let top = shifted_by * 8;
                // For all JXL block types, block_w is a power-of-2 <= 64 aligned to block_w,
                // so the block never straddles a 64-pixel CfL tile boundary horizontally.
                let cfl_x = left / 64;

                let need_transpose = dct_select.need_transpose();
                let matrix_x = if need_transpose {
                    dequant_matrices.get_transposed(0, dct_select)
                } else {
                    dequant_matrices.get(0, dct_select)
                };
                let matrix_y = if need_transpose {
                    dequant_matrices.get_transposed(1, dct_select)
                } else {
                    dequant_matrices.get(1, dct_select)
                };
                let matrix_b = if need_transpose {
                    dequant_matrices.get_transposed(2, dct_select)
                } else {
                    dequant_matrices.get(2, dct_select)
                };
                let mul = [
                    65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[0],
                    65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[1],
                    65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[2],
                ];

                let compact_x = compact_store.get_channel_unchecked(shifted_bx, shifted_by, 0);
                let compact_y = compact_store.get_channel_unchecked(shifted_bx, shifted_by, 1);
                let compact_b = compact_store.get_channel_unchecked(shifted_bx, shifted_by, 2);

                let (sx, rest) = scratch.split_at_mut(block_size);
                let (sy, sb) = rest.split_at_mut(block_size);

                // --- Step 1: fused dequant + CfL → compact f32 scratch ---
                profile_stage!(profile, generic_dequant_ns, {
                    for y in 0..block_h {
                        let abs_y = top + y;
                        let cfl_y = abs_y / 64;
                        let kx = lf_chan_corr.base_correlation_x
                            + (x_from_y.get(cfl_x, cfl_y) as f32
                                / lf_chan_corr.colour_factor as f32);
                        let kb = lf_chan_corr.base_correlation_b
                            + (b_from_y.get(cfl_x, cfl_y) as f32
                                / lf_chan_corr.colour_factor as f32);
                        let row = y * block_w;
                        if use_avx2 {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                dequant_cfl_row_avx2_typed(
                                    &compact_x[row..row + block_w],
                                    &compact_y[row..row + block_w],
                                    &compact_b[row..row + block_w],
                                    &mut sx[row..row + block_w],
                                    &mut sy[row..row + block_w],
                                    &mut sb[row..row + block_w],
                                    &matrix_x[row..row + block_w],
                                    &matrix_y[row..row + block_w],
                                    &matrix_b[row..row + block_w],
                                    quant_bias,
                                    quant_bias_numerator,
                                    mul,
                                    kx,
                                    kb,
                                );
                            }
                        } else {
                            dequant_cfl_row_scalar_typed(
                                &compact_x[row..row + block_w],
                                &compact_y[row..row + block_w],
                                &compact_b[row..row + block_w],
                                &mut sx[row..row + block_w],
                                &mut sy[row..row + block_w],
                                &mut sb[row..row + block_w],
                                &matrix_x[row..row + block_w],
                                &matrix_y[row..row + block_w],
                                &matrix_b[row..row + block_w],
                                quant_bias,
                                quant_bias_numerator,
                                mul,
                                kx,
                                kb,
                            );
                        }
                    }
                });

                // --- Step 2: insert LF + IDCT for each channel, data stays compact ---
                profile_stage!(profile, generic_idct_ns, {
                    {
                        let mut coeff = MutableSubgrid::from_buf(sx, block_w, block_h, block_w);
                        insert_lf_dc(&mut coeff, &lf[0], shifted_bx, shifted_by, dct_select);
                    }
                    impls::transform_single_block_compact(sx, block_w, block_h, dct_select);
                    {
                        let mut coeff = MutableSubgrid::from_buf(sy, block_w, block_h, block_w);
                        insert_lf_dc(&mut coeff, &lf[1], shifted_bx, shifted_by, dct_select);
                    }
                    impls::transform_single_block_compact(sy, block_w, block_h, dct_select);
                    {
                        let mut coeff = MutableSubgrid::from_buf(sb, block_w, block_h, block_w);
                        insert_lf_dc(&mut coeff, &lf[2], shifted_bx, shifted_by, dct_select);
                    }
                    impls::transform_single_block_compact(sb, block_w, block_h, dct_select);
                });

                // --- Step 3: scatter compact f32 → strided output grid ---
                profile_stage!(profile, generic_scatter_ns, {
                    for row in 0..block_h {
                        let src = row * block_w;
                        let dst = left..left + block_w;
                        grid_x.get_row_mut(top + row)[dst.clone()]
                            .copy_from_slice(&sx[src..src + block_w]);
                        grid_y.get_row_mut(top + row)[dst.clone()]
                            .copy_from_slice(&sy[src..src + block_w]);
                        grid_b.get_row_mut(top + row)[dst].copy_from_slice(&sb[src..src + block_w]);
                    }
                });
            },
        );
    });
}

fn compact_hf_lf_subgrids<'a>(
    lf_image: &'a ImageWithRegion,
    group_idx: u32,
    frame_header: &FrameHeader,
) -> [SharedSubgrid<'a, f32>; 3] {
    let lf_regions = <[_; 3]>::try_from(&lf_image.regions_and_shifts()[..3]).unwrap();
    let [lf_x, lf_y, lf_b] = lf_image.as_color_floats();
    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();
    let (group_width, group_height) = frame_header.group_size_for(group_idx);
    let group_x = group_idx % groups_per_row;
    let group_y = group_idx / groups_per_row;
    let lf_base_left = group_x * group_dim / 8;
    let lf_base_top = group_y * group_dim / 8;

    [
        (lf_regions[0], lf_x),
        (lf_regions[1], lf_y),
        (lf_regions[2], lf_b),
    ]
    .map(|((lf_region, shift), lf)| {
        let lf_base_left = lf_base_left.checked_add_signed(-lf_region.left).unwrap();
        let lf_base_top = lf_base_top.checked_add_signed(-lf_region.top).unwrap();
        let lf_width = (lf_region.width - lf_base_left).min(group_width.div_ceil(8));
        let lf_height = (lf_region.height - lf_base_top).min(group_height.div_ceil(8));
        let lf_base_left = (lf_base_left as usize) >> shift.hshift();
        let lf_base_top = (lf_base_top as usize) >> shift.vshift();
        let (lf_width, lf_height) = shift.shift_size((lf_width, lf_height));
        lf.as_subgrid().subgrid(
            lf_base_left..(lf_base_left + lf_width as usize),
            lf_base_top..(lf_base_top + lf_height as usize),
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn dequant_cfl_direct_transform_block(
    compact: [&[i32]; 3],
    out: &mut [MutableSubgrid<'_, f32>; 3],
    lf: &[SharedSubgrid<'_, f32>; 3],
    shifted_bx: usize,
    shifted_by: usize,
    dct_select: TransformType,
    hf_mul: i32,
    dequant_matrices: &DequantMatrixSet,
    quantizer: &Quantizer,
    qm_scale: [f32; 3],
    quant_bias: [f32; 3],
    quant_bias_numerator: f32,
    lf_chan_corr: &LfChannelCorrelation,
    x_from_y: &SharedSubgrid<i32>,
    b_from_y: &SharedSubgrid<i32>,
    use_avx2: bool,
    scratch: &mut [f32],
    profile: VardctProfileHandle,
) {
    #[cfg(feature = "__profile")]
    PROFILE_TRANSFORM_COUNTS[dct_select as usize].fetch_add(1, Ordering::Relaxed);

    let [compact_x, compact_y, compact_b] = compact;
    let (bw8, bh8) = dct_select.dct_select_size();
    let bw8 = bw8 as usize;
    let bh8 = bh8 as usize;
    let block_w = bw8 * 8;
    let block_h = bh8 * 8;
    let block_size = block_w * block_h;
    debug_assert!(scratch.len() >= block_size * 3);
    let left = shifted_bx * 8;
    let top = shifted_by * 8;
    let cfl_x = left / 64;

    let need_transpose = dct_select.need_transpose();
    let matrix_x = if need_transpose {
        dequant_matrices.get_transposed(0, dct_select)
    } else {
        dequant_matrices.get(0, dct_select)
    };
    let matrix_y = if need_transpose {
        dequant_matrices.get_transposed(1, dct_select)
    } else {
        dequant_matrices.get(1, dct_select)
    };
    let matrix_b = if need_transpose {
        dequant_matrices.get_transposed(2, dct_select)
    } else {
        dequant_matrices.get(2, dct_select)
    };
    let mul = [
        65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[0],
        65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[1],
        65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[2],
    ];

    let (sx, rest) = scratch.split_at_mut(block_size);
    let (sy, sb) = rest.split_at_mut(block_size);

    profile_stage!(profile, generic_dequant_ns, {
        for y in 0..block_h {
            let abs_y = top + y;
            let cfl_y = abs_y / 64;
            let kx = lf_chan_corr.base_correlation_x
                + (x_from_y.get(cfl_x, cfl_y) as f32 / lf_chan_corr.colour_factor as f32);
            let kb = lf_chan_corr.base_correlation_b
                + (b_from_y.get(cfl_x, cfl_y) as f32 / lf_chan_corr.colour_factor as f32);
            let row = y * block_w;
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    dequant_cfl_row_avx2_typed(
                        &compact_x[row..row + block_w],
                        &compact_y[row..row + block_w],
                        &compact_b[row..row + block_w],
                        &mut sx[row..row + block_w],
                        &mut sy[row..row + block_w],
                        &mut sb[row..row + block_w],
                        &matrix_x[row..row + block_w],
                        &matrix_y[row..row + block_w],
                        &matrix_b[row..row + block_w],
                        quant_bias,
                        quant_bias_numerator,
                        mul,
                        kx,
                        kb,
                    );
                }
            } else {
                dequant_cfl_row_scalar_typed(
                    &compact_x[row..row + block_w],
                    &compact_y[row..row + block_w],
                    &compact_b[row..row + block_w],
                    &mut sx[row..row + block_w],
                    &mut sy[row..row + block_w],
                    &mut sb[row..row + block_w],
                    &matrix_x[row..row + block_w],
                    &matrix_y[row..row + block_w],
                    &matrix_b[row..row + block_w],
                    quant_bias,
                    quant_bias_numerator,
                    mul,
                    kx,
                    kb,
                );
            }
        }
    });

    profile_stage!(profile, generic_idct_ns, {
        {
            let mut coeff = MutableSubgrid::from_buf(sx, block_w, block_h, block_w);
            insert_lf_dc(&mut coeff, &lf[0], shifted_bx, shifted_by, dct_select);
        }
        impls::transform_single_block_compact(sx, block_w, block_h, dct_select);
        {
            let mut coeff = MutableSubgrid::from_buf(sy, block_w, block_h, block_w);
            insert_lf_dc(&mut coeff, &lf[1], shifted_bx, shifted_by, dct_select);
        }
        impls::transform_single_block_compact(sy, block_w, block_h, dct_select);
        {
            let mut coeff = MutableSubgrid::from_buf(sb, block_w, block_h, block_w);
            insert_lf_dc(&mut coeff, &lf[2], shifted_bx, shifted_by, dct_select);
        }
        impls::transform_single_block_compact(sb, block_w, block_h, dct_select);
    });

    profile_stage!(profile, generic_scatter_ns, {
        let [grid_x, grid_y, grid_b] = out;
        for row in 0..block_h {
            let src = row * block_w;
            let dst = left..left + block_w;
            grid_x.get_row_mut(top + row)[dst.clone()].copy_from_slice(&sx[src..src + block_w]);
            grid_y.get_row_mut(top + row)[dst.clone()].copy_from_slice(&sy[src..src + block_w]);
            grid_b.get_row_mut(top + row)[dst].copy_from_slice(&sb[src..src + block_w]);
        }
    });
}

/// Insert LF DC coefficients into the compact f32 coefficient buffer.
/// For small blocks (DCT8 etc.): sets position (0,0) to the LF value.
/// For large blocks (Dct16x32 etc.): fills the bw×bh DC band, applies
/// a forward DCT on that band, and scales each value.
#[inline]
fn insert_lf_dc(
    coeff: &mut MutableSubgrid<'_, f32>,
    lf: &SharedSubgrid<f32>,
    shifted_bx: usize,
    shifted_by: usize,
    dct_select: TransformType,
) {
    use TransformType::*;
    let (bw, bh) = dct_select.dct_select_size();
    let bw = bw as usize;
    let bh = bh as usize;
    let mut out = coeff.borrow_mut().subgrid(0..bw, 0..bh);
    if matches!(
        dct_select,
        Hornuss | Dct2 | Dct4 | Dct8x4 | Dct4x8 | Dct8 | Afv0 | Afv1 | Afv2 | Afv3
    ) {
        *out.get_mut(0, 0) = lf.get(shifted_bx, shifted_by);
        return;
    }
    let logbw = bw.trailing_zeros() as usize;
    let logbh = bh.trailing_zeros() as usize;
    for y in 0..bh {
        for x in 0..bw {
            *out.get_mut(x, y) = lf.get(shifted_bx + x, shifted_by + y);
        }
    }
    generic::dct_2d(&mut out, dct_common::DctDirection::Forward);
    for y in 0..bh {
        for x in 0..bw {
            *out.get_mut(x, y) /=
                dct_common::scale_f(y, 5 - logbh) * dct_common::scale_f(x, 5 - logbw);
        }
    }
}

pub fn dequant_hf_varblock_grouped<S: Sample>(
    out: &mut [MutableSubgrid<'_, f32>; 3],
    group_idx: u32,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    lf_global: &LfGlobal<S>,
    lf_groups: &HashMap<u32, LfGroup<S>>,
    hf_global: &HfGlobal,
) {
    let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });
    let oim = &image_header.metadata.opsin_inverse_matrix;
    let quantizer = &lf_global.vardct.as_ref().unwrap().quantizer;
    let dequant_matrices = &hf_global.dequant_matrices;

    let qm_scale = [
        0.8f32.powi(frame_header.x_qm_scale as i32 - 2),
        1.0f32,
        0.8f32.powi(frame_header.b_qm_scale as i32 - 2),
    ];

    let quant_bias_numerator = oim.quant_bias_numerator;
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();

    let group_x = group_idx % groups_per_row;
    let group_y = group_idx / groups_per_row;

    let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
    let Some(lf_group) = lf_groups.get(&lf_group_idx) else {
        return;
    };
    let left_in_lf = ((group_x % 8) * (group_dim / 8)) as usize;
    let top_in_lf = ((group_y % 8) * (group_dim / 8)) as usize;

    let Some(hf_meta) = &lf_group.hf_meta else {
        return;
    };

    let block_info = &hf_meta.block_info;
    let lf_width = (block_info.width() - left_in_lf).min(group_dim as usize / 8);
    let lf_height = (block_info.height() - top_in_lf).min(group_dim as usize / 8);
    let block_info = hf_meta.block_info.as_subgrid().subgrid(
        left_in_lf..(left_in_lf + lf_width),
        top_in_lf..(top_in_lf + lf_height),
    );

    for (channel, coeff) in out.iter_mut().enumerate() {
        let quant_bias = oim.quant_bias[channel];
        let shift = shifts_cbycr[channel];
        for_each_varblocks(
            &block_info,
            shift,
            |VarblockInfo {
                 shifted_bx,
                 shifted_by,
                 dct_select,
                 hf_mul,
             }| {
                let (bw, bh) = dct_select.dct_select_size();
                let left = shifted_bx * 8;
                let top = shifted_by * 8;

                let bw = bw as usize;
                let bh = bh as usize;
                let width = bw * 8;
                let height = bh * 8;

                let need_transpose = dct_select.need_transpose();
                let mul =
                    65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[channel];

                let matrix = if need_transpose {
                    dequant_matrices.get_transposed(channel, dct_select)
                } else {
                    dequant_matrices.get(channel, dct_select)
                };

                let mut coeff = coeff
                    .borrow_mut()
                    .subgrid(left..(left + width), top..(top + height));
                for (y, matrix_row) in matrix.chunks_exact(width).enumerate() {
                    let row = coeff.get_row_mut(y);
                    if use_avx2 {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            dequant_row_avx2(
                                row,
                                matrix_row,
                                quant_bias,
                                quant_bias_numerator,
                                mul,
                            );
                        }
                    } else {
                        dequant_row_scalar(row, matrix_row, quant_bias, quant_bias_numerator, mul);
                    }
                }
            },
        );
    }
}

#[inline(always)]
fn dequant_row_scalar(
    q: &mut [f32],
    m: &[f32],
    quant_bias: f32,
    quant_bias_numerator: f32,
    mul: f32,
) {
    for (q, &m) in q.iter_mut().zip(m) {
        let qn = q.to_bits() as i32;
        *q = qn as f32;
        if q.abs() <= 1.0 {
            *q *= quant_bias;
        } else {
            *q -= quant_bias_numerator / *q;
        }
        *q *= m;
        *q *= mul;
    }
}

#[inline(always)]
fn dequant_row_scalar_typed(
    input: &[i32],
    output: &mut [f32],
    m: &[f32],
    quant_bias: f32,
    quant_bias_numerator: f32,
    mul: f32,
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len(), m.len());

    for ((&qn, out), &m) in input.iter().zip(output.iter_mut()).zip(m) {
        *out = qn as f32;
        if out.abs() <= 1.0 {
            *out *= quant_bias;
        } else {
            *out -= quant_bias_numerator / *out;
        }
        *out *= m;
        *out *= mul;
    }
}

#[inline(always)]
fn dequant_one_scalar(
    qn: i32,
    m: f32,
    quant_bias: f32,
    quant_bias_numerator: f32,
    mul: f32,
) -> f32 {
    let mut q = qn as f32;
    if q.abs() <= 1.0 {
        q *= quant_bias;
    } else {
        q -= quant_bias_numerator / q;
    }
    q * m * mul
}

#[inline(always)]
fn dequant_cfl_row_scalar_typed(
    input_x: &[i32],
    input_y: &[i32],
    input_b: &[i32],
    output_x: &mut [f32],
    output_y: &mut [f32],
    output_b: &mut [f32],
    matrix_x: &[f32],
    matrix_y: &[f32],
    matrix_b: &[f32],
    quant_bias: [f32; 3],
    quant_bias_numerator: f32,
    mul: [f32; 3],
    kx: f32,
    kb: f32,
) {
    debug_assert_eq!(input_x.len(), input_y.len());
    debug_assert_eq!(input_x.len(), input_b.len());
    debug_assert_eq!(input_x.len(), output_x.len());
    debug_assert_eq!(input_x.len(), output_y.len());
    debug_assert_eq!(input_x.len(), output_b.len());

    for i in 0..input_x.len() {
        let fy = dequant_one_scalar(
            input_y[i],
            matrix_y[i],
            quant_bias[1],
            quant_bias_numerator,
            mul[1],
        );
        output_y[i] = fy;
        output_x[i] = dequant_one_scalar(
            input_x[i],
            matrix_x[i],
            quant_bias[0],
            quant_bias_numerator,
            mul[0],
        ) + kx * fy;
        output_b[i] = dequant_one_scalar(
            input_b[i],
            matrix_b[i],
            quant_bias[2],
            quant_bias_numerator,
            mul[2],
        ) + kb * fy;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_row_avx2(
    q: &mut [f32],
    m: &[f32],
    quant_bias: f32,
    quant_bias_numerator: f32,
    mul: f32,
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(q.len(), m.len());

    let vbias = _mm256_set1_ps(quant_bias);
    let vbias_num = _mm256_set1_ps(quant_bias_numerator);
    let vone = _mm256_set1_ps(1.0);
    let vmul = _mm256_set1_ps(mul);
    let sign_mask = _mm256_set1_ps(-0.0);

    let mut i = 0usize;
    while i + 8 <= q.len() {
        let vq_bits = _mm256_loadu_ps(q.as_ptr().add(i));
        let vq = _mm256_cvtepi32_ps(_mm256_castps_si256(vq_bits));
        let abs_q = _mm256_andnot_ps(sign_mask, vq);
        let bias_small = _mm256_mul_ps(vq, vbias);
        // Replace slow _mm256_div_ps (10-cycle throughput) with rcp + Newton-Raphson
        // (4+0.5+0.5 = 5-cycle throughput). rcp_ps gives ~12-bit; one NR step gives 24-bit.
        // When |q| <= 1 the blend selects bias_small, so NaN from rcp(0) is harmless.
        let vq_rcp0 = _mm256_rcp_ps(vq);
        let vq_rcp = _mm256_mul_ps(vq_rcp0, _mm256_fnmadd_ps(vq, vq_rcp0, _mm256_set1_ps(2.0)));
        let bias_large = _mm256_fnmadd_ps(vbias_num, vq_rcp, vq); // q - bias_num/q
        let mask = _mm256_cmp_ps(abs_q, vone, _CMP_LE_OS);
        let biased = _mm256_blendv_ps(bias_large, bias_small, mask);
        let vm = _mm256_loadu_ps(m.as_ptr().add(i));
        let out = _mm256_mul_ps(_mm256_mul_ps(biased, vm), vmul);
        _mm256_storeu_ps(q.as_mut_ptr().add(i), out);
        i += 8;
    }

    dequant_row_scalar(&mut q[i..], &m[i..], quant_bias, quant_bias_numerator, mul);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_row_avx2_typed(
    input: &[i32],
    output: &mut [f32],
    m: &[f32],
    quant_bias: f32,
    quant_bias_numerator: f32,
    mul: f32,
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(input.len(), m.len());

    let vbias = _mm256_set1_ps(quant_bias);
    let vbias_num = _mm256_set1_ps(quant_bias_numerator);
    let vone = _mm256_set1_ps(1.0);
    let vmul = _mm256_set1_ps(mul);
    let sign_mask = _mm256_set1_ps(-0.0);

    let mut i = 0usize;
    while i + 8 <= input.len() {
        let vq = _mm256_cvtepi32_ps(_mm256_loadu_si256(input.as_ptr().add(i) as *const __m256i));
        let abs_q = _mm256_andnot_ps(sign_mask, vq);
        let bias_small = _mm256_mul_ps(vq, vbias);
        let vq_rcp0 = _mm256_rcp_ps(vq);
        let vq_rcp = _mm256_mul_ps(vq_rcp0, _mm256_fnmadd_ps(vq, vq_rcp0, _mm256_set1_ps(2.0)));
        let bias_large = _mm256_fnmadd_ps(vbias_num, vq_rcp, vq);
        let mask = _mm256_cmp_ps(abs_q, vone, _CMP_LE_OS);
        let biased = _mm256_blendv_ps(bias_large, bias_small, mask);
        let vm = _mm256_loadu_ps(m.as_ptr().add(i));
        let out = _mm256_mul_ps(_mm256_mul_ps(biased, vm), vmul);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), out);
        i += 8;
    }

    dequant_row_scalar_typed(
        &input[i..],
        &mut output[i..],
        &m[i..],
        quant_bias,
        quant_bias_numerator,
        mul,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dequant_cfl_row_avx2_typed(
    input_x: &[i32],
    input_y: &[i32],
    input_b: &[i32],
    output_x: &mut [f32],
    output_y: &mut [f32],
    output_b: &mut [f32],
    matrix_x: &[f32],
    matrix_y: &[f32],
    matrix_b: &[f32],
    quant_bias: [f32; 3],
    quant_bias_numerator: f32,
    mul: [f32; 3],
    kx: f32,
    kb: f32,
) {
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn dequant_vec(
        input: *const i32,
        matrix: *const f32,
        quant_bias: f32,
        quant_bias_numerator: f32,
        mul: f32,
    ) -> __m256 {
        let vbias = _mm256_set1_ps(quant_bias);
        let vbias_num = _mm256_set1_ps(quant_bias_numerator);
        let vone = _mm256_set1_ps(1.0);
        let vmul = _mm256_set1_ps(mul);
        let sign_mask = _mm256_set1_ps(-0.0);

        let vq = _mm256_cvtepi32_ps(_mm256_loadu_si256(input as *const __m256i));
        let abs_q = _mm256_andnot_ps(sign_mask, vq);
        let bias_small = _mm256_mul_ps(vq, vbias);
        let vq_rcp0 = _mm256_rcp_ps(vq);
        let vq_rcp = _mm256_mul_ps(vq_rcp0, _mm256_fnmadd_ps(vq, vq_rcp0, _mm256_set1_ps(2.0)));
        let bias_large = _mm256_fnmadd_ps(vbias_num, vq_rcp, vq);
        let mask = _mm256_cmp_ps(abs_q, vone, _CMP_LE_OS);
        let biased = _mm256_blendv_ps(bias_large, bias_small, mask);
        let vm = _mm256_loadu_ps(matrix);
        _mm256_mul_ps(_mm256_mul_ps(biased, vm), vmul)
    }

    debug_assert_eq!(input_x.len(), input_y.len());
    debug_assert_eq!(input_x.len(), input_b.len());
    debug_assert_eq!(input_x.len(), output_x.len());
    debug_assert_eq!(input_x.len(), output_y.len());
    debug_assert_eq!(input_x.len(), output_b.len());

    let vkx = _mm256_set1_ps(kx);
    let vkb = _mm256_set1_ps(kb);
    let mut i = 0usize;
    while i + 8 <= input_x.len() {
        let vy = dequant_vec(
            input_y.as_ptr().add(i),
            matrix_y.as_ptr().add(i),
            quant_bias[1],
            quant_bias_numerator,
            mul[1],
        );
        let vx = dequant_vec(
            input_x.as_ptr().add(i),
            matrix_x.as_ptr().add(i),
            quant_bias[0],
            quant_bias_numerator,
            mul[0],
        );
        let vb = dequant_vec(
            input_b.as_ptr().add(i),
            matrix_b.as_ptr().add(i),
            quant_bias[2],
            quant_bias_numerator,
            mul[2],
        );
        _mm256_storeu_ps(output_y.as_mut_ptr().add(i), vy);
        _mm256_storeu_ps(output_x.as_mut_ptr().add(i), _mm256_fmadd_ps(vkx, vy, vx));
        _mm256_storeu_ps(output_b.as_mut_ptr().add(i), _mm256_fmadd_ps(vkb, vy, vb));
        i += 8;
    }

    dequant_cfl_row_scalar_typed(
        &input_x[i..],
        &input_y[i..],
        &input_b[i..],
        &mut output_x[i..],
        &mut output_y[i..],
        &mut output_b[i..],
        &matrix_x[i..],
        &matrix_y[i..],
        &matrix_b[i..],
        quant_bias,
        quant_bias_numerator,
        mul,
        kx,
        kb,
    );
}

pub fn chroma_from_luma_lf(
    coeff_xyb: [&mut AlignedGrid<f32>; 3],
    lf_chan_corr: &LfChannelCorrelation,
) {
    let LfChannelCorrelation {
        colour_factor,
        base_correlation_x,
        base_correlation_b,
        x_factor_lf,
        b_factor_lf,
        ..
    } = *lf_chan_corr;

    let x_factor = x_factor_lf as i32 - 128;
    let b_factor = b_factor_lf as i32 - 128;
    let kx = base_correlation_x + (x_factor as f32 / colour_factor as f32);
    let kb = base_correlation_b + (b_factor as f32 / colour_factor as f32);

    let [x, y, b] = coeff_xyb;
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe {
            chroma_from_luma_avx2(x.buf_mut(), y.buf(), b.buf_mut(), kx, kb);
        }
        return;
    }

    for ((x, y), b) in x.buf_mut().iter_mut().zip(y.buf()).zip(b.buf_mut()) {
        let y = *y;
        *x += kx * y;
        *b += kb * y;
    }
}

pub fn chroma_from_luma_hf_grouped(
    coeff_xyb: &mut [MutableSubgrid<'_, f32>; 3],
    x_from_y: &SharedSubgrid<i32>,
    b_from_y: &SharedSubgrid<i32>,
    lf_chan_corr: &LfChannelCorrelation,
) {
    let [coeff_x, coeff_y, coeff_b] = coeff_xyb;

    let gw = coeff_x.width();
    let gh = coeff_x.height();
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    for y in 0..gh {
        let x_from_y = x_from_y.get_row(y / 64);
        let b_from_y = b_from_y.get_row(y / 64);

        let coeff_x = coeff_x.get_row_mut(y);
        let coeff_y = coeff_y.get_row_mut(y);
        let coeff_b = coeff_b.get_row_mut(y);

        for (x64, (&kx, &kb)) in x_from_y.iter().zip(b_from_y).enumerate() {
            let kx =
                lf_chan_corr.base_correlation_x + (kx as f32 / lf_chan_corr.colour_factor as f32);
            let kb =
                lf_chan_corr.base_correlation_b + (kb as f32 / lf_chan_corr.colour_factor as f32);

            let start = x64 * 64;
            let len = (gw - start).min(64);
            if use_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    chroma_from_luma_avx2(
                        &mut coeff_x[start..(start + len)],
                        &coeff_y[start..(start + len)],
                        &mut coeff_b[start..(start + len)],
                        kx,
                        kb,
                    );
                }
            } else {
                for dx in 0..len {
                    let x = start + dx;
                    let coeff_y = coeff_y[x];
                    coeff_x[x] += kx * coeff_y;
                    coeff_b[x] += kb * coeff_y;
                }
            }
        }
    }
}

fn all_blocks_are_dct8(block_info: &SharedSubgrid<BlockInfo>) -> bool {
    use BlockInfo::*;
    use TransformType::Dct8;

    for by in 0..block_info.height() {
        for bx in 0..block_info.width() {
            match block_info.get(bx, by) {
                Data {
                    dct_select: Dct8, ..
                }
                | Uninit => {}
                _ => return false,
            }
        }
    }
    true
}

pub fn dequant_cfl_transform_compact_dct8_grouped<S: Sample>(
    compact_store: &CompactHfCoeffStore,
    grid: &mut [MutableSubgrid<'_, f32>; 3],
    lf_image: &ImageWithRegion,
    group_idx: u32,
    image_header: &ImageHeader,
    frame_header: &FrameHeader,
    lf_global: &LfGlobal<S>,
    lf_groups: &HashMap<u32, LfGroup<S>>,
    hf_global: &HfGlobal,
    x_from_y: &SharedSubgrid<i32>,
    b_from_y: &SharedSubgrid<i32>,
) {
    let oim = &image_header.metadata.opsin_inverse_matrix;
    let lf_global_vardct = lf_global.vardct.as_ref().unwrap();
    let quantizer = &lf_global_vardct.quantizer;
    let dequant_matrices = &hf_global.dequant_matrices;
    let lf_chan_corr = &lf_global_vardct.lf_chan_corr;

    let qm_scale = [
        0.8f32.powi(frame_header.x_qm_scale as i32 - 2),
        1.0f32,
        0.8f32.powi(frame_header.b_qm_scale as i32 - 2),
    ];
    let quant_bias = [oim.quant_bias[0], oim.quant_bias[1], oim.quant_bias[2]];
    let quant_bias_numerator = oim.quant_bias_numerator;

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();
    let group_x = group_idx % groups_per_row;
    let group_y = group_idx / groups_per_row;
    let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
    let Some(lf_group) = lf_groups.get(&lf_group_idx) else {
        return;
    };
    let left_in_lf = ((group_x % 8) * (group_dim / 8)) as usize;
    let top_in_lf = ((group_y % 8) * (group_dim / 8)) as usize;
    let Some(hf_meta) = &lf_group.hf_meta else {
        return;
    };
    let block_info = {
        let lf_width = (hf_meta.block_info.width() - left_in_lf).min(group_dim as usize / 8);
        let lf_height = (hf_meta.block_info.height() - top_in_lf).min(group_dim as usize / 8);
        hf_meta.block_info.as_subgrid().subgrid(
            left_in_lf..(left_in_lf + lf_width),
            top_in_lf..(top_in_lf + lf_height),
        )
    };

    let lf_regions = <[_; 3]>::try_from(&lf_image.regions_and_shifts()[..3]).unwrap();
    let [lf_x, lf_y, lf_b] = lf_image.as_color_floats();
    let (group_width, group_height) = frame_header.group_size_for(group_idx);
    let lf_base_left = group_x * group_dim / 8;
    let lf_base_top = group_y * group_dim / 8;
    let lf = [
        (lf_regions[0], lf_x),
        (lf_regions[1], lf_y),
        (lf_regions[2], lf_b),
    ]
    .map(|((lf_region, ch_shift), lf)| {
        let lf_base_left = lf_base_left.checked_add_signed(-lf_region.left).unwrap();
        let lf_base_top = lf_base_top.checked_add_signed(-lf_region.top).unwrap();
        let lf_width = (lf_region.width - lf_base_left).min(group_width.div_ceil(8));
        let lf_height = (lf_region.height - lf_base_top).min(group_height.div_ceil(8));
        let lf_base_left = (lf_base_left as usize) >> ch_shift.hshift();
        let lf_base_top = (lf_base_top as usize) >> ch_shift.vshift();
        let (lf_width, lf_height) = ch_shift.shift_size((lf_width, lf_height));
        lf.as_subgrid().subgrid(
            lf_base_left..(lf_base_left + lf_width as usize),
            lf_base_top..(lf_base_top + lf_height as usize),
        )
    });

    #[repr(align(32))]
    struct Scratch([f32; 64]);
    let [gx, gy, gb] = grid;
    for_each_varblocks(
        &block_info,
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, 0),
        |VarblockInfo {
             shifted_bx,
             shifted_by,
             dct_select,
             hf_mul,
         }| {
            debug_assert_eq!(dct_select, TransformType::Dct8);
            let left = shifted_bx * 8;
            let top = shifted_by * 8;
            let compact_x = compact_store.get_channel(shifted_bx, shifted_by, 0);
            let compact_y = compact_store.get_channel(shifted_bx, shifted_by, 1);
            let compact_b = compact_store.get_channel(shifted_bx, shifted_by, 2);
            let matrix_x = dequant_matrices.get(0, dct_select);
            let matrix_y = dequant_matrices.get(1, dct_select);
            let matrix_b = dequant_matrices.get(2, dct_select);
            let mul = [
                65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[0],
                65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[1],
                65536.0 / (quantizer.global_scale as f32 * hf_mul as f32) * qm_scale[2],
            ];
            let kx = lf_chan_corr.base_correlation_x
                + (x_from_y.get(left / 64, top / 64) as f32 / lf_chan_corr.colour_factor as f32);
            let kb = lf_chan_corr.base_correlation_b
                + (b_from_y.get(left / 64, top / 64) as f32 / lf_chan_corr.colour_factor as f32);

            let mut sx = Scratch([0.0; 64]);
            let mut sy = Scratch([0.0; 64]);
            let mut sb = Scratch([0.0; 64]);
            for row in 0..8 {
                let start = row * 8;
                let end = start + 8;
                if use_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        dequant_cfl_row_avx2_typed(
                            &compact_x[start..end],
                            &compact_y[start..end],
                            &compact_b[start..end],
                            &mut sx.0[start..end],
                            &mut sy.0[start..end],
                            &mut sb.0[start..end],
                            &matrix_x[start..end],
                            &matrix_y[start..end],
                            &matrix_b[start..end],
                            quant_bias,
                            quant_bias_numerator,
                            mul,
                            kx,
                            kb,
                        );
                    }
                } else {
                    dequant_cfl_row_scalar_typed(
                        &compact_x[start..end],
                        &compact_y[start..end],
                        &compact_b[start..end],
                        &mut sx.0[start..end],
                        &mut sy.0[start..end],
                        &mut sb.0[start..end],
                        &matrix_x[start..end],
                        &matrix_y[start..end],
                        &matrix_b[start..end],
                        quant_bias,
                        quant_bias_numerator,
                        mul,
                        kx,
                        kb,
                    );
                }
            }

            sx.0[0] = lf[0].get(shifted_bx, shifted_by);
            sy.0[0] = lf[1].get(shifted_bx, shifted_by);
            sb.0[0] = lf[2].get(shifted_bx, shifted_by);
            impls::compact_idct_8x8(&mut sx.0);
            impls::compact_idct_8x8(&mut sy.0);
            impls::compact_idct_8x8(&mut sb.0);

            for row in 0..8usize {
                let dst = left..left + 8;
                let src = row * 8;
                gx.get_row_mut(top + row)[dst.clone()].copy_from_slice(&sx.0[src..src + 8]);
                gy.get_row_mut(top + row)[dst.clone()].copy_from_slice(&sy.0[src..src + 8]);
                gb.get_row_mut(top + row)[dst].copy_from_slice(&sb.0[src..src + 8]);
            }
        },
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn chroma_from_luma_avx2(
    coeff_x: &mut [f32],
    coeff_y: &[f32],
    coeff_b: &mut [f32],
    kx: f32,
    kb: f32,
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(coeff_x.len(), coeff_y.len());
    debug_assert_eq!(coeff_x.len(), coeff_b.len());

    let vkx = _mm256_set1_ps(kx);
    let vkb = _mm256_set1_ps(kb);
    let mut i = 0usize;
    while i + 8 <= coeff_x.len() {
        let vy = _mm256_loadu_ps(coeff_y.as_ptr().add(i));
        let vx = _mm256_loadu_ps(coeff_x.as_ptr().add(i));
        let vb = _mm256_loadu_ps(coeff_b.as_ptr().add(i));
        let out_x = _mm256_fmadd_ps(vkx, vy, vx);
        let out_b = _mm256_fmadd_ps(vkb, vy, vb);
        _mm256_storeu_ps(coeff_x.as_mut_ptr().add(i), out_x);
        _mm256_storeu_ps(coeff_b.as_mut_ptr().add(i), out_b);
        i += 8;
    }

    for ((x, y), b) in coeff_x[i..]
        .iter_mut()
        .zip(&coeff_y[i..])
        .zip(coeff_b[i..].iter_mut())
    {
        let y = *y;
        *x += kx * y;
        *b += kb * y;
    }
}

pub fn transform_with_lf_grouped<S: Sample>(
    lf: &ImageWithRegion,
    coeff_out: &mut [MutableSubgrid<'_, f32>; 3],
    group_idx: u32,
    frame_header: &FrameHeader,
    lf_groups: &HashMap<u32, LfGroup<S>>,
) {
    let lf_regions = <[_; 3]>::try_from(&lf.regions_and_shifts()[..3]).unwrap();
    let [lf_x, lf_y, lf_b] = lf.as_color_floats();
    let shifts_cbycr: [_; 3] = std::array::from_fn(|idx| {
        ChannelShift::from_jpeg_upsampling(frame_header.jpeg_upsampling, idx)
    });

    let group_dim = frame_header.group_dim();
    let groups_per_row = frame_header.groups_per_row();
    let (group_width, group_height) = frame_header.group_size_for(group_idx);

    let group_x = group_idx % groups_per_row;
    let group_y = group_idx / groups_per_row;
    let lf_base_left = group_x * group_dim / 8;
    let lf_base_top = group_y * group_dim / 8;
    let lf = [
        (lf_regions[0], lf_x),
        (lf_regions[1], lf_y),
        (lf_regions[2], lf_b),
    ]
    .map(|((lf_region, shift), lf)| {
        let lf_base_left = lf_base_left.checked_add_signed(-lf_region.left).unwrap();
        let lf_base_top = lf_base_top.checked_add_signed(-lf_region.top).unwrap();
        let lf_width = (lf_region.width - lf_base_left).min(group_width.div_ceil(8));
        let lf_height = (lf_region.height - lf_base_top).min(group_height.div_ceil(8));
        let lf_base_left = lf_base_left as usize;
        let lf_base_top = lf_base_top as usize;

        let lf_base_left = lf_base_left >> shift.hshift();
        let lf_base_top = lf_base_top >> shift.vshift();
        let (lf_width, lf_height) = shift.shift_size((lf_width, lf_height));
        lf.as_subgrid().subgrid(
            lf_base_left..(lf_base_left + lf_width as usize),
            lf_base_top..(lf_base_top + lf_height as usize),
        )
    });

    let lf_group_idx = frame_header.lf_group_idx_from_group_idx(group_idx);
    let Some(lf_group) = lf_groups.get(&lf_group_idx) else {
        return;
    };
    let left_in_lf = ((group_x % 8) * (group_dim / 8)) as usize;
    let top_in_lf = ((group_y % 8) * (group_dim / 8)) as usize;

    let Some(hf_meta) = &lf_group.hf_meta else {
        for (coeff, lf) in coeff_out.iter_mut().zip(lf) {
            for y in 0..coeff.height() {
                let coeff_row = coeff.get_row_mut(y);
                let lf_row = lf.get_row(y / 8);
                for (x, v) in coeff_row.iter_mut().enumerate() {
                    *v = lf_row[x / 8];
                }
            }
        }
        return;
    };

    let block_info = {
        let lf_region = lf_regions[0].0;
        let lf_base_left = lf_base_left.checked_add_signed(-lf_region.left).unwrap();
        let lf_base_top = lf_base_top.checked_add_signed(-lf_region.top).unwrap();
        let lf_width = (lf_region.width - lf_base_left).min(group_width.div_ceil(8));
        let lf_height = (lf_region.height - lf_base_top).min(group_height.div_ceil(8));

        hf_meta.block_info.as_subgrid().subgrid(
            left_in_lf..(left_in_lf + lf_width as usize),
            top_in_lf..(top_in_lf + lf_height as usize),
        )
    };

    impls::transform_varblocks(&lf, coeff_out, shifts_cbycr, &block_info);
}

#[derive(Debug)]
struct VarblockInfo {
    shifted_bx: usize,
    shifted_by: usize,
    dct_select: TransformType,
    hf_mul: i32,
}

#[inline(always)]
fn for_each_varblocks(
    block_info: &SharedSubgrid<BlockInfo>,
    shift: ChannelShift,
    mut f: impl FnMut(VarblockInfo),
) {
    let w8 = block_info.width();
    let h8 = block_info.height();
    let vshift = shift.vshift();
    let hshift = shift.hshift();

    for by in 0..h8 {
        for bx in 0..w8 {
            let BlockInfo::Data { dct_select, hf_mul } = block_info.get(bx, by) else {
                continue;
            };
            let shifted_bx = bx >> hshift;
            let shifted_by = by >> vshift;
            if hshift != 0 || vshift != 0 {
                if (shifted_bx << hshift) != bx || (shifted_by << vshift) != by {
                    continue;
                }
                if !matches!(
                    block_info.get(shifted_bx, shifted_by),
                    BlockInfo::Data { .. }
                ) {
                    continue;
                }
            }

            f(VarblockInfo {
                shifted_bx,
                shifted_by,
                dct_select,
                hf_mul,
            })
        }
    }
}
