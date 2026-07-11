use crate::jxl_decode::jxl_bitstream::Bitstream;
use crate::jxl_decode::jxl_grid::{AllocTracker, MutableSubgrid, SharedSubgrid};
use crate::jxl_decode::jxl_modular::{ChannelShift, Sample};

use crate::jxl_decode::jxl_vardct::{BlockInfo, HfBlockContext, HfPass, Result, TransformType};

/// Per-group compact HF coefficient storage.
///
/// Each `BlockInfo::Data` cell stores three contiguous channel slabs in X/Y/B order.
#[derive(Debug, Clone)]
pub struct CompactHfCoeffStore {
    pub data: Vec<i32>,
    /// cell_start[by * grid_width + bx] = offset into `data` for channel 0.
    pub cell_start: Vec<u32>,
    /// cell_size[by * grid_width + bx] = coefficients per channel. Zero for non-Data cells.
    pub cell_size: Vec<u32>,
    pub grid_width: usize,
    pub grid_height: usize,
}

impl CompactHfCoeffStore {
    pub fn new(block_info: &SharedSubgrid<BlockInfo>) -> Self {
        let grid_width = block_info.width();
        let grid_height = block_info.height();
        let grid_len = grid_width
            .checked_mul(grid_height)
            .expect("compact HF grid size overflows usize");

        let mut cell_start = vec![0u32; grid_len];
        let mut cell_size = vec![0u32; grid_len];
        let mut total_coeff = 0usize;

        for by in 0..grid_height {
            for bx in 0..grid_width {
                let flat = by * grid_width + bx;
                let BlockInfo::Data { dct_select, .. } = block_info.get(bx, by) else {
                    continue;
                };

                let (bw, bh) = dct_select.dct_select_size();
                let block_size = bw as usize * bh as usize * 64;
                let start = u32::try_from(total_coeff)
                    .expect("compact HF coefficient store exceeds u32 address space");
                let size = u32::try_from(block_size)
                    .expect("compact HF block size exceeds u32 address space");
                cell_start[flat] = start;
                cell_size[flat] = size;
                total_coeff = total_coeff
                    .checked_add(block_size * 3)
                    .expect("compact HF coefficient store size overflows usize");
            }
        }

        Self {
            data: vec![0i32; total_coeff],
            cell_start,
            cell_size,
            grid_width,
            grid_height,
        }
    }

    #[inline]
    pub fn get_channel_mut(&mut self, bx: usize, by: usize, c: usize) -> &mut [i32] {
        debug_assert!(c < 3);
        let flat = by * self.grid_width + bx;
        let start = self.cell_start[flat] as usize;
        let size = self.cell_size[flat] as usize;
        assert!(size != 0, "compact HF cell ({bx},{by}) is not a Data block");
        &mut self.data[start + c * size..start + (c + 1) * size]
    }

    #[inline]
    pub fn get_channel(&self, bx: usize, by: usize, c: usize) -> &[i32] {
        debug_assert!(c < 3);
        let flat = by * self.grid_width + bx;
        let start = self.cell_start[flat] as usize;
        let size = self.cell_size[flat] as usize;
        assert!(size != 0, "compact HF cell ({bx},{by}) is not a Data block");
        &self.data[start + c * size..start + (c + 1) * size]
    }

    /// Like `get_channel` but skips the non-zero size assert (for use inside hot loops
    /// where the block type is already guaranteed by `for_each_varblocks`).
    #[inline]
    pub fn get_channel_unchecked(&self, bx: usize, by: usize, c: usize) -> &[i32] {
        debug_assert!(c < 3);
        let flat = by * self.grid_width + bx;
        let start = self.cell_start[flat] as usize;
        let size = self.cell_size[flat] as usize;
        debug_assert!(size != 0, "compact HF cell ({bx},{by}) is not a Data block");
        &self.data[start + c * size..start + (c + 1) * size]
    }
}

/// Parameters for decoding `HfCoeff`.
#[derive(Debug)]
pub struct HfCoeffParams<'a, 'b, S: Sample> {
    pub num_hf_presets: u32,
    pub hf_block_ctx: &'a HfBlockContext,
    pub block_info: SharedSubgrid<'a, BlockInfo>,
    pub jpeg_upsampling: [u32; 3],
    pub lf_quant: Option<[SharedSubgrid<'a, S>; 3]>,
    pub hf_pass: &'a HfPass,
    pub coeff_shift: u32,
    pub tracker: Option<&'b AllocTracker>,
}

/// Decode and write HF coefficients from the bitstream.
pub fn write_hf_coeff<S: Sample>(
    bitstream: &mut Bitstream,
    params: HfCoeffParams<S>,
    hf_coeff_output: &mut [MutableSubgrid<i32>; 3],
) -> Result<()> {
    const COEFF_FREQ_CONTEXT: [u32; 63] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19,
        20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27,
        27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
    ];
    const COEFF_NUM_NONZERO_CONTEXT: [u32; 63] = [
        0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152, 152,
        180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
        206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
        206, 206, 206, 206, 206, 206, 206,
    ];

    let HfCoeffParams {
        num_hf_presets,
        hf_block_ctx,
        block_info,
        jpeg_upsampling,
        lf_quant,
        hf_pass,
        coeff_shift,
        tracker,
    } = params;
    let mut dist = hf_pass.clone_decoder();

    let HfBlockContext {
        qf_thresholds,
        lf_thresholds,
        block_ctx_map,
        num_block_clusters,
    } = hf_block_ctx;
    let lf_idx_mul =
        (lf_thresholds[0].len() + 1) * (lf_thresholds[1].len() + 1) * (lf_thresholds[2].len() + 1);
    let hf_idx_mul = qf_thresholds.len() + 1;
    let upsampling_shifts: [_; 3] =
        std::array::from_fn(|idx| ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx));
    let hshifts = upsampling_shifts.map(|shift| shift.hshift());
    let vshifts = upsampling_shifts.map(|shift| shift.vshift());

    let hfp_bits = num_hf_presets.next_power_of_two().trailing_zeros();
    let hfp = bitstream.read_bits(hfp_bits as usize)?;
    if hfp >= num_hf_presets {
        tracing::error!(hfp, num_hf_presets, "selected HF preset out of bounds");
        return Err(crate::jxl_decode::jxl_bitstream::Error::ValidationFailed(
            "selected HF preset out of bounds",
        )
        .into());
    }

    let ctx_size = 495 * *num_block_clusters;
    let cluster_map = dist.cluster_map()[(ctx_size * hfp) as usize..][..ctx_size as usize].to_vec();

    dist.begin(bitstream)?;

    let width = block_info.width();
    let height = block_info.height();
    let non_zeros_grid_lengths =
        upsampling_shifts.map(|shift| shift.shift_size((width as u32, height as u32)).0 as usize);

    let _non_zeros_grid_handle = tracker
        .map(|tracker| {
            let len =
                non_zeros_grid_lengths[0] + non_zeros_grid_lengths[1] + non_zeros_grid_lengths[2];
            tracker.alloc::<u32>(len)
        })
        .transpose()?;
    let mut non_zeros_grid_row = [
        vec![0u32; non_zeros_grid_lengths[0]],
        vec![0u32; non_zeros_grid_lengths[1]],
        vec![0u32; non_zeros_grid_lengths[2]],
    ];

    for y in 0..height {
        for x in 0..width {
            let BlockInfo::Data {
                dct_select,
                hf_mul: qf,
            } = block_info.get(x, y)
            else {
                continue;
            };
            let (w8, h8) = dct_select.dct_select_size();
            let num_blocks = w8 * h8; // power of 2
            let num_blocks_log = num_blocks.trailing_zeros();
            let order_id = dct_select.order_id();

            let lf_idx = if let Some(lf_quant) = &lf_quant {
                let mut idx = 0usize;
                for c in [0, 2, 1] {
                    let lf_thresholds = &lf_thresholds[c];
                    idx *= lf_thresholds.len() + 1;

                    let x = x >> hshifts[c];
                    let y = y >> vshifts[c];
                    let q = lf_quant[c].get(x, y);
                    for &threshold in lf_thresholds {
                        if q.to_i32() > threshold {
                            idx += 1;
                        }
                    }
                }
                idx
            } else {
                0
            };

            let hf_idx = {
                let mut idx = 0usize;
                for &threshold in qf_thresholds {
                    if qf > threshold as i32 {
                        idx += 1;
                    }
                }
                idx
            };

            for c in 0..3 {
                let ch_idx = c * 13 + order_id as usize;
                let c = [1, 0, 2][c]; // y, x, b

                let hshift = hshifts[c];
                let vshift = vshifts[c];
                let sx = x >> hshift;
                let sy = y >> vshift;
                if hshift != 0 || vshift != 0 {
                    if sx << hshift != x || sy << vshift != y {
                        continue;
                    }
                    if !matches!(block_info.get(sx, sy), BlockInfo::Data { .. }) {
                        continue;
                    }
                }

                let idx = (ch_idx * hf_idx_mul + hf_idx) * lf_idx_mul + lf_idx;
                let block_ctx = block_ctx_map[idx] as u32;
                let non_zeros_ctx = {
                    let predicted = if sy == 0 {
                        if sx == 0 {
                            32
                        } else {
                            non_zeros_grid_row[c][sx - 1]
                        }
                    } else if sx == 0 {
                        non_zeros_grid_row[c][sx]
                    } else {
                        (non_zeros_grid_row[c][sx] + non_zeros_grid_row[c][sx - 1] + 1) >> 1
                    };
                    debug_assert!(predicted < 64);

                    let idx = if predicted >= 8 {
                        4 + predicted / 2
                    } else {
                        predicted
                    };
                    block_ctx + idx * num_block_clusters
                };

                let mut non_zeros = dist.read_varint_with_multiplier_clustered(
                    bitstream,
                    cluster_map[non_zeros_ctx as usize],
                    0,
                )?;
                if non_zeros > (63 << num_blocks_log) {
                    tracing::error!(non_zeros, num_blocks, "non_zeros too large");
                    return Err(crate::jxl_decode::jxl_bitstream::Error::ValidationFailed(
                        "non_zeros too large",
                    )
                    .into());
                }

                let non_zeros_val = (non_zeros + num_blocks - 1) >> num_blocks_log;
                for dx in 0..w8 as usize {
                    non_zeros_grid_row[c][sx + dx] = non_zeros_val;
                }
                if non_zeros == 0 {
                    continue;
                }

                let coeff_grid = &mut hf_coeff_output[c];
                let mut is_prev_coeff_nonzero = (non_zeros <= num_blocks * 4) as u32;
                let order = hf_pass.order(order_id as usize, c);

                let coeff_ctx_base = block_ctx * 458 + 37 * num_block_clusters;
                let cluster_map = &cluster_map[coeff_ctx_base as usize..][..458];
                for (idx, &coeff_coord) in order[num_blocks as usize..].iter().enumerate() {
                    let coeff_ctx = {
                        let non_zeros = (non_zeros - 1) >> num_blocks_log;
                        let idx = idx >> num_blocks_log;
                        (COEFF_NUM_NONZERO_CONTEXT[non_zeros as usize] + COEFF_FREQ_CONTEXT[idx])
                            * 2
                            + is_prev_coeff_nonzero
                    };
                    let cluster = *cluster_map.get(coeff_ctx as usize).ok_or_else(|| {
                        tracing::error!("too many zeros in varblock HF coefficient");
                        crate::jxl_decode::jxl_bitstream::Error::ValidationFailed(
                            "too many zeros in varblock HF coefficient",
                        )
                    })?;
                    let ucoeff =
                        dist.read_varint_with_multiplier_clustered(bitstream, cluster, 0)?;
                    if ucoeff == 0 {
                        is_prev_coeff_nonzero = 0;
                        continue;
                    }

                    let coeff =
                        crate::jxl_decode::jxl_bitstream::unpack_signed(ucoeff) << coeff_shift;
                    let (mut dx, mut dy) = coeff_coord;
                    if dct_select.need_transpose() {
                        std::mem::swap(&mut dx, &mut dy);
                    }
                    let x = sx * 8 + dx as usize;
                    let y = sy * 8 + dy as usize;

                    *coeff_grid.get_mut(x, y) += coeff;

                    is_prev_coeff_nonzero = 1;
                    non_zeros -= 1;

                    if non_zeros == 0 {
                        break;
                    }
                }
            }
        }
    }

    dist.finalize()?;

    Ok(())
}

/// Decode and write HF coefficients from the bitstream into compact per-block buffers.
pub fn write_hf_coeff_compact<S: Sample>(
    bitstream: &mut Bitstream,
    params: HfCoeffParams<S>,
    compact_store: &mut CompactHfCoeffStore,
) -> Result<()> {
    const COEFF_FREQ_CONTEXT: [u32; 63] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19,
        20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27,
        27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
    ];
    const COEFF_NUM_NONZERO_CONTEXT: [u32; 63] = [
        0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152, 152,
        180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
        206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
        206, 206, 206, 206, 206, 206, 206,
    ];

    let HfCoeffParams {
        num_hf_presets,
        hf_block_ctx,
        block_info,
        jpeg_upsampling,
        lf_quant,
        hf_pass,
        coeff_shift,
        tracker,
    } = params;
    let mut dist = hf_pass.clone_decoder();

    let HfBlockContext {
        qf_thresholds,
        lf_thresholds,
        block_ctx_map,
        num_block_clusters,
    } = hf_block_ctx;
    let lf_idx_mul =
        (lf_thresholds[0].len() + 1) * (lf_thresholds[1].len() + 1) * (lf_thresholds[2].len() + 1);
    let hf_idx_mul = qf_thresholds.len() + 1;
    let upsampling_shifts: [_; 3] =
        std::array::from_fn(|idx| ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx));
    let hshifts = upsampling_shifts.map(|shift| shift.hshift());
    let vshifts = upsampling_shifts.map(|shift| shift.vshift());

    let hfp_bits = num_hf_presets.next_power_of_two().trailing_zeros();
    let hfp = bitstream.read_bits(hfp_bits as usize)?;
    if hfp >= num_hf_presets {
        tracing::error!(hfp, num_hf_presets, "selected HF preset out of bounds");
        return Err(crate::jxl_decode::jxl_bitstream::Error::ValidationFailed(
            "selected HF preset out of bounds",
        )
        .into());
    }

    let ctx_size = 495 * *num_block_clusters;
    let cluster_map = dist.cluster_map()[(ctx_size * hfp) as usize..][..ctx_size as usize].to_vec();

    dist.begin(bitstream)?;

    let width = block_info.width();
    let height = block_info.height();
    let non_zeros_grid_lengths =
        upsampling_shifts.map(|shift| shift.shift_size((width as u32, height as u32)).0 as usize);

    let _non_zeros_grid_handle = tracker
        .map(|tracker| {
            let len =
                non_zeros_grid_lengths[0] + non_zeros_grid_lengths[1] + non_zeros_grid_lengths[2];
            tracker.alloc::<u32>(len)
        })
        .transpose()?;
    let mut non_zeros_grid_row = [
        vec![0u32; non_zeros_grid_lengths[0]],
        vec![0u32; non_zeros_grid_lengths[1]],
        vec![0u32; non_zeros_grid_lengths[2]],
    ];

    for y in 0..height {
        for x in 0..width {
            let BlockInfo::Data {
                dct_select,
                hf_mul: qf,
            } = block_info.get(x, y)
            else {
                continue;
            };
            let (w8, h8) = dct_select.dct_select_size();
            let num_blocks = w8 * h8; // power of 2
            let num_blocks_log = num_blocks.trailing_zeros();
            let order_id = dct_select.order_id();

            let lf_idx = if let Some(lf_quant) = &lf_quant {
                let mut idx = 0usize;
                for c in [0, 2, 1] {
                    let lf_thresholds = &lf_thresholds[c];
                    idx *= lf_thresholds.len() + 1;

                    let x = x >> hshifts[c];
                    let y = y >> vshifts[c];
                    let q = lf_quant[c].get(x, y);
                    for &threshold in lf_thresholds {
                        if q.to_i32() > threshold {
                            idx += 1;
                        }
                    }
                }
                idx
            } else {
                0
            };

            let hf_idx = {
                let mut idx = 0usize;
                for &threshold in qf_thresholds {
                    if qf > threshold as i32 {
                        idx += 1;
                    }
                }
                idx
            };

            for c in 0..3 {
                let ch_idx = c * 13 + order_id as usize;
                let c = [1, 0, 2][c]; // y, x, b

                let hshift = hshifts[c];
                let vshift = vshifts[c];
                let sx = x >> hshift;
                let sy = y >> vshift;
                if hshift != 0 || vshift != 0 {
                    if sx << hshift != x || sy << vshift != y {
                        continue;
                    }
                    if !matches!(block_info.get(sx, sy), BlockInfo::Data { .. }) {
                        continue;
                    }
                }

                let idx = (ch_idx * hf_idx_mul + hf_idx) * lf_idx_mul + lf_idx;
                let block_ctx = block_ctx_map[idx] as u32;
                let non_zeros_ctx = {
                    let predicted = if sy == 0 {
                        if sx == 0 {
                            32
                        } else {
                            non_zeros_grid_row[c][sx - 1]
                        }
                    } else if sx == 0 {
                        non_zeros_grid_row[c][sx]
                    } else {
                        (non_zeros_grid_row[c][sx] + non_zeros_grid_row[c][sx - 1] + 1) >> 1
                    };
                    debug_assert!(predicted < 64);

                    let idx = if predicted >= 8 {
                        4 + predicted / 2
                    } else {
                        predicted
                    };
                    block_ctx + idx * num_block_clusters
                };

                let mut non_zeros = dist.read_varint_with_multiplier_clustered(
                    bitstream,
                    cluster_map[non_zeros_ctx as usize],
                    0,
                )?;
                if non_zeros > (63 << num_blocks_log) {
                    tracing::error!(non_zeros, num_blocks, "non_zeros too large");
                    return Err(crate::jxl_decode::jxl_bitstream::Error::ValidationFailed(
                        "non_zeros too large",
                    )
                    .into());
                }

                let non_zeros_val = (non_zeros + num_blocks - 1) >> num_blocks_log;
                for dx in 0..w8 as usize {
                    non_zeros_grid_row[c][sx + dx] = non_zeros_val;
                }
                if non_zeros == 0 {
                    continue;
                }

                let block_w = w8 as usize * 8;
                let compact = compact_store.get_channel_mut(sx, sy, c);
                let mut is_prev_coeff_nonzero = (non_zeros <= num_blocks * 4) as u32;
                let order = hf_pass.order(order_id as usize, c);

                let coeff_ctx_base = block_ctx * 458 + 37 * num_block_clusters;
                let cluster_map = &cluster_map[coeff_ctx_base as usize..][..458];
                for (idx, &coeff_coord) in order[num_blocks as usize..].iter().enumerate() {
                    let coeff_ctx = {
                        let non_zeros = (non_zeros - 1) >> num_blocks_log;
                        let idx = idx >> num_blocks_log;
                        (COEFF_NUM_NONZERO_CONTEXT[non_zeros as usize] + COEFF_FREQ_CONTEXT[idx])
                            * 2
                            + is_prev_coeff_nonzero
                    };
                    let cluster = *cluster_map.get(coeff_ctx as usize).ok_or_else(|| {
                        tracing::error!("too many zeros in varblock HF coefficient");
                        crate::jxl_decode::jxl_bitstream::Error::ValidationFailed(
                            "too many zeros in varblock HF coefficient",
                        )
                    })?;
                    let ucoeff =
                        dist.read_varint_with_multiplier_clustered(bitstream, cluster, 0)?;
                    if ucoeff == 0 {
                        is_prev_coeff_nonzero = 0;
                        continue;
                    }

                    let coeff =
                        crate::jxl_decode::jxl_bitstream::unpack_signed(ucoeff) << coeff_shift;
                    let (mut dx, mut dy) = coeff_coord;
                    if dct_select.need_transpose() {
                        std::mem::swap(&mut dx, &mut dy);
                    }

                    compact[dy as usize * block_w + dx as usize] += coeff;

                    is_prev_coeff_nonzero = 1;
                    non_zeros -= 1;

                    if non_zeros == 0 {
                        break;
                    }
                }
            }
        }
    }

    dist.finalize()?;

    Ok(())
}

/// Decodes one non-progressive, non-subsampled group's coefficients a block at a time.
///
/// The caller supplies reusable X/Y/B coefficient scratch and consumes each completed block
/// before the next one is decoded. This avoids allocating and clearing a full group buffer.
pub fn write_hf_coeff_direct<S: Sample>(
    bitstream: &mut Bitstream,
    params: HfCoeffParams<S>,
    scratch: &mut Vec<i32>,
    mut on_block: impl FnMut(usize, usize, TransformType, i32, [&[i32]; 3]),
) -> Result<()> {
    const COEFF_FREQ_CONTEXT: [u32; 63] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19,
        20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27,
        27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
    ];
    const COEFF_NUM_NONZERO_CONTEXT: [u32; 63] = [
        0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152, 152,
        180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
        206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
        206, 206, 206, 206, 206, 206, 206,
    ];

    let HfCoeffParams {
        num_hf_presets,
        hf_block_ctx,
        block_info,
        jpeg_upsampling,
        lf_quant,
        hf_pass,
        coeff_shift,
        tracker,
    } = params;
    debug_assert!(jpeg_upsampling.into_iter().all(|shift| shift == 0));
    let mut dist = hf_pass.clone_decoder();

    let HfBlockContext {
        qf_thresholds,
        lf_thresholds,
        block_ctx_map,
        num_block_clusters,
    } = hf_block_ctx;
    let lf_idx_mul =
        (lf_thresholds[0].len() + 1) * (lf_thresholds[1].len() + 1) * (lf_thresholds[2].len() + 1);
    let hf_idx_mul = qf_thresholds.len() + 1;
    let hfp_bits = num_hf_presets.next_power_of_two().trailing_zeros();
    let hfp = bitstream.read_bits(hfp_bits as usize)?;
    if hfp >= num_hf_presets {
        tracing::error!(hfp, num_hf_presets, "selected HF preset out of bounds");
        return Err(crate::jxl_decode::jxl_bitstream::Error::ValidationFailed(
            "selected HF preset out of bounds",
        )
        .into());
    }

    let ctx_size = 495 * *num_block_clusters;
    let cluster_map = dist.cluster_map()[(ctx_size * hfp) as usize..][..ctx_size as usize].to_vec();
    dist.begin(bitstream)?;

    let width = block_info.width();
    let height = block_info.height();
    let _non_zeros_grid_handle = tracker
        .map(|tracker| tracker.alloc::<u32>(width * 3))
        .transpose()?;
    let mut non_zeros_grid_row = [vec![0u32; width], vec![0u32; width], vec![0u32; width]];

    for y in 0..height {
        for x in 0..width {
            let BlockInfo::Data {
                dct_select,
                hf_mul: qf,
            } = block_info.get(x, y)
            else {
                continue;
            };
            let (w8, h8) = dct_select.dct_select_size();
            let num_blocks = w8 * h8;
            let num_blocks_log = num_blocks.trailing_zeros();
            let order_id = dct_select.order_id();
            let block_size = w8 as usize * h8 as usize * 64;
            scratch.resize(block_size * 3, 0);
            scratch[..block_size * 3].fill(0);

            for c in 0..3 {
                let lf_idx = if let Some(lf_quant) = &lf_quant {
                    let mut idx = 0usize;
                    for c in [0, 2, 1] {
                        let lf_thresholds = &lf_thresholds[c];
                        idx *= lf_thresholds.len() + 1;

                        let q = lf_quant[c].get(x, y);
                        for &threshold in lf_thresholds {
                            if q.to_i32() > threshold {
                                idx += 1;
                            }
                        }
                    }
                    idx
                } else {
                    0
                };

                let hf_idx = qf_thresholds
                    .iter()
                    .filter(|&&threshold| qf > threshold as i32)
                    .count();
                let ch_idx = c * 13 + order_id as usize;
                let c = [1, 0, 2][c];
                let idx = (ch_idx * hf_idx_mul + hf_idx) * lf_idx_mul + lf_idx;
                let block_ctx = block_ctx_map[idx] as u32;
                let non_zeros_ctx = {
                    let predicted = if y == 0 {
                        if x == 0 {
                            32
                        } else {
                            non_zeros_grid_row[c][x - 1]
                        }
                    } else if x == 0 {
                        non_zeros_grid_row[c][x]
                    } else {
                        (non_zeros_grid_row[c][x] + non_zeros_grid_row[c][x - 1] + 1) >> 1
                    };
                    debug_assert!(predicted < 64);

                    let idx = if predicted >= 8 {
                        4 + predicted / 2
                    } else {
                        predicted
                    };
                    block_ctx + idx * num_block_clusters
                };

                let mut non_zeros = dist.read_varint_with_multiplier_clustered(
                    bitstream,
                    cluster_map[non_zeros_ctx as usize],
                    0,
                )?;
                if non_zeros > (63 << num_blocks_log) {
                    tracing::error!(non_zeros, num_blocks, "non_zeros too large");
                    return Err(crate::jxl_decode::jxl_bitstream::Error::ValidationFailed(
                        "non_zeros too large",
                    )
                    .into());
                }

                let non_zeros_val = (non_zeros + num_blocks - 1) >> num_blocks_log;
                for dx in 0..w8 as usize {
                    non_zeros_grid_row[c][x + dx] = non_zeros_val;
                }
                if non_zeros == 0 {
                    continue;
                }

                let block_w = w8 as usize * 8;
                let compact = &mut scratch[c * block_size..(c + 1) * block_size];
                let mut is_prev_coeff_nonzero = (non_zeros <= num_blocks * 4) as u32;
                let order = hf_pass.order(order_id as usize, c);

                let coeff_ctx_base = block_ctx * 458 + 37 * num_block_clusters;
                let cluster_map = &cluster_map[coeff_ctx_base as usize..][..458];
                for (idx, &coeff_coord) in order[num_blocks as usize..].iter().enumerate() {
                    let coeff_ctx = {
                        let non_zeros = (non_zeros - 1) >> num_blocks_log;
                        let idx = idx >> num_blocks_log;
                        (COEFF_NUM_NONZERO_CONTEXT[non_zeros as usize] + COEFF_FREQ_CONTEXT[idx])
                            * 2
                            + is_prev_coeff_nonzero
                    };
                    let cluster = *cluster_map.get(coeff_ctx as usize).ok_or_else(|| {
                        tracing::error!("too many zeros in varblock HF coefficient");
                        crate::jxl_decode::jxl_bitstream::Error::ValidationFailed(
                            "too many zeros in varblock HF coefficient",
                        )
                    })?;
                    let ucoeff =
                        dist.read_varint_with_multiplier_clustered(bitstream, cluster, 0)?;
                    if ucoeff == 0 {
                        is_prev_coeff_nonzero = 0;
                        continue;
                    }

                    let coeff =
                        crate::jxl_decode::jxl_bitstream::unpack_signed(ucoeff) << coeff_shift;
                    let (mut dx, mut dy) = coeff_coord;
                    if dct_select.need_transpose() {
                        std::mem::swap(&mut dx, &mut dy);
                    }

                    compact[dy as usize * block_w + dx as usize] += coeff;

                    is_prev_coeff_nonzero = 1;
                    non_zeros -= 1;

                    if non_zeros == 0 {
                        break;
                    }
                }
            }

            on_block(
                x,
                y,
                dct_select,
                qf,
                [
                    &scratch[..block_size],
                    &scratch[block_size..block_size * 2],
                    &scratch[block_size * 2..block_size * 3],
                ],
            );
        }
    }

    dist.finalize()?;

    Ok(())
}
