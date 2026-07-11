use crate::jxl_decode::jxl_bitstream::Bitstream;
use crate::jxl_decode::jxl_grid::{AllocTracker, MutableSubgrid};
use crate::jxl_decode::jxl_modular::{
    ChannelShift, MaConfig, Sample, image::TransformedModularSubimage,
};
use crate::jxl_decode::jxl_threadpool::JxlThreadPool;
use crate::jxl_decode::jxl_vardct::{
    CompactHfCoeffStore, HfCoeffParams, TransformType, write_hf_coeff, write_hf_coeff_compact,
    write_hf_coeff_direct,
};

use super::{HfGlobal, LfGlobalVarDct, LfGroup};
use crate::jxl_decode::jxl_frame::{FrameHeader, Result};

#[derive(Debug)]
pub struct PassGroupParams<'frame, 'buf, 'g, 'tracker, S: Sample> {
    pub frame_header: &'frame FrameHeader,
    pub lf_group: &'frame LfGroup<S>,
    pub pass_idx: u32,
    pub group_idx: u32,
    pub global_ma_config: Option<&'frame MaConfig>,
    pub modular: Option<TransformedModularSubimage<'g, S>>,
    pub vardct: Option<PassGroupParamsVardct<'frame, 'buf, 'g>>,
    pub allow_partial: bool,
    pub tracker: Option<&'tracker AllocTracker>,
    pub pool: &'frame JxlThreadPool,
}

#[derive(Debug)]
pub struct PassGroupParamsVardct<'frame, 'buf, 'g> {
    pub lf_vardct: &'frame LfGlobalVarDct,
    pub hf_global: &'frame HfGlobal,
    pub hf_coeff_output: &'buf mut [MutableSubgrid<'g, i32>; 3],
}

#[derive(Debug)]
pub struct PassGroupParamsVardctCompact<'frame, 'buf> {
    pub lf_vardct: &'frame LfGlobalVarDct,
    pub hf_global: &'frame HfGlobal,
    pub hf_coeff_compact: &'buf mut CompactHfCoeffStore,
}

fn make_hf_coeff_params<'frame, 'tracker, S: Sample>(
    frame_header: &'frame FrameHeader,
    lf_group: &'frame LfGroup<S>,
    pass_idx: u32,
    group_idx: u32,
    lf_vardct: &'frame LfGlobalVarDct,
    hf_global: &'frame HfGlobal,
    tracker: Option<&'tracker AllocTracker>,
) -> Option<HfCoeffParams<'frame, 'tracker, S>> {
    let hf_meta = lf_group.hf_meta.as_ref()?;
    let hf_pass = &hf_global.hf_passes[pass_idx as usize];
    let coeff_shift = frame_header
        .passes
        .shift
        .get(pass_idx as usize)
        .copied()
        .unwrap_or(0);

    let group_col = group_idx % frame_header.groups_per_row();
    let group_row = group_idx / frame_header.groups_per_row();
    let lf_col = (group_col % 8) as usize;
    let lf_row = (group_row % 8) as usize;
    let group_dim_blocks = (frame_header.group_dim() / 8) as usize;

    let block_info = &hf_meta.block_info;

    let block_left = lf_col * group_dim_blocks;
    let block_top = lf_row * group_dim_blocks;
    let block_width = (block_info.width() - block_left).min(group_dim_blocks);
    let block_height = (block_info.height() - block_top).min(group_dim_blocks);

    let jpeg_upsampling = frame_header.jpeg_upsampling;
    let block_info = block_info.as_subgrid().subgrid(
        block_left..(block_left + block_width),
        block_top..(block_top + block_height),
    );
    let lf_quant: Option<[_; 3]> = lf_group.lf_coeff.as_ref().map(|lf_coeff| {
        let lf_quant_channels = lf_coeff.lf_quant.image().unwrap().image_channels();
        std::array::from_fn(|idx| {
            let lf_quant = &lf_quant_channels[[1, 0, 2][idx]];
            let shift = ChannelShift::from_jpeg_upsampling(jpeg_upsampling, idx);

            let block_left = block_left >> shift.hshift();
            let block_top = block_top >> shift.vshift();
            let (block_width, block_height) =
                shift.shift_size((block_width as u32, block_height as u32));
            lf_quant.as_subgrid().subgrid(
                block_left..(block_left + block_width as usize),
                block_top..(block_top + block_height as usize),
            )
        })
    });

    Some(HfCoeffParams {
        num_hf_presets: hf_global.num_hf_presets,
        hf_block_ctx: &lf_vardct.hf_block_ctx,
        block_info,
        jpeg_upsampling,
        lf_quant,
        hf_pass,
        coeff_shift,
        tracker,
    })
}

pub fn decode_pass_group<S: Sample>(
    bitstream: &mut Bitstream,
    params: PassGroupParams<S>,
) -> Result<()> {
    let PassGroupParams {
        frame_header,
        lf_group,
        pass_idx,
        group_idx,
        global_ma_config,
        modular,
        vardct,
        allow_partial,
        tracker,
        pool,
    } = params;

    if let Some(PassGroupParamsVardct {
        lf_vardct,
        hf_global,
        hf_coeff_output,
    }) = vardct
    {
        if let Some(params) = make_hf_coeff_params(
            frame_header,
            lf_group,
            pass_idx,
            group_idx,
            lf_vardct,
            hf_global,
            tracker,
        ) {
            match write_hf_coeff(bitstream, params, hf_coeff_output) {
                Err(e) if e.unexpected_eof() && allow_partial => {
                    tracing::debug!("Partially decoded HfCoeff");
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
                Ok(_) => {}
            };
        }
    }

    if let Some(modular) = modular {
        decode_pass_group_modular(
            bitstream,
            frame_header,
            global_ma_config,
            pass_idx,
            group_idx,
            modular,
            allow_partial,
            tracker,
            pool,
        )?;
    }

    Ok(())
}

pub fn decode_pass_group_compact<S: Sample>(
    bitstream: &mut Bitstream,
    params: PassGroupParamsCompact<S>,
) -> Result<()> {
    let PassGroupParamsCompact {
        frame_header,
        lf_group,
        pass_idx,
        group_idx,
        global_ma_config,
        modular,
        vardct,
        allow_partial,
        tracker,
        pool,
    } = params;

    if let Some(PassGroupParamsVardctCompact {
        lf_vardct,
        hf_global,
        hf_coeff_compact,
    }) = vardct
    {
        if let Some(params) = make_hf_coeff_params(
            frame_header,
            lf_group,
            pass_idx,
            group_idx,
            lf_vardct,
            hf_global,
            tracker,
        ) {
            match write_hf_coeff_compact(bitstream, params, hf_coeff_compact) {
                Err(e) if e.unexpected_eof() && allow_partial => {
                    tracing::debug!("Partially decoded HfCoeff");
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
                Ok(_) => {}
            };
        }
    }

    if let Some(modular) = modular {
        decode_pass_group_modular(
            bitstream,
            frame_header,
            global_ma_config,
            pass_idx,
            group_idx,
            modular,
            allow_partial,
            tracker,
            pool,
        )?;
    }

    Ok(())
}

/// Decodes a non-progressive VarDCT pass directly into per-block coefficient scratch.
pub fn decode_hf_coeff_direct<S: Sample>(
    bitstream: &mut Bitstream,
    frame_header: &FrameHeader,
    lf_group: &LfGroup<S>,
    pass_idx: u32,
    group_idx: u32,
    lf_vardct: &LfGlobalVarDct,
    hf_global: &HfGlobal,
    tracker: Option<&AllocTracker>,
    scratch: &mut Vec<i32>,
    on_block: impl FnMut(usize, usize, TransformType, i32, [&[i32]; 3]),
) -> Result<()> {
    let Some(params) = make_hf_coeff_params(
        frame_header,
        lf_group,
        pass_idx,
        group_idx,
        lf_vardct,
        hf_global,
        tracker,
    ) else {
        return Ok(());
    };

    write_hf_coeff_direct(bitstream, params, scratch, on_block).map_err(Into::into)
}

#[derive(Debug)]
pub struct PassGroupParamsCompact<'frame, 'buf, 'g, 'tracker, S: Sample> {
    pub frame_header: &'frame FrameHeader,
    pub lf_group: &'frame LfGroup<S>,
    pub pass_idx: u32,
    pub group_idx: u32,
    pub global_ma_config: Option<&'frame MaConfig>,
    pub modular: Option<TransformedModularSubimage<'g, S>>,
    pub vardct: Option<PassGroupParamsVardctCompact<'frame, 'buf>>,
    pub allow_partial: bool,
    pub tracker: Option<&'tracker AllocTracker>,
    pub pool: &'frame JxlThreadPool,
}

#[allow(clippy::too_many_arguments)]
pub fn decode_pass_group_modular<S: Sample>(
    bitstream: &mut Bitstream,
    frame_header: &FrameHeader,
    global_ma_config: Option<&MaConfig>,
    pass_idx: u32,
    group_idx: u32,
    modular: TransformedModularSubimage<S>,
    allow_partial: bool,
    tracker: Option<&AllocTracker>,
    pool: &JxlThreadPool,
) -> Result<()> {
    if modular.is_empty() {
        return Ok(());
    }

    let mut modular = modular.recursive(bitstream, global_ma_config, tracker)?;
    let mut subimage = modular.prepare_subimage()?;
    subimage.decode(
        bitstream,
        1 + 3 * frame_header.num_lf_groups()
            + 17
            + pass_idx * frame_header.num_groups()
            + group_idx,
        allow_partial,
    )?;
    subimage.finish(pool);
    Ok(())
}
