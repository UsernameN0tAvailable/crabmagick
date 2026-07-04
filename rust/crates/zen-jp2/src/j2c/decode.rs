//! Decoding JPEG2000 code streams.
//!
//! This is the "core" module of the crate that orchestrates all
//! stages in such a way that a given codestream is decoded into its
//! component channels.
// Parallel tile scatter uses raw pointers for disjoint writes.
#![allow(unsafe_code)]

use alloc::boxed::Box;
use alloc::vec::Vec;

use super::bitplane::{BitPlaneDecodeBuffers, BitPlaneDecodeContext};
use super::build::{CodeBlock, Decomposition, Layer, Precinct, Segment, SubBand, SubBandType};
use super::codestream::{ComponentInfo, Header, ProgressionOrder, QuantizationStyle};
use super::idwt::IDWTOutput;
use super::progression::{
    IteratorInput, ProgressionData, component_position_resolution_layer_progression,
    layer_resolution_component_position_progression,
    position_component_resolution_layer_progression,
    resolution_layer_component_position_progression,
    resolution_position_component_layer_progression,
};
use super::tag_tree::TagNode;
use super::tile::{ComponentTile, ResolutionTile, Tile};
use super::{ComponentData, bitplane, build, idwt, mct, segment, tile};
use crate::error::{DecodingError, Result, TileError, bail};
use crate::j2c::segment::MAX_BITPLANE_COUNT;
use crate::math::SimdBuffer;
use crate::reader::BitReader;
use core::ops::Range;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub(crate) fn decode<'a>(
    data: &'a [u8],
    header: &'a Header<'a>,
    ctx: &mut DecoderContext<'a>,
) -> Result<()> {
    let mut reader = BitReader::new(data);
    let tiles = tile::parse(&mut reader, header)?;

    if tiles.is_empty() {
        bail!(TileError::Invalid);
    }

    ctx.reset(header, &tiles[0]);

    #[cfg(feature = "parallel")]
    if tiles.len() > 1 {
        // SAFETY: Each tile covers a disjoint (x0..x1, y0..y1) region of
        // the image grid. `store()` writes only to rows [y0..y1] × cols
        // [x0..x1] of `channel_data[c].container`, so no two rayon tasks
        // write to the same f32. Same pattern as jxl-encoder transform.rs.
        let ptr = ChannelDataPtr::new(&mut ctx.channel_data);
        tiles.par_iter().try_for_each(move |tile| {
            trace!(
                "tile {} rect [{},{} {}x{}]",
                tile.idx,
                tile.rect.x0,
                tile.rect.y0,
                tile.rect.width(),
                tile.rect.height(),
            );
            // SAFETY: disjoint tile regions — no two tasks touch the same element.
            let channel_data = unsafe { ptr.as_mut_slice() };
            let mut tile_ctx = TileDecodeContext::default();
            let mut storage = DecompositionStorage::default();
            let progression_iterator = make_progression_iterator(tile)?;
            decode_tile(
                tile,
                header,
                progression_iterator,
                &mut tile_ctx,
                channel_data,
                &mut storage,
            )
        })?;
    } else {
        for tile in &tiles {
            trace!(
                "tile {} rect [{},{} {}x{}]",
                tile.idx,
                tile.rect.x0,
                tile.rect.y0,
                tile.rect.width(),
                tile.rect.height(),
            );

            let progression_iterator = make_progression_iterator(tile)?;

            decode_tile(
                tile,
                header,
                progression_iterator,
                &mut ctx.tile_decode_context,
                &mut ctx.channel_data,
                &mut ctx.storage,
            )?;
        }
    }

    #[cfg(not(feature = "parallel"))]
    for tile in &tiles {
        trace!(
            "tile {} rect [{},{} {}x{}]",
            tile.idx,
            tile.rect.x0,
            tile.rect.y0,
            tile.rect.width(),
            tile.rect.height(),
        );

        let progression_iterator = make_progression_iterator(tile)?;

        decode_tile(
            tile,
            header,
            progression_iterator,
            &mut ctx.tile_decode_context,
            &mut ctx.channel_data,
            &mut ctx.storage,
        )?;
    }

    // Note that this assumes that either all tiles have MCT or none of them.
    // In theory, only some could have it... But hopefully no such cursed
    // images exist!
    if tiles[0].mct {
        mct::apply_inverse(&mut ctx.channel_data, &tiles[0].component_infos, header)?;
    }

    apply_sign_shift(&mut ctx.channel_data, &header.component_infos);

    Ok(())
}

/// A decoder context for decoding JPEG2000 images.
#[derive(Default)]
pub struct DecoderContext<'a> {
    tile_decode_context: TileDecodeContext,
    /// The raw, decoded samples for each channel.
    pub(crate) channel_data: Vec<ComponentData>,
    storage: DecompositionStorage<'a>,
}

impl DecoderContext<'_> {
    fn reset(&mut self, header: &Header<'_>, initial_tile: &Tile<'_>) {
        self.tile_decode_context.reset();
        self.storage.reset();

        self.channel_data.clear();
        // TODO: SIMD Buffers should be reused across runs!
        for info in &initial_tile.component_infos {
            self.channel_data.push(ComponentData {
                container: SimdBuffer::zeros(
                    header.size_data.image_width() as usize
                        * header.size_data.image_height() as usize,
                ),
                bit_depth: info.size_info.precision,
            });
        }
    }
}

fn make_progression_iterator<'a, 'b>(
    tile: &'b Tile<'a>,
) -> Result<Box<dyn Iterator<Item = ProgressionData> + 'b>> {
    let iter_input = IteratorInput::new(tile);

    let progression_iterator: Box<dyn Iterator<Item = ProgressionData>> =
        match tile.progression_order {
            ProgressionOrder::LayerResolutionComponentPosition => {
                Box::new(layer_resolution_component_position_progression(iter_input))
            }
            ProgressionOrder::ResolutionLayerComponentPosition => {
                Box::new(resolution_layer_component_position_progression(iter_input))
            }
            ProgressionOrder::ResolutionPositionComponentLayer => Box::new(
                resolution_position_component_layer_progression(iter_input)
                    .ok_or(DecodingError::InvalidProgressionIterator)?,
            ),
            ProgressionOrder::PositionComponentResolutionLayer => Box::new(
                position_component_resolution_layer_progression(iter_input)
                    .ok_or(DecodingError::InvalidProgressionIterator)?,
            ),
            ProgressionOrder::ComponentPositionResolutionLayer => Box::new(
                component_position_resolution_layer_progression(iter_input)
                    .ok_or(DecodingError::InvalidProgressionIterator)?,
            ),
        };

    Ok(progression_iterator)
}

/// A raw-pointer wrapper that is `Send + Sync` so it can cross rayon task
/// boundaries. Only safe when the callers guarantee disjoint element access.
#[cfg(feature = "parallel")]
struct ChannelDataPtr(*mut ComponentData, usize);

#[cfg(feature = "parallel")]
// SAFETY: callers ensure no two tasks write to the same ComponentData element.
unsafe impl Send for ChannelDataPtr {}
#[cfg(feature = "parallel")]
// SAFETY: same as Send — disjoint writes only.
unsafe impl Sync for ChannelDataPtr {}

#[cfg(feature = "parallel")]
impl ChannelDataPtr {
    fn new(v: &mut Vec<ComponentData>) -> Self {
        Self(v.as_mut_ptr(), v.len())
    }
    /// # Safety
    /// Caller must ensure no two concurrent accesses touch overlapping elements.
    unsafe fn as_mut_slice(&self) -> &mut [ComponentData] {
        unsafe { core::slice::from_raw_parts_mut(self.0, self.1) }
    }
}

/// A raw-pointer wrapper for sub-band coefficient storage used by parallel
/// code-block decode. Safe only when callers guarantee disjoint writes.
#[cfg(feature = "parallel")]
struct CoeffPtr(*mut f32, usize);

#[cfg(feature = "parallel")]
// SAFETY: callers ensure no two tasks write to the same coefficient element.
unsafe impl Send for CoeffPtr {}
#[cfg(feature = "parallel")]
// SAFETY: same as Send — disjoint writes only.
unsafe impl Sync for CoeffPtr {}

#[cfg(feature = "parallel")]
impl CoeffPtr {
    fn new(v: &mut [f32]) -> Self {
        Self(v.as_mut_ptr(), v.len())
    }

    /// # Safety
    /// Caller must ensure no two concurrent accesses touch overlapping elements.
    unsafe fn as_mut_slice(&self) -> &mut [f32] {
        unsafe { core::slice::from_raw_parts_mut(self.0, self.1) }
    }
}

fn decode_tile<'a, 'b>(
    tile: &'b Tile<'a>,
    header: &Header<'_>,
    progression_iterator: Box<dyn Iterator<Item = ProgressionData> + '_>,
    tile_ctx: &mut TileDecodeContext,
    channel_data: &mut [ComponentData],
    storage: &mut DecompositionStorage<'a>,
) -> Result<()> {
    storage.reset();

    // This is the method that orchestrates all steps.

    // First, we build the decompositions, including their sub-bands, precincts
    // and code blocks.
    build::build(tile, storage)?;
    // Next, we parse the layers/segments for each code block.
    segment::parse(tile, progression_iterator, header, storage)?;
    // We then decode the bitplanes of each code block, yielding the
    // (possibly dequantized) coefficients of each code block.
    decode_component_tile_bit_planes(tile, tile_ctx, storage, header)?;

    // Unlike before, we interleave the apply_idwt and store stages
    // for each component tile so we can reuse allocations better.
    for (idx, component_info) in header.component_infos.iter().enumerate() {
        // Next, we apply the inverse discrete wavelet transform.
        idwt::apply(
            storage,
            tile_ctx,
            idx,
            header,
            component_info.wavelet_transform(),
        );
        // Finally, we store the raw samples for the tile area in the correct
        // location. Note that in case we have MCT, we are not applying it yet.
        // It will be applied in the very end once all tiles have been processed.
        // The reason we do this is that applying MCT requires access to the
        // data from _all_ components. If we didn't defer this until the end
        // we would have to collect the IDWT outputs of all components before
        // applying it. By not applying MCT here, we can get away with doing
        // IDWT and store on a per-component basis. Thus, we only need to
        // store one IDWT output at a time, allowing for better reuse of
        // allocations.
        store(
            tile,
            header,
            tile_ctx,
            &mut channel_data[idx],
            component_info,
        );
    }

    Ok(())
}

/// All decompositions for a single tile.
#[derive(Clone)]
pub(crate) struct TileDecompositions {
    pub(crate) first_ll_sub_band: usize,
    pub(crate) decompositions: Range<usize>,
}

impl TileDecompositions {
    pub(crate) fn sub_band_iter(
        &self,
        resolution: u8,
        decompositions: &[Decomposition],
    ) -> SubBandIter {
        let indices = if resolution == 0 {
            [
                self.first_ll_sub_band,
                self.first_ll_sub_band,
                self.first_ll_sub_band,
            ]
        } else {
            decompositions[self.decompositions.clone()][resolution as usize - 1].sub_bands
        };

        SubBandIter {
            next_idx: 0,
            indices,
            resolution,
        }
    }
}

#[derive(Clone)]
pub(crate) struct SubBandIter {
    resolution: u8,
    next_idx: usize,
    indices: [usize; 3],
}

impl Iterator for SubBandIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let value = if self.resolution == 0 {
            if self.next_idx > 0 {
                None
            } else {
                Some(self.indices[0])
            }
        } else if self.next_idx >= self.indices.len() {
            None
        } else {
            Some(self.indices[self.next_idx])
        };

        self.next_idx += 1;

        value
    }
}

/// A buffer so that we can reuse allocations for layers/code blocks/etc.
/// across different tiles.
#[derive(Default)]
pub(crate) struct DecompositionStorage<'a> {
    pub(crate) segments: Vec<Segment<'a>>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) code_blocks: Vec<CodeBlock>,
    pub(crate) precincts: Vec<Precinct>,
    pub(crate) tag_tree_nodes: Vec<TagNode>,
    pub(crate) coefficients: Vec<f32>,
    pub(crate) sub_bands: Vec<SubBand>,
    pub(crate) decompositions: Vec<Decomposition>,
    pub(crate) tile_decompositions: Vec<TileDecompositions>,
}

impl DecompositionStorage<'_> {
    fn reset(&mut self) {
        self.segments.clear();
        self.layers.clear();
        self.code_blocks.clear();
        // No need to clear the coefficients, as they will be resized
        // and then overridden.
        // self.coefficients.clear();
        self.precincts.clear();
        self.sub_bands.clear();
        self.decompositions.clear();
        self.tile_decompositions.clear();
        self.tag_tree_nodes.clear();
    }
}

/// A reusable context used during the decoding of a single tile.
///
/// Some of the fields are temporary in nature and reset after moving on to the
/// next tile, some contain global state.
#[derive(Default)]
pub(crate) struct TileDecodeContext {
    /// A reusable buffer for the IDWT output.
    pub(crate) idwt_output: IDWTOutput,
    /// A scratch buffer used during IDWT.
    pub(crate) idwt_scratch_buffer: Vec<f32>,
    /// A reusable context for decoding code blocks.
    pub(crate) bit_plane_decode_context: BitPlaneDecodeContext,
    /// Reusable buffers for decoding bitplanes.
    pub(crate) bit_plane_decode_buffers: BitPlaneDecodeBuffers,
}

impl TileDecodeContext {
    fn reset(&mut self) {
        // This method doesn't do anything, just keeping it there in case
        // it's needed in the future.
        // Bitplane decode context and buffers will be reset in the
        // corresponding methods. IDWT output and scratch buffer will be
        // overridden on demand, so those don't need to be reset either.
    }
}

fn decode_component_tile_bit_planes<'a>(
    tile: &Tile<'a>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    header: &Header<'_>,
) -> Result<()> {
    for (tile_decompositions_idx, component_info) in tile.component_infos.iter().enumerate() {
        // Only decode the resolution levels we actually care about.
        for resolution in
            0..component_info.num_resolution_levels() - header.skipped_resolution_levels
        {
            let tile_composition = &storage.tile_decompositions[tile_decompositions_idx];
            let sub_band_iter = tile_composition.sub_band_iter(resolution, &storage.decompositions);

            for sub_band_idx in sub_band_iter {
                decode_sub_band_bitplanes(
                    sub_band_idx,
                    resolution,
                    component_info,
                    tile_ctx,
                    storage,
                    header,
                )?;
            }
        }
    }

    Ok(())
}

fn decode_sub_band_bitplanes(
    sub_band_idx: usize,
    resolution: u8,
    component_info: &ComponentInfo,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    header: &Header<'_>,
) -> Result<()> {
    let sub_band = &storage.sub_bands[sub_band_idx];

    let dequantization_step = {
        if component_info.quantization_info.quantization_style == QuantizationStyle::NoQuantization
        {
            1.0
        } else {
            let (exponent, mantissa) =
                component_info.exponent_mantissa(sub_band.sub_band_type, resolution)?;

            let r_b = {
                let log_gain = match sub_band.sub_band_type {
                    SubBandType::LowLow => 0,
                    SubBandType::LowHigh => 1,
                    SubBandType::HighLow => 1,
                    SubBandType::HighHigh => 2,
                };

                component_info.size_info.precision as u16 + log_gain
            };

            crate::math::pow2i(r_b as i32 - exponent as i32) * (1.0 + (mantissa as f32) / 2048.0)
        }
    };

    let num_bitplanes = {
        let (exponent, _) = component_info.exponent_mantissa(sub_band.sub_band_type, resolution)?;
        // Equation (E-2)
        let num_bitplanes = (component_info.quantization_info.guard_bits as u16)
            .checked_add(exponent)
            .and_then(|x| x.checked_sub(1))
            .ok_or(DecodingError::InvalidBitplaneCount)?;

        if num_bitplanes > MAX_BITPLANE_COUNT as u16 {
            bail!(DecodingError::TooManyBitplanes);
        }

        num_bitplanes as u8
    };

    #[cfg(feature = "parallel")]
    {
        let work: Vec<usize> = sub_band
            .precincts
            .clone()
            .flat_map(|precinct_idx| storage.precincts[precinct_idx].code_blocks.clone())
            .collect();

        if work.len() > 4 {
            let coeff_ptr = CoeffPtr::new(&mut storage.coefficients[sub_band.coefficients.clone()]);
            let storage = &*storage;
            let sub_band_width = sub_band.rect.width() as usize;
            let sub_band_x0 = sub_band.rect.x0;
            let sub_band_y0 = sub_band.rect.y0;
            let sub_band_type = sub_band.sub_band_type;
            let style = &component_info.coding_style.parameters.code_block_style;
            let strict = header.strict;

            work.into_par_iter().try_for_each_init(
                TileDecodeContext::default,
                |task_ctx, code_block_idx| {
                    let code_block = &storage.code_blocks[code_block_idx];
                    bitplane::decode(
                        code_block,
                        sub_band_type,
                        num_bitplanes,
                        style,
                        task_ctx,
                        storage,
                        strict,
                    )?;

                    let x_offset = (code_block.rect.x0 - sub_band_x0) as usize;
                    let y_offset = (code_block.rect.y0 - sub_band_y0) as usize;
                    let mut base_idx = y_offset * sub_band_width + x_offset;

                    // SAFETY: each code block covers a disjoint region in the
                    // sub-band coefficient grid, so no two tasks write the same
                    // coefficient.
                    let base_store = unsafe { coeff_ptr.as_mut_slice() };

                    for coefficients in task_ctx.bit_plane_decode_context.coefficient_rows() {
                        let out_row = &mut base_store[base_idx..];

                        for (output, coefficient) in
                            out_row.iter_mut().zip(coefficients.iter().copied())
                        {
                            *output = coefficient.get() as f32;
                            *output *= dequantization_step;
                        }

                        base_idx += sub_band_width;
                    }

                    Ok::<(), crate::error::DecodeError>(())
                },
            )?;

            return Ok(());
        }
    }

    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            bitplane::decode(
                code_block,
                sub_band.sub_band_type,
                num_bitplanes,
                &component_info.coding_style.parameters.code_block_style,
                tile_ctx,
                storage,
                header.strict,
            )?;

            // Turn the signs and magnitudes into singular coefficients and
            // copy them into the sub-band.

            let x_offset = code_block.rect.x0 - sub_band.rect.x0;
            let y_offset = code_block.rect.y0 - sub_band.rect.y0;

            let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
            let mut base_idx = (y_offset * sub_band.rect.width()) as usize + x_offset as usize;

            for coefficients in tile_ctx.bit_plane_decode_context.coefficient_rows() {
                let out_row = &mut base_store[base_idx..];

                for (output, coefficient) in out_row.iter_mut().zip(coefficients.iter().copied()) {
                    *output = coefficient.get() as f32;
                    *output *= dequantization_step;
                }

                base_idx += sub_band.rect.width() as usize;
            }
        }
    }

    Ok(())
}

fn apply_sign_shift(channel_data: &mut [ComponentData], component_infos: &[ComponentInfo]) {
    use crate::math::{Level, dispatch, f32x8};

    for (channel, component_info) in channel_data.iter_mut().zip(component_infos.iter()) {
        let offset = (1_u32 << (component_info.size_info.precision - 1)) as f32;
        dispatch!(Level::new(), simd => {
            let offset_v = f32x8::splat(simd, offset);
            for chunk in channel.container.chunks_exact_mut(8) {
                let v = f32x8::from_slice(simd, chunk);
                (v + offset_v).store(chunk);
            }
        });
    }
}

fn store<'a>(
    tile: &'a Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    channel_data: &mut ComponentData,
    component_info: &ComponentInfo,
) {
    let idwt_output = &mut tile_ctx.idwt_output;

    let component_tile = ComponentTile::new(tile, component_info);
    let resolution_tile = ResolutionTile::new(
        component_tile,
        component_info.num_resolution_levels() - 1 - header.skipped_resolution_levels,
    );

    let (scale_x, scale_y) = (
        component_info.size_info.horizontal_resolution,
        component_info.size_info.vertical_resolution,
    );

    let (image_x_offset, image_y_offset) = (
        header.size_data.image_area_x_offset,
        header.size_data.image_area_y_offset,
    );

    if scale_x == 1 && scale_y == 1 {
        // If no sub-sampling, use a fast path where we copy rows of coefficients
        // at once.

        // The rect of the IDWT output corresponds to the rect of the highest
        // decomposition level of the tile, which is usually not 1:1 aligned
        // with the actual tile rectangle. We also need to account for the
        // offset of the reference grid.

        let skip_x = image_x_offset.saturating_sub(idwt_output.rect.x0);
        let skip_y = image_y_offset.saturating_sub(idwt_output.rect.y0);

        let input_row_iter = idwt_output
            .coefficients
            .chunks_exact(idwt_output.rect.width() as usize)
            .skip(skip_y as usize)
            .take(idwt_output.rect.height() as usize);

        let output_row_iter = channel_data
            .container
            .chunks_exact_mut(header.size_data.image_width() as usize)
            .skip(resolution_tile.rect.y0.saturating_sub(image_y_offset) as usize);

        for (input_row, output_row) in input_row_iter.zip(output_row_iter) {
            let input_row = &input_row[skip_x as usize..];
            let output_row = &mut output_row
                [resolution_tile.rect.x0.saturating_sub(image_x_offset) as usize..]
                [..input_row.len()];

            output_row.copy_from_slice(input_row);
        }
    } else {
        let image_width = header.size_data.image_width();
        let image_height = header.size_data.image_height();

        let x_shrink_factor = header.size_data.x_shrink_factor;
        let y_shrink_factor = header.size_data.y_shrink_factor;

        let x_offset = header
            .size_data
            .image_area_x_offset
            .div_ceil(x_shrink_factor);
        let y_offset = header
            .size_data
            .image_area_y_offset
            .div_ceil(y_shrink_factor);

        // Otherwise, copy sample by sample.
        for y in resolution_tile.rect.y0..resolution_tile.rect.y1 {
            let relative_y = (y - component_tile.rect.y0) as usize;
            let reference_grid_y = (scale_y as u32 * y) / y_shrink_factor;

            for x in resolution_tile.rect.x0..resolution_tile.rect.x1 {
                let relative_x = (x - component_tile.rect.x0) as usize;
                let reference_grid_x = (scale_x as u32 * x) / x_shrink_factor;

                let sample = idwt_output.coefficients
                    [relative_y * idwt_output.rect.width() as usize + relative_x];

                for x_position in u32::max(reference_grid_x, x_offset)
                    ..u32::min(reference_grid_x + scale_x as u32, image_width + x_offset)
                {
                    for y_position in u32::max(reference_grid_y, y_offset)
                        ..u32::min(reference_grid_y + scale_y as u32, image_height + y_offset)
                    {
                        let pos = (y_position - y_offset) as usize * image_width as usize
                            + (x_position - x_offset) as usize;

                        channel_data.container[pos] = sample;
                    }
                }
            }
        }
    }
}
