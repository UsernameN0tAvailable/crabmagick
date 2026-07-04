// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Algorithms and constants derived from libjxl (BSD-3-Clause).
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Palette transform for modular encoding.
//!
//! Replaces multi-channel pixel data with single-channel palette indices,
//! plus a palette meta-channel. Huge win for graphics/screenshots (19-57%).

use super::channel::{Channel, ModularImage};
use crate::error::Result;

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use std::collections::HashMap;

/// Maximum number of palette colors to consider for multi-channel palette.
/// Matches libjxl `enc_params.h:121` (`palette_colors = 1 << 10`).
/// This is an encoder-only tuning parameter, not a spec limit.
pub const MAX_PALETTE_COLORS: usize = 1024;

/// Percentage threshold for per-channel compaction (ChannelCompact).
/// A channel is compacted when its unique value count is below this
/// percentage of its value range. Matches libjxl
/// `enc_params.h:118` (`channel_colors_pre_transform_percent = 95.0`).
pub const CHANNEL_COLORS_PERCENT: f32 = 95.0;

// ── Delta palette constants (72 entries, from libjxl palette.h) ──────────────

/// 72 built-in delta palette entries. Each is [R, G, B].
/// Negative indices map into this table with sign/magnitude pairing.
const DELTA_PALETTE: [[i32; 3]; 72] = [
    [0, 0, 0],
    [4, 4, 4],
    [11, 0, 0],
    [0, 0, -13],
    [0, -12, 0],
    [-10, -10, -10],
    [-18, -18, -18],
    [-27, -27, -27],
    [-18, -18, 0],
    [0, 0, -32],
    [-32, 0, 0],
    [-37, -37, -37],
    [0, -32, -32],
    [24, 24, 45],
    [50, 50, 50],
    [-45, -24, -24],
    [-24, -45, -45],
    [0, -24, -24],
    [-34, -34, 0],
    [-24, 0, -24],
    [-45, -45, -24],
    [64, 64, 64],
    [-32, 0, -32],
    [0, -32, 0],
    [-32, 0, 32],
    [-24, -45, -24],
    [45, 24, 45],
    [24, -24, -45],
    [-45, -24, 24],
    [80, 80, 80],
    [64, 0, 0],
    [0, 0, -64],
    [0, -64, -64],
    [-24, -24, 45],
    [96, 96, 96],
    [64, 64, 0],
    [45, -24, -24],
    [34, -34, 0],
    [112, 112, 112],
    [24, -45, -45],
    [45, 45, -24],
    [0, -32, 32],
    [24, -24, 45],
    [0, 96, 96],
    [45, -24, 24],
    [24, -45, -24],
    [-24, -45, 24],
    [0, -64, 0],
    [96, 0, 0],
    [128, 128, 128],
    [64, 0, 64],
    [144, 144, 144],
    [96, 96, 0],
    [-36, -36, 36],
    [45, -24, -45],
    [45, -45, -24],
    [0, 0, -96],
    [0, 128, 128],
    [0, 96, 0],
    [45, 24, -45],
    [-128, 0, 0],
    [24, -45, 24],
    [-45, 24, -45],
    [64, 0, -64],
    [64, -64, -64],
    [96, 0, 96],
    [45, -45, 24],
    [24, 45, -45],
    [64, 64, -64],
    [128, 128, 0],
    [0, 0, -128],
    [-24, 45, -45],
];

/// 5x5x5 color cube for the larger implicit palette.
const LARGE_CUBE: i32 = 5;
/// 4x4x4 color cube for the smaller implicit palette.
const SMALL_CUBE: i32 = 4;
const SMALL_CUBE_BITS: u32 = 2;
/// Offset where the large cube starts (after small cube: 4^3 = 64).
const LARGE_CUBE_OFFSET: i32 = SMALL_CUBE * SMALL_CUBE * SMALL_CUBE; // 64
/// Total implicit palette size: 64 + 125 = 189.
const IMPLICIT_PALETTE_SIZE: usize =
    (LARGE_CUBE_OFFSET + LARGE_CUBE * LARGE_CUBE * LARGE_CUBE) as usize;
/// Minimum implicit palette index (negative: -(2*72-1) = -143).
const MIN_IMPLICIT_PALETTE_INDEX: i32 = -(2 * 72 - 1);

/// Get the value for implicit palette entry at given index and channel.
/// For index < 0: delta palette. For index >= palette_size: implicit color cubes.
/// For 0 <= index < palette_size: explicit palette lookup from the palette channel.
fn get_palette_value(
    palette_data: &[i32],
    palette_row_stride: usize,
    index: i32,
    c: usize,
    palette_size: i32,
    bit_depth: i32,
) -> i32 {
    if index < 0 {
        if c >= 3 {
            return 0;
        }
        let idx = -(index + 1);
        let idx = idx % (1 + 2 * (DELTA_PALETTE.len() as i32 - 1));
        let multipliers = [-1i32, 1];
        let mut result =
            DELTA_PALETTE[((idx + 1) >> 1) as usize][c] * multipliers[(idx & 1) as usize];
        if bit_depth > 8 {
            result *= 1 << (bit_depth - 8);
        }
        result
    } else if index >= palette_size && index < palette_size + LARGE_CUBE_OFFSET {
        if c >= 3 {
            return 0;
        }
        let idx = index - palette_size;
        let val = (idx >> (c as u32 * SMALL_CUBE_BITS)) % SMALL_CUBE;
        scale::<4>(val as u64, bit_depth as u64) + (1 << bit_depth.saturating_sub(3).max(0))
    } else if index >= palette_size + LARGE_CUBE_OFFSET {
        if c >= 3 {
            return 0;
        }
        let mut idx = index - palette_size - LARGE_CUBE_OFFSET;
        match c {
            1 => idx /= LARGE_CUBE,
            2 => idx /= LARGE_CUBE * LARGE_CUBE,
            _ => {}
        }
        scale::<4>(idx as u64 % LARGE_CUBE as u64, bit_depth as u64)
    } else {
        // Explicit palette: index is within [0, palette_size)
        palette_data[c * palette_row_stride + index as usize]
    }
}

/// Scale a value to bit_depth range: (value * ((1 << bit_depth) - 1)) / denom.
/// Uses right-shift for denom=4 (matching libjxl's Scale<4>).
fn scale<const DENOM: u64>(value: u64, bit_depth: u64) -> i32 {
    ((value * ((1u64 << bit_depth) - 1)) / DENOM) as i32
}

/// Perceptual color distance metric matching libjxl's ColorDistance.
/// Weighted L2 + brightness-dependent weights + luminance L1 term.
fn color_distance(a: &[f32], b: &[i32]) -> f32 {
    let nb = a.len().min(b.len());
    let mut distance = 0.0f32;
    let mut ave3 = 0.0f32;
    if nb >= 3 {
        ave3 = (a[0] + b[0] as f32 + a[1] + b[1] as f32 + a[2] + b[2] as f32) * (1.21 / 3.0);
    }
    let mut sum_a = 0.0f32;
    let mut sum_b = 0.0f32;
    for c in 0..nb {
        let difference = a[c] - b[c] as f32;
        let mut weight: f32 = if c == 0 {
            3.0
        } else if c == 1 {
            5.0
        } else {
            2.0
        };
        if c < 3 && (a[c] + b[c] as f32 >= ave3) {
            let add_w = [1.15f32, 1.15, 1.12];
            weight += add_w[c];
            if c == 2 && (a[2] + b[2] as f32) < 1.22 * ave3 {
                weight -= 0.5;
            }
        }
        distance += difference * difference * weight * weight;
        let sum_weight: f32 = if c == 0 {
            3.0
        } else if c == 1 {
            5.0
        } else {
            1.0
        };
        sum_a += a[c] * sum_weight;
        sum_b += b[c] as f32 * sum_weight;
    }
    distance *= 4.0;
    let sum_difference = sum_a - sum_b;
    distance += sum_difference * sum_difference;
    distance
}

/// Reorder a palette using greedy nearest-neighbor traversal through color space.
///
/// Minimizes squeeze residuals at color-block boundaries: consecutive palette
/// indices are close in color space, so index jumps at image boundaries produce
/// small residual values → small ANS alphabet → better compression.
///
/// Uses libjxl's perceptual `color_distance` metric. O(N²) for N palette entries.
/// Reorder a palette using greedy nearest-neighbor traversal through color space.
///
/// Minimizes squeeze residuals at color-block boundaries: consecutive palette
/// indices are close in color space, so index jumps at image boundaries produce
/// small residual values → small ANS alphabet → better compression.
///
/// Uses libjxl's perceptual `color_distance` metric. O(N²) for N palette entries.
/// This variant works with `[i32; 4]` keys (nc significant channels, rest padded 0).
fn nearest_neighbor_palette_order_arr(palette: Vec<[i32; 4]>, nc: usize) -> Vec<[i32; 4]> {
    let n = palette.len();
    if n <= 1 {
        return palette;
    }

    let mut used = vec![false; n];
    let mut ordered = Vec::with_capacity(n);

    // Start from the darkest (lowest luminance) color to match libjxl convention.
    let start = palette
        .iter()
        .enumerate()
        .min_by_key(|(_, c)| c[0] as i64 * 299 + c[1] as i64 * 587 + c[2] as i64 * 114)
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut current = start;
    used[current] = true;
    ordered.push(palette[current]);

    for _ in 1..n {
        let mut cur_f32 = [0.0f32; 4];
        for i in 0..nc {
            cur_f32[i] = palette[current][i] as f32;
        }
        let next = (0..n)
            .filter(|&i| !used[i])
            .min_by(|&a, &b| {
                let da = color_distance(&cur_f32[..nc], &palette[a][..nc]);
                let db = color_distance(&cur_f32[..nc], &palette[b][..nc]);
                da.partial_cmp(&db).unwrap_or(core::cmp::Ordering::Equal)
            })
            .unwrap();
        used[next] = true;
        ordered.push(palette[next]);
        current = next;
    }

    ordered
}

/// Quantize a color to an implicit palette index (high-quality large cube).
fn quantize_to_implicit_palette(
    color: &[i32],
    palette_size: i32,
    bit_depth: i32,
    high_quality: bool,
) -> i32 {
    let quant = (1i64 << bit_depth) - 1;
    let half = if bit_depth > 1 {
        1i64 << (bit_depth - 1)
    } else {
        0
    };
    let mut index = 0i32;
    let mut multiplier = 1i32;

    if high_quality {
        for &value in color.iter().take(3) {
            let quantized = (((LARGE_CUBE as i64 - 1) * value as i64 + half) / quant) as i32;
            index += quantized * multiplier;
            multiplier *= LARGE_CUBE;
        }
        index + palette_size + LARGE_CUBE_OFFSET
    } else {
        for (c, &value) in color.iter().enumerate().take(3) {
            let _ = c;
            let mut v = value as i64 - (1i64 << bit_depth.saturating_sub(3).max(0));
            v = v.max(0);
            let mut quantized = (((LARGE_CUBE as i64 - 1) * v + half) / quant) as i32;
            if quantized > SMALL_CUBE - 1 {
                quantized = SMALL_CUBE - 1;
            }
            index += quantized * multiplier;
            multiplier *= SMALL_CUBE;
        }
        index + palette_size
    }
}

/// Symmetric rounding around zero.
fn round_int(value: i32, div: i32) -> i32 {
    if value < 0 {
        return -round_int(-value, div);
    }
    (value + div / 2) / div
}

#[inline]
fn read_color4(channels: &[Channel], x: usize, y: usize, nc: usize) -> [i32; 4] {
    let mut color = [0i32; 4];
    for i in 0..nc {
        color[i] = channels[i].get(x, y);
    }
    color
}

#[inline]
fn color_to_f32(color: &[i32; 4], nc: usize) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for i in 0..nc {
        out[i] = color[i] as f32;
    }
    out
}

#[inline]
fn clamped_gradient_predict(
    prev_row: Option<&[[i32; 3]]>,
    qrow: &[[i32; 3]],
    x: usize,
    nc: usize,
) -> [i32; 3] {
    let mut predictions = [0i32; 3];
    for c in 0..nc {
        let w = if x > 0 { qrow[x - 1][c] } else { 0 };
        let n = prev_row.map_or(0, |r| r[x][c]);
        let nw = if x > 0 {
            prev_row.map_or(0, |r| r[x - 1][c])
        } else {
            0
        };
        let grad = n as i64 + w as i64 - nw as i64;
        let lo = n.min(w) as i64;
        let hi = n.max(w) as i64;
        predictions[c] = grad.clamp(lo, hi) as i32;
    }
    predictions
}

/// Result of palette analysis.
pub struct PaletteAnalysis {
    /// Whether palette transform is beneficial.
    pub use_palette: bool,
    /// Number of unique colors found.
    pub num_colors: usize,
    /// The palette (colors as fixed-size [i32; 4], padded with 0 for num_c < 4).
    pub palette: Vec<[i32; 4]>,
    /// Map from color (as [i32; 4], padded) to palette index. No per-lookup allocation.
    pub color_to_index: HashMap<[i32; 4], i32>,
}

/// Analyze whether a single channel benefits from ChannelCompact (per-channel palette).
///
/// Matches libjxl's single-channel palette heuristic (enc_modular.cc:395-438):
/// if less than `channel_colors_percent`% of the value range actually occurs,
/// and the palette is less than 6.25% the pixel count (nb_pixels/16), compact it.
///
/// Returns `Some(PaletteAnalysis)` with `num_c=1` palette if beneficial.
pub fn analyze_channel_compact(
    channel: &Channel,
    channel_colors_percent: f32,
) -> Option<PaletteAnalysis> {
    let width = channel.width();
    let height = channel.height();
    let nb_pixels = width * height;

    // Find min and max values in the channel
    let mut min_val = i32::MAX;
    let mut max_val = i32::MIN;
    for y in 0..height {
        for x in 0..width {
            let v = channel.get(x, y);
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
    }

    if min_val > max_val {
        return None; // empty channel
    }

    let range = (max_val as i64 - min_val as i64 + 1) as usize;

    // Matching libjxl: nb_colors = min(nb_pixels/16, channel_colors_percent/100 * range)
    let nb_colors_limit =
        (nb_pixels / 16).min((channel_colors_percent as f64 / 100.0 * range as f64) as usize);

    if nb_colors_limit == 0 {
        return None;
    }

    // Collect unique values, bail early if too many
    let mut unique_values = alloc::collections::BTreeSet::new();
    for y in 0..height {
        for x in 0..width {
            unique_values.insert(channel.get(x, y));
            if unique_values.len() > nb_colors_limit {
                return None;
            }
        }
    }

    let actual_unique = unique_values.len();
    if actual_unique <= 1 {
        return None; // single value, no benefit
    }

    // Build palette: sorted unique values as [i32; 4] (single-channel, rest padded 0)
    let palette: Vec<[i32; 4]> = unique_values.iter().map(|&v| [v, 0, 0, 0]).collect();
    let mut color_to_index = HashMap::new();
    for (i, &color) in palette.iter().enumerate() {
        color_to_index.insert(color, i as i32);
    }

    Some(PaletteAnalysis {
        use_palette: true,
        num_colors: actual_unique,
        palette,
        color_to_index,
    })
}

/// Analyze an image to determine if palette transform is beneficial.
///
/// For lossless: palette is beneficial if num_unique_colors <= max_colors.
/// `begin_c` and `num_c` specify the channel range to palettize (e.g., 0..3 for RGB).
pub fn analyze_palette(
    image: &ModularImage,
    begin_c: usize,
    num_c: usize,
    max_colors: usize,
) -> PaletteAnalysis {
    let width = image.width();
    let height = image.height();

    // Collect unique colors using thread-local HashMaps merged at the end.
    // Each thread scans its rows independently; merge counts at the end.
    let nc = num_c.min(4);
    let src_channels: &[Channel] =
        &image.channels[begin_c..begin_c + num_c.min(image.channels.len() - begin_c)];
    let color_counts: HashMap<[i32; 4], u32> = crate::parallel::parallel_accumulate(
        height,
        64, // only parallelize when there are enough rows
        || HashMap::<[i32; 4], u32>::new(),
        |mut acc, y| {
            let rows: [&[i32]; 4] =
                core::array::from_fn(|i| if i < nc { src_channels[i].row(y) } else { &[] });
            let mut key = [0i32; 4];
            for x in 0..width {
                for i in 0..nc {
                    key[i] = rows[i][x];
                }
                *acc.entry(key).or_insert(0) += 1;
            }
            acc
        },
        |mut a, b| {
            for (color, count) in b {
                *a.entry(color).or_insert(0) += count;
            }
            a
        },
    );

    let num_colors = color_counts.len();

    // Palette is not useful when:
    // - Too many colors (exceeds max)
    // - Only 1 color (no benefit)
    // - Colors ≥ 50% of pixel count (palette + index overhead exceeds savings)
    //   The palette channel itself is (num_colors × num_c) values, so the break-even
    //   point is roughly when palette overhead equals channel elimination savings.
    let num_pixels = width * height;
    // Palette is less useful when color count approaches pixel count:
    // palette channel overhead (num_colors × num_c values) exceeds savings.
    let too_many_relative = num_c >= 2 && num_colors * 2 > num_pixels;
    if num_colors > max_colors || num_colors <= 1 || too_many_relative {
        return PaletteAnalysis {
            use_palette: false,
            num_colors,
            palette: Vec::new(),
            color_to_index: HashMap::new(),
        };
    }

    // Sort palette using nearest-neighbor greedy ordering for better compression.
    // This minimizes squeeze residuals at color-block boundaries by ensuring
    // adjacent palette indices are close in color space.
    // For single-channel: just sort by value (already optimal).
    // Sort the raw keys first to make ordering deterministic regardless of HashMap
    // iteration order (which is non-deterministic across runs and parallel builds).
    let mut palette: Vec<[i32; 4]> = color_counts.keys().copied().collect();
    palette.sort(); // normalize input order before nearest-neighbor

    if num_c >= 3 {
        palette = nearest_neighbor_palette_order_arr(palette, num_c);
    }

    // Build index map (no allocation per lookup)
    let mut color_to_index = HashMap::with_capacity(palette.len());
    for (i, &color) in palette.iter().enumerate() {
        color_to_index.insert(color, i as i32);
    }

    PaletteAnalysis {
        use_palette: true,
        num_colors,
        palette,
        color_to_index,
    }
}

/// Apply the palette transform to a modular image (lossless).
///
/// Replaces channels `begin_c..begin_c+num_c` with:
/// - A palette meta-channel (width=nb_colors, height=num_c) at position `begin_c`
/// - An index channel (same width/height as original) at position `begin_c+1`
///
/// Returns the number of colors in the palette.
pub fn apply_palette(
    image: &mut ModularImage,
    begin_c: usize,
    num_c: usize,
    analysis: &PaletteAnalysis,
) -> Result<usize> {
    let width = image.width();
    let height = image.height();
    let nb_colors = analysis.palette.len();
    let nc = num_c.min(4);

    // Create palette meta-channel.
    // In JXL, the palette is stored as nb_colors wide, num_c high.
    // palette_channel.get(i, c) = palette[i][c]
    let mut palette_channel = Channel::new(nb_colors, num_c)?;
    for (i, &color) in analysis.palette.iter().enumerate() {
        for c in 0..num_c {
            palette_channel.set(i, c, color[c]);
        }
    }

    // Create index channel — write directly into output rows in parallel (no per-row Vec).
    let mut index_channel = Channel::new(width, height)?;
    let src: &[Channel] = &image.channels[begin_c..begin_c + num_c];
    let cti: &HashMap<[i32; 4], i32> = &analysis.color_to_index;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        index_channel
            .data_mut()
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, out_row)| {
                let rows: [&[i32]; 4] =
                    core::array::from_fn(|i| if i < nc { src[i].row(y) } else { &[] });
                let mut key = [0i32; 4];
                for (x, slot) in out_row.iter_mut().enumerate() {
                    for i in 0..nc {
                        key[i] = rows[i][x];
                    }
                    *slot = cti[&key];
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (y, out_row) in index_channel.data_mut().chunks_mut(width).enumerate() {
            let rows: [&[i32]; 4] =
                core::array::from_fn(|i| if i < nc { src[i].row(y) } else { &[] });
            let mut key = [0i32; 4];
            for (x, slot) in out_row.iter_mut().enumerate() {
                for i in 0..nc {
                    key[i] = rows[i][x];
                }
                *slot = cti[&key];
            }
        }
    }

    // Replace the original channels with palette meta-channel + index channel.
    // Consume channels directly to avoid unnecessary clones.
    let orig_channels = core::mem::take(&mut image.channels);
    let mut new_channels = Vec::with_capacity(orig_channels.len() - num_c + 2);
    let mut palette_ch = Some(palette_channel);
    let mut index_ch = Some(index_channel);
    for (i, ch) in orig_channels.into_iter().enumerate() {
        if i == begin_c {
            new_channels.push(palette_ch.take().unwrap());
            new_channels.push(index_ch.take().unwrap());
        }
        if i < begin_c || i >= begin_c + num_c {
            new_channels.push(ch);
        }
    }
    if let Some(palette_ch) = palette_ch {
        new_channels.push(palette_ch);
        new_channels.push(index_ch.unwrap());
    }
    image.channels = new_channels;

    Ok(nb_colors)
}

/// Apply palette transform from an immutable original image, producing a NEW palettized image.
///
/// This avoids the expensive clone of the full original image (e.g., 192MB for a 4000×4000 RGB
/// image). Only the index channel (~64MB) and tiny meta channel are allocated — the original
/// RGB channels are read but never copied.
///
/// Returns the palettized image (2 channels: meta + index) and the number of colors.
pub fn apply_palette_from_ref(
    image: &ModularImage,
    begin_c: usize,
    num_c: usize,
    analysis: &PaletteAnalysis,
) -> Result<(ModularImage, usize)> {
    let width = image.width();
    let height = image.height();
    let nb_colors = analysis.palette.len();
    let nc = num_c.min(4);

    // Build palette meta-channel (tiny: nb_colors wide, num_c tall)
    let mut palette_channel = Channel::new(nb_colors, num_c)?;
    for (i, &color) in analysis.palette.iter().enumerate() {
        for c in 0..num_c {
            palette_channel.set(i, c, color[c]);
        }
    }

    // Build index channel by looking up each pixel in the color→index map.
    // Reads from the original channels (no copy of original data needed).
    let mut index_channel = Channel::new(width, height)?;
    let src: &[Channel] = &image.channels[begin_c..begin_c + num_c];
    let cti: &HashMap<[i32; 4], i32> = &analysis.color_to_index;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        index_channel
            .data_mut()
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, out_row)| {
                let rows: [&[i32]; 4] =
                    core::array::from_fn(|i| if i < nc { src[i].row(y) } else { &[] });
                let mut key = [0i32; 4];
                for (x, slot) in out_row.iter_mut().enumerate() {
                    for i in 0..nc {
                        key[i] = rows[i][x];
                    }
                    *slot = cti[&key];
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (y, out_row) in index_channel.data_mut().chunks_mut(width).enumerate() {
            let rows: [&[i32]; 4] =
                core::array::from_fn(|i| if i < nc { src[i].row(y) } else { &[] });
            let mut key = [0i32; 4];
            for (x, slot) in out_row.iter_mut().enumerate() {
                for i in 0..nc {
                    key[i] = rows[i][x];
                }
                *slot = cti[&key];
            }
        }
    }

    // Build the output modular image with only palette + index channels.
    // Any channels outside [begin_c, begin_c+num_c) are cloned (small extra channels).
    let mut new_channels = Vec::with_capacity(image.channels.len() - num_c + 2);
    let mut palette_ch = Some(palette_channel);
    let mut index_ch = Some(index_channel);
    for (i, ch) in image.channels.iter().enumerate() {
        if i == begin_c {
            new_channels.push(palette_ch.take().unwrap());
            new_channels.push(index_ch.take().unwrap());
        }
        if i < begin_c || i >= begin_c + num_c {
            new_channels.push(ch.clone());
        }
    }
    if let Some(palette_ch) = palette_ch {
        new_channels.push(palette_ch);
        new_channels.push(index_ch.unwrap());
    }

    let palettized = ModularImage {
        channels: new_channels,
        bit_depth: image.bit_depth,
        is_grayscale: image.is_grayscale,
        has_alpha: false, // palette transform fuses all color channels
    };
    Ok((palettized, nb_colors))
}

/// Result of lossy palette analysis and application.
pub struct LossyPaletteResult {
    /// Number of explicit palette colors (not counting deltas).
    pub nb_colors: usize,
    /// Number of delta palette entries.
    pub nb_deltas: usize,
    /// Predictor to use (0=Zero if no deltas used, or specified predictor).
    pub predictor: u8,
}

#[allow(clippy::needless_range_loop)]
/// Analyze and apply lossy palette transform with delta palette support.
///
/// Two-pass algorithm matching libjxl's FwdPalette:
/// 1. First pass discovers frequent color deltas (residuals between predicted and actual colors).
/// 2. Second pass applies the palette with error diffusion, using discovered deltas.
///
/// Returns `Some(LossyPaletteResult)` on success, `None` if palette is not beneficial.
pub fn apply_lossy_palette(
    image: &mut ModularImage,
    begin_c: usize,
    num_c: usize,
    max_palette_colors: usize,
) -> Option<LossyPaletteResult> {
    if num_c < 3 || image.bit_depth < 8 {
        return None;
    }

    let width = image.width();
    let height = image.height();
    let bit_depth = image.bit_depth.min(24) as i32;
    let num_pixels = width * height;
    let max_deltas: usize = 128;
    let nc = num_c.min(4);
    let rgb_nc = num_c.min(3);
    let src_channels: &[Channel] = &image.channels[begin_c..begin_c + num_c];

    // ── Pass 1: Discover frequent deltas ─────────────────────────────────────

    // Build candidate palette from cross-pattern frequent colors
    let mut color_freq: HashMap<[i32; 4], usize> = HashMap::new();
    let mut cross_colors: HashMap<[i32; 4], usize> = HashMap::new();

    // Count all color frequencies and cross-pattern colors
    for y in 0..height {
        for x in 0..width {
            let color = read_color4(src_channels, x, y, nc);
            *color_freq.entry(color).or_insert(0) += 1;

            // Check cross pattern (center matches all 4 neighbors)
            if x > 0 && x + 1 < width && y > 0 && y + 1 < height {
                let makes_cross =
                    [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)]
                        .iter()
                        .all(|&(dx, dy)| {
                            let nx = (x as i32 + dx) as usize;
                            let ny = (y as i32 + dy) as usize;
                            (0..nc).all(|i| src_channels[i].get(nx, ny) == color[i])
                        });
                if makes_cross {
                    *cross_colors.entry(color).or_insert(0) += 1;
                }
            }
        }
    }

    // Build candidate palette: cross-pattern colors that are frequent enough
    let freq_threshold = 5 + num_pixels / 100; // 0.01 image fraction
    let mut candidate_palette: Vec<[i32; 4]> = Vec::new();
    let mut palette_set: HashMap<[i32; 4], bool> = HashMap::new();

    for (color, count) in &cross_colors {
        if *count > freq_threshold && palette_set.insert(*color, true).is_none() {
            candidate_palette.push(*color);
        }
    }

    // Add remaining unique colors up to max
    // Build implicit color lookup for dedup
    let mut implicit_colors: Vec<[i32; 4]> = Vec::with_capacity(IMPLICIT_PALETTE_SIZE);
    let mut is_implicit: HashMap<[i32; 4], bool> = HashMap::new();
    for k in 0..IMPLICIT_PALETTE_SIZE {
        let mut color = [0i32; 4];
        for (c, slot) in color.iter_mut().enumerate().take(nc) {
            *slot = get_palette_value(&[], 0, k as i32, c, 0, bit_depth);
        }
        is_implicit.insert(color, true);
        implicit_colors.push(color);
    }

    let mut implicit_used = 0usize;
    for y in 0..height {
        if candidate_palette.len() >= max_palette_colors {
            break;
        }
        for x in 0..width {
            if candidate_palette.len() >= max_palette_colors {
                break;
            }
            let color = read_color4(src_channels, x, y, nc);
            if palette_set.contains_key(&color) {
                continue;
            }
            palette_set.insert(color, true);
            if is_implicit.contains_key(&color) {
                implicit_used += 1;
            } else {
                candidate_palette.push(color);
                if candidate_palette.len() > max_palette_colors {
                    // Too many colors for palette, but we can still do lossy
                    // by using implicit cubes for the rest
                    candidate_palette.pop();
                    break;
                }
            }
        }
    }

    // If very few candidate colors and no implicit usage, palette won't help
    if candidate_palette.is_empty() && implicit_used <= 1 {
        return None;
    }

    // Pass 1: collect deltas (residuals from prediction) for each pixel
    // Use a simple quantized image to get predictions
    let mut deltas_r: Vec<i32> = Vec::with_capacity(num_pixels);
    let mut deltas_g: Vec<i32> = Vec::with_capacity(num_pixels);
    let mut deltas_b: Vec<i32> = Vec::with_capacity(num_pixels);
    let mut delta_distances: Vec<f32> = Vec::with_capacity(num_pixels);

    // For pass 1, build inv_palette for quick lookup
    let nb_deltas_pass1 = 0usize; // No deltas in pass 1
    let nb_colors_pass1 = candidate_palette.len();
    let total_palette_size = nb_deltas_pass1 + nb_colors_pass1;

    // Build palette data array for get_palette_value lookups
    let palette_row_stride = total_palette_size;
    let mut palette_data = vec![0i32; nc * palette_row_stride.max(1)];
    for (i, color) in candidate_palette.iter().enumerate() {
        for c in 0..nc {
            palette_data[c * palette_row_stride + nb_deltas_pass1 + i] = color[c];
        }
    }

    // Build a fast lookup from color -> palette index
    let mut inv_palette: HashMap<[i32; 4], usize> = HashMap::new();
    for (i, color) in candidate_palette.iter().enumerate() {
        inv_palette.insert(*color, nb_deltas_pass1 + i);
    }
    // Add implicit colors to inv_palette
    for (k, color) in implicit_colors.iter().enumerate() {
        inv_palette.entry(*color).or_insert(total_palette_size + k);
    }

    // Pass 1 quantization: find best index for each pixel, record deltas
    let mut quant_rows: Vec<Vec<[i32; 3]>> = Vec::with_capacity(height);
    for y in 0..height {
        let mut qrow: Vec<[i32; 3]> = Vec::with_capacity(width);
        let prev_row = if y > 0 {
            Some(quant_rows[y - 1].as_slice())
        } else {
            None
        };
        for x in 0..width {
            let color = read_color4(src_channels, x, y, nc);
            let color_f = color_to_f32(&color, rgb_nc);
            let predictions = clamped_gradient_predict(prev_row, &qrow, x, rgb_nc);

            // Try all palette entries, implicit cubes
            let mut best_index = 0i32;
            let mut best_distance = f32::INFINITY;
            let mut best_val = [0i32; 3];

            let try_index =
                |index: i32, best_idx: &mut i32, best_dist: &mut f32, best_v: &mut [i32; 3]| {
                    let mut qval = [0i32; 3];
                    for (c, slot) in qval.iter_mut().enumerate().take(rgb_nc) {
                        *slot = get_palette_value(
                            &palette_data,
                            palette_row_stride,
                            index,
                            c,
                            total_palette_size as i32,
                            bit_depth,
                        );
                    }
                    let cd = 32.0 / (1i64 << (2 * (bit_depth - 8)).max(0)) as f32
                        * color_distance(&color_f[..rgb_nc], &qval[..rgb_nc]);

                    let index_penalty: f32 = if index == -1 {
                        -124.0
                    } else if index < 0 {
                        -2.0 * index as f32
                    } else if index < total_palette_size as i32 {
                        150.0
                    } else if index < total_palette_size as i32 + LARGE_CUBE_OFFSET {
                        70.0
                    } else {
                        256.0
                    };
                    let dist = cd + index_penalty;
                    if dist < *best_dist {
                        *best_dist = dist;
                        *best_idx = index;
                        *best_v = qval;
                    }
                };

            // Try all explicit palette + implicit delta entries
            for idx in MIN_IMPLICIT_PALETTE_INDEX..total_palette_size as i32 {
                try_index(idx, &mut best_index, &mut best_distance, &mut best_val);
            }
            // Try implicit color cubes
            try_index(
                quantize_to_implicit_palette(
                    &color[..rgb_nc],
                    total_palette_size as i32,
                    bit_depth,
                    false,
                ),
                &mut best_index,
                &mut best_distance,
                &mut best_val,
            );
            try_index(
                quantize_to_implicit_palette(
                    &color[..rgb_nc],
                    total_palette_size as i32,
                    bit_depth,
                    true,
                ),
                &mut best_index,
                &mut best_distance,
                &mut best_val,
            );

            // Record delta (ideal residual from prediction)
            if rgb_nc >= 3 {
                deltas_r.push(color[0] - predictions[0]);
                deltas_g.push(color[1] - predictions[1]);
                deltas_b.push(color[2] - predictions[2]);
                delta_distances.push(best_distance);
            }

            qrow.push(best_val);
        }
        quant_rows.push(qrow);
    }

    // Find frequent color deltas
    let bucket_size = 3i32 << (bit_depth - 8).max(0);
    let mut delta_freq: BTreeMap<[i32; 3], f64> = BTreeMap::new();
    for i in 0..deltas_r.len() {
        let key = [
            round_int(deltas_r[i], bucket_size),
            round_int(deltas_g[i], bucket_size),
            round_int(deltas_b[i], bucket_size),
        ];
        if key == [0, 0, 0] {
            continue;
        }
        *delta_freq.entry(key).or_insert(0.0) += (delta_distances[i] as f64).sqrt().sqrt();
    }

    // Weight by magnitude and normalize
    let delta_distance_multiplier = 1.0 / num_pixels as f32;
    for (key, freq) in delta_freq.iter_mut() {
        let dist = color_distance(&[0.0, 0.0, 0.0], &[key[0], key[1], key[2]]).sqrt() + 1.0;
        *freq *= dist as f64 * delta_distance_multiplier as f64;
    }

    // Sort by weighted frequency, take top deltas
    let mut sorted_deltas: Vec<([i32; 3], f64)> = delta_freq.into_iter().collect();
    sorted_deltas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    let mut frequent_deltas: Vec<[i32; 3]> = Vec::new();
    for (delta, freq) in &sorted_deltas {
        if frequent_deltas.len() >= max_deltas {
            break;
        }
        if *freq < 17.0 {
            break;
        }
        frequent_deltas.push([
            delta[0] * bucket_size,
            delta[1] * bucket_size,
            delta[2] * bucket_size,
        ]);
    }

    let nb_deltas = frequent_deltas.len();

    // ── Pass 2: Apply palette with deltas and error diffusion ────────────────

    // Sort candidate palette: common colors first (dark→bright), then rare (bright→dark)
    let freq_sort_threshold = 4usize;
    if num_c >= 3 {
        candidate_palette.sort_by(|a, b| {
            let lum_a = 0.299 * a[0] as f32 + 0.587 * a[1] as f32 + 0.114 * a[2] as f32 + 0.1;
            let lum_b = 0.299 * b[0] as f32 + 0.587 * b[1] as f32 + 0.114 * b[2] as f32 + 0.1;
            let freq_a = color_freq.get(a).copied().unwrap_or(0);
            let freq_b = color_freq.get(b).copied().unwrap_or(0);
            let key_a = if freq_a > freq_sort_threshold {
                -lum_a
            } else {
                lum_a
            };
            let key_b = if freq_b > freq_sort_threshold {
                -lum_b
            } else {
                lum_b
            };
            key_a
                .partial_cmp(&key_b)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
    }

    // Also add implicit colors that are frequent enough to the explicit palette
    let mut final_palette = candidate_palette.clone();
    for &color in &implicit_colors {
        if color_freq.get(&color).copied().unwrap_or(0) > 10 {
            final_palette.push(color);
        }
    }

    let nb_colors = final_palette.len();
    let total_size = nb_deltas + nb_colors;

    // Build final palette data
    let final_row_stride = total_size.max(1);
    let mut final_palette_data = vec![0i32; nc * final_row_stride];

    // Write deltas first
    for (i, delta) in frequent_deltas.iter().enumerate() {
        for c in 0..rgb_nc {
            final_palette_data[c * final_row_stride + i] = delta[c];
        }
    }
    // Write explicit colors after deltas
    for (i, color) in final_palette.iter().enumerate() {
        for c in 0..nc {
            final_palette_data[c * final_row_stride + nb_deltas + i] = color[c];
        }
    }

    // Rebuild inv_palette for pass 2
    let mut inv_palette2: HashMap<[i32; 4], usize> = HashMap::new();
    for (i, color) in final_palette.iter().enumerate() {
        inv_palette2.insert(*color, nb_deltas + i);
    }
    for (k, color) in implicit_colors.iter().enumerate() {
        inv_palette2.entry(*color).or_insert(total_size + k);
    }

    // Error diffusion buffers: 3 rows, each with width + 4 padding
    let mut error_rows: [Vec<[f32; 3]>; 3] = [
        vec![[0.0; 3]; width + 4],
        vec![[0.0; 3]; width + 4],
        vec![[0.0; 3]; width + 4],
    ];

    // Quantized output for prediction
    let mut quant_out: Vec<Vec<[i32; 3]>> = Vec::with_capacity(height);

    // Create palette channel (total_size wide, num_c high)
    let mut palette_channel = Channel::new(total_size, num_c).ok()?;
    for c in 0..num_c {
        for i in 0..total_size {
            palette_channel.set(i, c, final_palette_data[c * final_row_stride + i]);
        }
    }

    // Create index channel
    let mut index_channel = Channel::new(width, height).ok()?;
    let mut delta_used = false;

    for y in 0..height {
        let mut qrow: Vec<[i32; 3]> = Vec::with_capacity(width);
        let prev_row = if y > 0 {
            Some(quant_out[y - 1].as_slice())
        } else {
            None
        };
        for x in 0..width {
            let orig_color = read_color4(src_channels, x, y, nc);
            let predictions = clamped_gradient_predict(prev_row, &qrow, x, rgb_nc);

            // Try two diffusion multipliers (matching libjxl)
            let mut best_index = 0i32;
            let mut best_distance = f32::INFINITY;
            let mut best_val = [0i32; 3];
            let mut best_is_delta = false;

            for &diff_mul in &[0.55f32, 0.75] {
                // Apply error diffusion to get corrected color
                let mut color_with_error = [0.0f32; 3];
                let mut color_clamped = [0i32; 4];
                let max_val = (1i64 << bit_depth) - 1;
                for c in 0..rgb_nc {
                    color_with_error[c] = orig_color[c] as f32 + diff_mul * error_rows[0][x + 2][c];
                    color_clamped[c] =
                        (color_with_error[c].round() as i64).clamp(0, max_val) as i32;
                }

                let try_index = |index: i32,
                                 best_idx: &mut i32,
                                 best_dist: &mut f32,
                                 best_v: &mut [i32; 3],
                                 best_delta: &mut bool| {
                    let mut qval = [0i32; 3];
                    for (c, slot) in qval.iter_mut().enumerate().take(rgb_nc) {
                        *slot = get_palette_value(
                            &final_palette_data,
                            final_row_stride,
                            index,
                            c,
                            total_size as i32,
                            bit_depth,
                        );
                        if index < nb_deltas as i32 {
                            *slot += predictions[c];
                        }
                    }
                    let cd = 32.0 / (1i64 << (2 * (bit_depth - 8)).max(0)) as f32
                        * color_distance(&color_with_error[..rgb_nc], &qval[..rgb_nc]);

                    let index_penalty: f32 = if index == -1 {
                        -124.0
                    } else if index < 0 {
                        -2.0 * index as f32
                    } else if index < nb_deltas as i32 {
                        250.0
                    } else if index < total_size as i32 {
                        150.0
                    } else if index < total_size as i32 + LARGE_CUBE_OFFSET {
                        70.0
                    } else {
                        256.0
                    };
                    let dist = cd + index_penalty;
                    if dist < *best_dist {
                        *best_dist = dist;
                        *best_idx = index;
                        *best_delta = index < nb_deltas as i32;
                        *best_v = qval;
                    }
                };

                // Try all entries: implicit deltas, explicit deltas, explicit palette, implicit cubes
                for idx in MIN_IMPLICIT_PALETTE_INDEX..total_size as i32 {
                    try_index(
                        idx,
                        &mut best_index,
                        &mut best_distance,
                        &mut best_val,
                        &mut best_is_delta,
                    );
                }
                try_index(
                    quantize_to_implicit_palette(
                        &color_clamped[..rgb_nc],
                        total_size as i32,
                        bit_depth,
                        false,
                    ),
                    &mut best_index,
                    &mut best_distance,
                    &mut best_val,
                    &mut best_is_delta,
                );
                try_index(
                    quantize_to_implicit_palette(
                        &color_clamped[..rgb_nc],
                        total_size as i32,
                        bit_depth,
                        true,
                    ),
                    &mut best_index,
                    &mut best_distance,
                    &mut best_val,
                    &mut best_is_delta,
                );
            }

            delta_used |= best_is_delta;
            index_channel.set(x, y, best_index);
            qrow.push(best_val);

            // Error diffusion (matching libjxl's 12-neighbor cancellation + spread)
            let mut len_error = 0.0f32;
            for c in 0..rgb_nc {
                let local_error = orig_color[c] as f32
                    + error_rows[0][x + 2][c] * 0.65 // average of two diff_muls
                    - best_val[c] as f32;
                len_error += local_error * local_error;
            }
            len_error = len_error.sqrt();
            let len_limit = (38 << (bit_depth - 8).max(0)) as f32;
            let modulate = if len_error > len_limit {
                len_limit / len_error
            } else {
                1.0
            };

            for c in 0..rgb_nc {
                let total_error_raw =
                    orig_color[c] as f32 + error_rows[0][x + 2][c] * 0.65 - best_val[c] as f32;

                // Cancellation: if neighboring error pixels have opposite sign, cancel
                let offsets: [(usize, usize); 11] = [
                    (0, 3),
                    (0, 4),
                    (1, 0),
                    (1, 1),
                    (1, 2),
                    (1, 3),
                    (1, 4),
                    (2, 0),
                    (2, 1),
                    (2, 2),
                    (2, 3),
                ];
                let mut total_available = 0.0f32;
                for &(row, col) in &offsets {
                    let idx = x + col;
                    if idx < error_rows[row].len() {
                        let e = error_rows[row][idx][c];
                        if e.is_sign_negative() != total_error_raw.is_sign_negative() {
                            total_available += e;
                        }
                    }
                }
                let weight = (total_error_raw.abs() / (total_available.abs() + 1e-3)).min(1.0);
                let mut total_error = total_error_raw;
                for &(row, col) in &offsets {
                    let idx = x + col;
                    if idx < error_rows[row].len() {
                        let e = error_rows[row][idx][c];
                        if e.is_sign_negative() != total_error_raw.is_sign_negative() {
                            total_error += weight * e;
                            error_rows[row][idx][c] *= 1.0 - weight;
                        }
                    }
                }
                total_error *= modulate;
                let remaining = total_error / 14.0;

                // Spread error
                if x + 3 < error_rows[0].len() {
                    error_rows[0][x + 3][c] += 2.0 * remaining;
                }
                if x + 4 < error_rows[0].len() {
                    error_rows[0][x + 4][c] += remaining;
                }
                if x < error_rows[1].len() {
                    error_rows[1][x][c] += remaining;
                }
                for i in 0..5 {
                    if x + i < error_rows[1].len() {
                        error_rows[1][x + i][c] += remaining;
                    }
                    if x + i < error_rows[2].len() {
                        error_rows[2][x + i][c] += remaining;
                    }
                }
            }
        }
        quant_out.push(qrow);

        // Rotate error rows
        let tmp = core::mem::take(&mut error_rows[0]);
        error_rows[0] = core::mem::take(&mut error_rows[1]);
        error_rows[1] = core::mem::take(&mut error_rows[2]);
        error_rows[2] = tmp;
        for v in &mut error_rows[2] {
            *v = [0.0; 3];
        }
    }

    // If no deltas were actually used, set predictor to Zero
    let predictor = if delta_used { 4u8 } else { 0u8 }; // 4 = ClampedGradient (matching libjxl)

    // Replace channels with palette + index
    let mut new_channels: Vec<Channel> = Vec::new();
    // Channels before begin_c (unchanged)
    for ch in image.channels[..begin_c].iter() {
        new_channels.push(ch.clone());
    }
    // Insert palette and index channels
    new_channels.push(palette_channel);
    new_channels.push(index_channel);
    // Channels after begin_c + num_c (unchanged, e.g. alpha)
    for ch in image.channels[begin_c + num_c..].iter() {
        new_channels.push(ch.clone());
    }
    image.channels = new_channels;

    Some(LossyPaletteResult {
        nb_colors,
        nb_deltas,
        predictor,
    })
}

/// Check if palette transform is worthwhile for this image.
///
/// Returns `Some((begin_c, num_c))` if palette should be applied, `None` otherwise.
pub fn should_use_palette(image: &ModularImage) -> Option<(usize, usize)> {
    try_use_palette(image, MAX_PALETTE_COLORS).map(|(bc, nc, _)| (bc, nc))
}

/// Like `should_use_palette`, but returns the full analysis to avoid a second scan.
///
/// Returns `Some((begin_c, num_c, analysis))` if palette is beneficial.
pub fn try_use_palette(
    image: &ModularImage,
    max_colors: usize,
) -> Option<(usize, usize, PaletteAnalysis)> {
    if image.channels.len() < 2 {
        // Grayscale — palette still helps for few-color images
        let analysis = analyze_palette(image, 0, 1, 256);
        if analysis.use_palette && analysis.num_colors <= 256 {
            return Some((0, 1, analysis));
        }
        return None;
    }

    // RGB or RGBA: palettize the color channels (not alpha)
    let num_color_channels = if image.has_alpha {
        image.channels.len() - 1
    } else {
        image.channels.len()
    };

    if num_color_channels < 2 {
        return None;
    }

    let analysis = analyze_palette(image, 0, num_color_channels, max_colors);

    if analysis.use_palette {
        Some((0, num_color_channels, analysis))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::channel::ModularImage;

    #[test]
    fn test_palette_analysis_few_colors() {
        // 4x4 image with only 3 colors
        let data: Vec<u8> = vec![
            255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0,
            0, 255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 255, 0,
        ];
        let image = ModularImage::from_rgb8(&data, 4, 4).unwrap();
        let analysis = analyze_palette(&image, 0, 3, 256);
        assert!(analysis.use_palette);
        assert_eq!(analysis.num_colors, 3);
    }

    #[test]
    fn test_palette_analysis_too_many_colors() {
        // Each pixel is unique
        let mut data = Vec::new();
        for i in 0..64u8 {
            data.push(i);
            data.push(i);
            data.push(i);
        }
        let image = ModularImage::from_rgb8(&data, 8, 8).unwrap();
        let analysis = analyze_palette(&image, 0, 3, 16);
        assert!(!analysis.use_palette); // 64 colors > max_colors=16
    }

    #[test]
    fn test_palette_apply() {
        // 2x2 image with 2 colors: red and blue
        let data: Vec<u8> = vec![255, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 255];
        let mut image = ModularImage::from_rgb8(&data, 2, 2).unwrap();
        assert_eq!(image.channels.len(), 3);

        let analysis = analyze_palette(&image, 0, 3, 256);
        assert!(analysis.use_palette);
        assert_eq!(analysis.num_colors, 2);

        let nb_colors = apply_palette(&mut image, 0, 3, &analysis).unwrap();
        assert_eq!(nb_colors, 2);

        // After palette: palette_channel + index_channel = 2 channels
        assert_eq!(image.channels.len(), 2);

        // Palette channel: width=2, height=3 (num_c=3 for RGB)
        assert_eq!(image.channels[0].width(), 2);
        assert_eq!(image.channels[0].height(), 3);

        // Index channel: width=2, height=2
        assert_eq!(image.channels[1].width(), 2);
        assert_eq!(image.channels[1].height(), 2);

        // All pixels should map to valid indices
        for y in 0..2 {
            for x in 0..2 {
                let idx = image.channels[1].get(x, y);
                assert!(idx >= 0 && idx < nb_colors as i32);
            }
        }
    }

    #[test]
    fn test_palette_roundtrip_values() {
        // Verify palette stores correct values
        let data: Vec<u8> = vec![10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60];
        let mut image = ModularImage::from_rgb8(&data, 2, 2).unwrap();

        let analysis = analyze_palette(&image, 0, 3, 256);
        assert_eq!(analysis.num_colors, 2);

        let _nb = apply_palette(&mut image, 0, 3, &analysis).unwrap();

        // Check that we can reconstruct original values from palette + indices
        let palette = &image.channels[0];
        let indices = &image.channels[1];

        for y in 0..2 {
            for x in 0..2 {
                let idx = indices.get(x, y) as usize;
                let r = palette.get(idx, 0);
                let g = palette.get(idx, 1);
                let b = palette.get(idx, 2);
                let orig_r = data[(y * 2 + x) * 3] as i32;
                let orig_g = data[(y * 2 + x) * 3 + 1] as i32;
                let orig_b = data[(y * 2 + x) * 3 + 2] as i32;
                assert_eq!((r, g, b), (orig_r, orig_g, orig_b));
            }
        }
    }
}
