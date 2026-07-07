use crate::jxl_decode::jxl_grid::{AlignedGrid, MutableSubgrid};
use crate::jxl_decode::jxl_threadpool::JxlThreadPool;

use crate::jxl_decode::jxl_render::{ImageWithRegion, Region};

use super::impls::generic::gabor::gabor_row_edge;

pub fn apply_gabor_like(
    fb: &mut ImageWithRegion,
    color_padded_region: Region,
    fb_scratch: &mut [AlignedGrid<f32>; 3],
    weights: [[f32; 2]; 3],
    pool: &crate::jxl_decode::jxl_threadpool::JxlThreadPool,
) {
    tracing::debug!("Running gaborish");
    let region = fb.regions_and_shifts()[0].0;
    assert!(region.contains(color_padded_region));
    let left = region.left.abs_diff(color_padded_region.left) as usize;
    let top = region.top.abs_diff(color_padded_region.top) as usize;
    let right = left + color_padded_region.width as usize;
    let bottom = top + color_padded_region.height as usize;

    let buffers = fb.as_color_floats_mut();
    let buffers = buffers.map(|g| g.as_subgrid_mut().subgrid(left..right, top..bottom));

    super::impls::apply_gabor_like(buffers, fb_scratch, weights, pool);

    let left = color_padded_region.left;
    let top = color_padded_region.top;
    for (idx, grid) in fb_scratch.iter_mut().enumerate() {
        let width = grid.width() as u32;
        let height = grid.height() as u32;
        let region = Region {
            width,
            height,
            left,
            top,
        };
        fb.swap_channel_f32(idx, grid, region);
    }
}

pub(super) struct GaborRow<'buf> {
    pub input_rows: [&'buf [f32]; 3],
    pub output_row: &'buf mut [f32],
    pub weights: [f32; 2],
}

pub(super) fn run_gabor_rows<'buf>(
    input: MutableSubgrid<'buf, f32>,
    output: &'buf mut AlignedGrid<f32>,
    weights: [f32; 2],
    pool: &JxlThreadPool,
    handle_row: for<'a> fn(GaborRow<'a>),
) {
    unsafe { run_gabor_rows_unsafe(input, output, weights, pool, handle_row) }
}

pub(super) unsafe fn run_gabor_rows_unsafe<'buf>(
    input: MutableSubgrid<'buf, f32>,
    output: &'buf mut AlignedGrid<f32>,
    weights: [f32; 2],
    pool: &JxlThreadPool,
    handle_row: for<'a> unsafe fn(GaborRow<'a>),
) {
    let width = input.width();
    let height = input.height();
    let output_buf = output.buf_mut();
    assert_eq!(output_buf.len(), width * height);

    if height == 1 {
        let input_buf = input.get_row(0);
        gabor_row_edge(input_buf, None, output_buf, weights);
        return;
    }

    {
        let input_buf_c = input.get_row(0);
        let input_buf_a = input.get_row(1);
        let output_buf = &mut output_buf[..width];
        gabor_row_edge(input_buf_c, Some(input_buf_a), output_buf, weights);
    }

    let (inner_rows, bottom_row) = output_buf[width..].split_at_mut((height - 2) * width);
    let output_rows = inner_rows
        .chunks_mut(width * 8)
        .enumerate()
        .collect::<Vec<_>>();

    pool.for_each_vec(output_rows, |(y8, output_rows)| {
        let it = output_rows.chunks_exact_mut(width);
        for (dy, output_row) in it.enumerate() {
            let y_up = y8 * 8 + dy;
            let input_rows = [
                input.get_row(y_up),
                input.get_row(y_up + 1),
                input.get_row(y_up + 2),
            ];
            let row = GaborRow {
                input_rows,
                output_row,
                weights,
            };
            unsafe {
                handle_row(row);
            }
        }
    });

    {
        let input_buf_c = input.get_row(height - 1);
        let input_buf_a = input.get_row(height - 2);
        let output_buf = bottom_row;
        gabor_row_edge(input_buf_c, Some(input_buf_a), output_buf, weights);
    }
}

/// Like `run_gabor_rows_unsafe`, but processes all 3 channels in a single parallel pass,
/// eliminating the 3 sequential pool barriers of calling it three times.
pub(super) unsafe fn run_gabor_3ch_rows_unsafe(
    inputs: [MutableSubgrid<'_, f32>; 3],
    outputs: &mut [AlignedGrid<f32>; 3],
    weights: [[f32; 2]; 3],
    pool: &JxlThreadPool,
    handle_row: for<'a> unsafe fn(GaborRow<'a>),
) {
    let width = inputs[0].width();
    let height = inputs[0].height();

    if height == 1 {
        for c in 0..3 {
            gabor_row_edge(inputs[c].get_row(0), None, outputs[c].buf_mut(), weights[c]);
        }
        return;
    }

    // Top + bottom edge rows for all 3 channels (scalar).
    for c in 0..3 {
        let out = outputs[c].buf_mut();
        gabor_row_edge(inputs[c].get_row(0), Some(inputs[c].get_row(1)), &mut out[..width], weights[c]);
        gabor_row_edge(inputs[c].get_row(height - 1), Some(inputs[c].get_row(height - 2)), &mut out[(height - 1) * width..], weights[c]);
    }

    if height <= 2 {
        return;
    }

    let num_inner = height - 2;
    let num_blocks = num_inner.div_ceil(8);

    // Raw pointers to the start of inner rows (after top edge) for each output channel.
    // Safety: these remain valid for the lifetime of `outputs`; each job accesses
    // non-overlapping rows (unique y8) and non-overlapping channels (distinct allocations).
    struct Job {
        y8: usize,
        rows_in_block: usize,
        out_ptrs: [*mut f32; 3],
    }
    unsafe impl Send for Job {}

    let out_ptrs = [
        outputs[0].buf_mut().as_mut_ptr().add(width),
        outputs[1].buf_mut().as_mut_ptr().add(width),
        outputs[2].buf_mut().as_mut_ptr().add(width),
    ];

    let jobs: Vec<Job> = (0..num_blocks)
        .map(|y8| Job {
            y8,
            rows_in_block: (num_inner - y8 * 8).min(8),
            out_ptrs,
        })
        .collect();

    pool.for_each_vec(jobs, |job| {
        for dy in 0..job.rows_in_block {
            let row_offset = job.y8 * 8 + dy;
            let y_up = row_offset; // inner row index 0 maps to image row y=1
            for c in 0..3 {
                let output_row = unsafe {
                    std::slice::from_raw_parts_mut(job.out_ptrs[c].add(row_offset * width), width)
                };
                let input_rows = [
                    inputs[c].get_row(y_up),
                    inputs[c].get_row(y_up + 1),
                    inputs[c].get_row(y_up + 2),
                ];
                unsafe {
                    handle_row(GaborRow { input_rows, output_row, weights: weights[c] });
                }
            }
        }
    });
}

#[allow(unused)]
pub(crate) fn run_gabor_row_generic(row: GaborRow) {
    super::impls::generic::gabor::run_gabor_row_generic(row)
}
