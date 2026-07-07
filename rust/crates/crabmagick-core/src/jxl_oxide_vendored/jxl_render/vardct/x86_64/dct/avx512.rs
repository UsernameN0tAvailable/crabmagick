#![allow(unsafe_op_in_unsafe_fn)]
//! AVX-512 entry point for the 2D DCT.
//!
//! A 16-wide `__m512` lane DCT would require the coefficient grid to be 64-byte
//! aligned with a stride that is a multiple of 16 (see `SharedSubgrid::as_vectored`).
//! The vendored grid allocator aligns buffers to 32 bytes (`AlignedGrid::ALIGN`), so
//! a `__m512` view of the DCT coefficient sub-grids essentially never succeeds and
//! would silently fall back to the scalar path — losing the SSE2/AVX2 acceleration.
//!
//! To guarantee correct and *fast* behavior on AVX-512 hardware we therefore route
//! this path through the tested AVX2 (8-wide) implementation, which only needs the
//! already-guaranteed 32-byte alignment and still doubles SSE2 throughput. AVX-512
//! capable CPUs run the AVX2 code without penalty (VEX-encoded, no license-based
//! frequency throttling), so this is the pragmatic, verifiably-correct choice.

use crate::jxl_oxide_vendored::jxl_grid::MutableSubgrid;
use crate::jxl_oxide_vendored::jxl_render::vardct::dct_common::DctDirection;

#[target_feature(enable = "avx512f,avx2,fma")]
pub(crate) unsafe fn dct_2d_avx512(io: &mut MutableSubgrid<'_>, direction: DctDirection) {
    super::avx2::dct_2d_avx2(io, direction)
}
