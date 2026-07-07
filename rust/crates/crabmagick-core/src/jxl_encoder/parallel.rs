// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! Parallel execution abstraction.
//!
//! When the `parallel` feature is enabled, uses rayon for parallel iteration.
//! Otherwise falls back to sequential iteration. This module provides a single
//! abstraction (`parallel_map`) so callers don't need `#[cfg]` blocks.

use crate::jxl_encoder::error::Result;

#[cfg(feature = "parallel")]
thread_local! {
    static FORCE_SEQUENTIAL: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(feature = "parallel")]
struct SequentialGuard;

#[cfg(feature = "parallel")]
impl Drop for SequentialGuard {
    fn drop(&mut self) {
        FORCE_SEQUENTIAL.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

#[cfg(feature = "parallel")]
pub(crate) fn with_sequential_maps<T>(f: impl FnOnce() -> T) -> T {
    FORCE_SEQUENTIAL.with(|depth| depth.set(depth.get() + 1));
    let _guard = SequentialGuard;
    f()
}

#[cfg(feature = "parallel")]
#[inline]
fn force_sequential() -> bool {
    FORCE_SEQUENTIAL.with(|depth| depth.get() != 0)
}

#[cfg(feature = "parallel")]
#[inline]
pub(crate) fn sequential_maps_forced() -> bool {
    force_sequential()
}

#[cfg(not(feature = "parallel"))]
#[inline]
pub(crate) fn sequential_maps_forced() -> bool {
    true
}

/// Map `f` over `0..n`, collecting results in index order.
///
/// Uses `rayon::par_iter` when the `parallel` feature is enabled,
/// otherwise uses sequential `(0..n).map(f).collect()`.
#[cfg(feature = "parallel")]
pub fn parallel_map<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    if force_sequential() {
        return (0..n).map(f).collect();
    }

    use rayon::prelude::*;
    (0..n).into_par_iter().map(f).collect()
}

/// Map `f` over `0..n`, collecting results in index order (sequential fallback).
#[cfg(not(feature = "parallel"))]
pub fn parallel_map<T, F>(n: usize, f: F) -> Vec<T>
where
    F: Fn(usize) -> T,
{
    (0..n).map(f).collect()
}

/// Map `f` over `0..n` where `f` returns `Result<T>`, collecting results in index order.
///
/// Returns the first error encountered, or all results.
#[cfg(feature = "parallel")]
pub fn parallel_map_result<T, F>(n: usize, f: F) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(usize) -> Result<T> + Send + Sync,
{
    if force_sequential() {
        return (0..n).map(f).collect();
    }

    use rayon::prelude::*;
    (0..n).into_par_iter().map(f).collect()
}

/// Map `f` over `0..n` where `f` returns `Result<T>` (sequential fallback).
#[cfg(not(feature = "parallel"))]
pub fn parallel_map_result<T, F>(n: usize, f: F) -> Result<Vec<T>>
where
    F: Fn(usize) -> Result<T>,
{
    (0..n).map(f).collect()
}

/// Parallel fold-reduce over `0..n`.
///
/// Each Rayon worker independently builds a local accumulator via `init()` + `fold(acc, i)`,
/// then the per-thread results are merged pairwise using `reduce(a, b)`.
///
/// Falls back to sequential when the `parallel` feature is disabled, when
/// `with_sequential_maps` is active, or when `n < min_parallel` (avoids Rayon overhead
/// for trivially small inputs).
///
/// Typical use: building per-context histograms in parallel over a slice of token groups,
/// then merging the per-thread histograms before building the ANS distribution.
#[cfg(feature = "parallel")]
pub fn parallel_accumulate<T, Init, Fold, Reduce>(
    n: usize,
    min_parallel: usize,
    init: Init,
    fold: Fold,
    reduce: Reduce,
) -> T
where
    T: Send,
    Init: Fn() -> T + Send + Sync,
    Fold: Fn(T, usize) -> T + Send + Sync,
    Reduce: Fn(T, T) -> T + Send + Sync,
{
    if force_sequential() || n < min_parallel {
        return (0..n).fold(init(), |acc, i| fold(acc, i));
    }

    use rayon::prelude::*;
    (0..n)
        .into_par_iter()
        .fold(|| init(), |acc, i| fold(acc, i))
        .reduce(|| init(), |a, b| reduce(a, b))
}

/// Parallel fold-reduce — sequential fallback.
#[cfg(not(feature = "parallel"))]
pub fn parallel_accumulate<T, Init, Fold, Reduce>(
    n: usize,
    _min_parallel: usize,
    init: Init,
    fold: Fold,
    _reduce: Reduce,
) -> T
where
    Init: Fn() -> T,
    Fold: Fn(T, usize) -> T,
{
    (0..n).fold(init(), |acc, i| fold(acc, i))
}

/// Apply `f` to each index in `0..n` in parallel (no return value).
///
/// Unlike `parallel_map`, no output is collected. Suitable for in-place parallel
/// writes where each invocation touches disjoint memory (caller must ensure
/// safety when captures contain raw pointers).
#[cfg(feature = "parallel")]
pub fn parallel_for_each<F>(n: usize, f: F)
where
    F: Fn(usize) + Send + Sync,
{
    if force_sequential() {
        (0..n).for_each(f);
        return;
    }
    use rayon::prelude::*;
    (0..n).into_par_iter().for_each(f);
}

/// Sequential fallback for `parallel_for_each`.
#[cfg(not(feature = "parallel"))]
pub fn parallel_for_each<F>(n: usize, f: F)
where
    F: Fn(usize),
{
    (0..n).for_each(f);
}

/// Parallel mutable slice processing: split `data` into chunks of `chunk_size` bytes,
/// then call `f(chunk_idx, chunk_slice)` on each chunk in parallel.
///
/// Each invocation receives a disjoint mutable slice of data, making the
/// parallel writes safe without unsafe code.
#[cfg(feature = "parallel")]
pub fn parallel_chunks_mut<T, F>(data: &mut [T], chunk_size: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    if sequential_maps_forced() {
        for (i, chunk) in data.chunks_mut(chunk_size).enumerate() {
            f(i, chunk);
        }
        return;
    }
    use rayon::prelude::*;
    data.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, chunk)| f(i, chunk));
}

/// Sequential fallback for `parallel_chunks_mut`.
#[cfg(not(feature = "parallel"))]
pub fn parallel_chunks_mut<T, F>(data: &mut [T], chunk_size: usize, f: F)
where
    F: Fn(usize, &mut [T]),
{
    for (i, chunk) in data.chunks_mut(chunk_size).enumerate() {
        f(i, chunk);
    }
}
