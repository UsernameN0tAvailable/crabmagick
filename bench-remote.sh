#!/usr/bin/env bash
set -euo pipefail

# Run benchmarks on the remote machine where libvips is available.
REMOTE_HOST="${CRABMAGICK_BENCH_HOST:-mattia@192.168.1.118}"
REMOTE_DIR="${CRABMAGICK_REMOTE_DIR:-~/Work/crabmagick/rust}"
REMOTE_RUSTFLAGS="${CRABMAGICK_BENCH_RUSTFLAGS:--C target-cpu=native}"

ssh "${REMOTE_HOST}" "cd ${REMOTE_DIR} && RUSTFLAGS='${REMOTE_RUSTFLAGS}' cargo bench -p crabmagick-core --bench vs_libvips --profile bench 2>&1"
