#!/usr/bin/env bash
# Build every daemon binary that is bundled with the Composer package locally.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly RUST_DIR="$REPO_ROOT/rust"
readonly BIN_DIR="$REPO_ROOT/bin"
readonly PACKAGE="crabmagick-daemon"
readonly DAEMON="crabmagick-daemon"

log() {
    printf '%s\n' "$*"
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

ensure_target() {
    local target="$1"

    if ! rustup target list --installed | grep -Fxq "$target"; then
        log "Installing Rust target $target"
        rustup target add "$target"
    fi
}

verify_binary() {
    local path="$1"
    local expected_arch="$2"
    local description

    description="$(file -Lb "$path")"
    if [[ "$description" != *"$expected_arch"* ]]; then
        die "$path has unexpected architecture: $description"
    fi

    log "  verified: $description"
}

build_variant() {
    local target="$1"
    local variant="$2"
    local rustflags="$3"
    local expected_arch="$4"
    local lto="$5"
    local output="$BIN_DIR/crabmagick-$variant-linux"
    local source="$RUST_DIR/target/$target/release/$DAEMON"

    log "Building crabmagick-$variant-linux ($target)"
    if [[ "$target" == "x86_64-unknown-linux-musl" ]]; then
        (
            cd "$RUST_DIR"
            RUSTFLAGS="$rustflags" \
                CARGO_PROFILE_RELEASE_LTO="$lto" \
                CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER=musl-gcc \
                cargo build --locked -p "$PACKAGE" --release --target "$target"
        )
    else
        (
            cd "$RUST_DIR"
            RUSTFLAGS="$rustflags" \
                CARGO_PROFILE_RELEASE_LTO="$lto" \
                CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
                cargo build --locked -p "$PACKAGE" --release --target "$target"
        )
    fi

    install -Dm755 "$source" "$output"
    verify_binary "$output" "$expected_arch"
}

if [[ "$(uname -s)" != "Linux" ]]; then
    die "the bundled artifacts target Linux and must be built from a Linux host"
fi

if (( $# != 0 )); then
    die "this command always builds the complete bundled-binary matrix; it accepts no arguments"
fi

require_command cargo
require_command rustup
require_command musl-gcc
require_command aarch64-linux-gnu-gcc
require_command file

ensure_target x86_64-unknown-linux-musl
ensure_target aarch64-unknown-linux-gnu

mkdir -p "$BIN_DIR"

# Keep this matrix aligned with Runtime's CPU-feature dispatch.
build_variant x86_64-unknown-linux-musl x86_64 \
    "-C target-feature=+crt-static" \
    "x86-64" \
    fat
build_variant x86_64-unknown-linux-musl x86_64-avx2 \
    "-C target-feature=+crt-static,+avx2,+fma" \
    "x86-64" \
    fat
build_variant x86_64-unknown-linux-musl x86_64-avx512 \
    "-C target-feature=+crt-static,+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl" \
    "x86-64" \
    fat
build_variant aarch64-unknown-linux-gnu aarch64 \
    "-C target-feature=+neon" \
    "ARM aarch64" \
    fat
# There are no SVE-specific kernels in the daemon today. Global SVE2 codegen leaves one
# LLVM object permanently incomplete when cross-compiling, so SVE hosts use the proven
# NEON baseline until targeted SVE kernels are added and benchmarked.
build_variant aarch64-unknown-linux-gnu aarch64-sve \
    "-C target-feature=+neon" \
    "ARM aarch64" \
    fat

log ""
log "Built bundled daemon binaries locally:"
sha256sum "$BIN_DIR"/crabmagick-*-linux
