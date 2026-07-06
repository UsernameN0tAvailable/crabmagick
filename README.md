# crabmagick

Pure-Rust image processing for PHP. A daemon binary is bundled inside the
Composer package and spawned automatically on first use.

**Zero configuration. No php.ini changes. No sudo. Just:**

```bash
composer require usernamn0tavailable/crabmagick
```

> **Looking for the PHP extension variant?** See
> [crabmagick-ext](https://github.com/UsernameN0tAvailable/crabmagick-ext) —
> same API, loaded as a native PHP extension via `PHP_INI_SCAN_DIR`.

---

## How it works

```
require vendor/autoload.php
        │
        ▼  bootstrap.php (autoload.files)
        │  • finds the right daemon binary for your CPU arch
        │  • checks /tmp/crabmagick-<uid>.sock
        │  • spawns daemon in background if not running
        │  • registers socket path with Runtime
        ▼
\Crabmagick\Image  →  Unix socket  →  crabmagick-daemon
                                       (pure Rust, statically linked)
                                       decode / crop / resize / encode
```

One daemon process is shared across all PHP-FPM workers for the same user.
The socket is at `/tmp/crabmagick-<uid>.sock` and is created automatically.

---

## Requirements

- PHP ≥ 8.1, Linux x86_64 or aarch64
- No libvips, no libjxl, no C compiler, no Rust toolchain

---

## PHP API

```php
// Fluent builder
$bytes = (new \Crabmagick\Image('/path/to/file.jxl'))
    ->region(0, 0, 512, 512)   // crop region (x, y, w, h)
    ->resize(256, 256)          // output dimensions (0 = proportional)
    ->rotate(90)                // clockwise: 90 | 180 | 270
    ->encode('jpeg', 85);       // 'jpeg' | 'webp' | 'png' | 'jxl' | 'avif'

// Square crop (IIIF "square" region — largest centred square)
$bytes = (new \Crabmagick\Image('/path/to/file.jxl'))
    ->square()
    ->resize(512)
    ->encode('webp', 80);

// Select page/frame (multi-page TIFF, animated WebP, PDF)
$bytes = (new \Crabmagick\Image('/path/to/document.tiff'))
    ->page(2)
    ->resize(1024)
    ->encode('jpeg', 90);

// One-shot helper
$bytes = \Crabmagick\process($path, $rx, $ry, $rw, $rh, $outW, $outH, 'jpeg', 85);

// Dimensions only (reads file header, no full decode for JXL)
$info = \Crabmagick\info($path);          // ['width' => 3360, 'height' => 4892]
$info = (new \Crabmagick\Image($path))->getInfo();

// Check daemon is up
\Crabmagick\Image::isAvailable();         // bool
```

---

## Supported formats

| Format | Decode | Encode |
|--------|--------|--------|
| JPEG   | ✅ | ✅ |
| PNG    | ✅ | ✅ |
| WebP   | ✅ | ✅ |
| JXL    | ✅ | ✅ |
| AVIF   | ✅ | ✅ |
| TIFF   | ✅ | — |
| GIF    | ✅ | — |
| BMP    | ✅ | — |
| SVG    | ✅ | — |
| JP2/J2K | ✅ | — |
| PDF    | ✅ (optional build) | — |

All decoders and encoders are pure Rust — zero C dependencies.

---

## Performance

Benchmarked on a 3360×4892 JXL source image, 512×512 output tile:

| Operation | Time |
|-----------|------|
| Cold tile (crop-during-decode) | ~29 ms |
| Full image decode, 800px output | ~570 ms |

The daemon adds ~0.3 ms of Unix socket overhead per request.
JXL crop-during-decode (`jxl-oxide` group-level region decode) avoids
decoding the full image for tile requests — the main source of the speedup
vs. libvips (~540 ms for the same cold tile).

---

## Bundled binaries

| Binary | Target | Use |
|--------|--------|-----|
| `crabmagick-x86_64-linux` | x86_64 baseline (SSE2) | Max compatibility |
| `crabmagick-x86_64-avx2-linux` | x86_64 with AVX2 (Haswell 2013+, Ryzen 2017+) | Recommended for most servers |
| `crabmagick-x86_64-avx512-linux` | x86_64 with AVX-512 (Skylake-X+, Zen 4+) | Highest throughput |
| `crabmagick-aarch64-linux` | aarch64 generic (AWS Graviton, RPi 4+) | |
| `crabmagick-aarch64-sve-linux` | aarch64 with SVE (Graviton 3+, Neoverse N2) | |

The right binary is selected automatically at runtime by reading `/proc/cpuinfo`.
Override via `CRABMAGICK_BINARY` env var if needed.

All binaries are statically linked (musl libc) — no dynamic dependencies beyond
the Linux kernel ABI (`linux-vdso`, `ld-musl`).

---

## Daemon lifecycle

The daemon runs as a background process owned by the user that first spawned it.
Multiple PHP-FPM workers (same user) all connect to the same daemon.

- **Socket path:** `/tmp/crabmagick-<uid>.sock`
- **Auto-spawn:** happens in `bootstrap.php` on every `require vendor/autoload.php`
  if the socket is not yet present
- **Restart:** if the daemon crashes, the next PHP request that finds the socket
  missing will re-spawn it automatically
- **Shutdown:** the daemon exits when the socket is deleted or on SIGTERM

To stop the daemon manually:
```bash
kill $(cat /tmp/crabmagick-<uid>.pid 2>/dev/null) 2>/dev/null || true
rm -f /tmp/crabmagick-<uid>.sock
```

---

## Build from source

Requires Rust stable and musl toolchain for static binaries:

```bash
rustup target add x86_64-unknown-linux-musl
sudo apt-get install musl-tools    # Ubuntu/Debian

cd rust

# x86_64 variants
RUSTFLAGS="-C target-cpu=x86-64          -C link-arg=-static-libgcc" \
  cargo build -p crabmagick-daemon --release --target x86_64-unknown-linux-musl
cp target/x86_64-unknown-linux-musl/release/crabmagick-daemon ../bin/crabmagick-x86_64-linux

RUSTFLAGS="-C target-cpu=haswell         -C link-arg=-static-libgcc" \
  cargo build -p crabmagick-daemon --release --target x86_64-unknown-linux-musl
cp target/x86_64-unknown-linux-musl/release/crabmagick-daemon ../bin/crabmagick-x86_64-avx2-linux

RUSTFLAGS="-C target-cpu=skylake-avx512  -C link-arg=-static-libgcc" \
  cargo build -p crabmagick-daemon --release --target x86_64-unknown-linux-musl
cp target/x86_64-unknown-linux-musl/release/crabmagick-daemon ../bin/crabmagick-x86_64-avx512-linux

# aarch64 (cross-compile)
rustup target add aarch64-unknown-linux-musl
sudo apt-get install gcc-aarch64-linux-gnu

RUSTFLAGS="-C target-cpu=generic         -C link-arg=-static-libgcc" \
  cargo build -p crabmagick-daemon --release --target aarch64-unknown-linux-musl
cp target/aarch64-unknown-linux-musl/release/crabmagick-daemon ../bin/crabmagick-aarch64-linux

RUSTFLAGS="-C target-cpu=generic -C target-feature=+sve -C link-arg=-static-libgcc" \
  cargo build -p crabmagick-daemon --release --target aarch64-unknown-linux-musl
cp target/aarch64-unknown-linux-musl/release/crabmagick-daemon ../bin/crabmagick-aarch64-sve-linux
```

---

## Bundled Rust crates

All pure Rust, zero C dependencies:

| Crate | Role |
|-------|------|
| `crabmagick-core` | Decode / crop / resize / encode pipeline |
| `crabmagick-daemon` | Unix socket server binary |
| `zenjpeg` | JPEG encode + decode |
| `fast-webp` | WebP encode + decode |
| `jxl-encoder` | JXL encode |
| `jxl-oxide` | JXL decode (crop-during-decode) |
| `zen-jp2` | JP2/J2K decode |
| `fast_image_resize` | SIMD resize |
| `resvg` | SVG rasterise |
