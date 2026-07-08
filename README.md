# CrabMagick

Pure-Rust image processing for PHP. Zero C dependencies, no libvips, no libjxl.

Two install modes — same `\CrabMagick\Image` API either way:

| Mode | Install | Performance |
|------|---------|-------------|
| **Daemon** (zero config) | `composer require usernamn0tavailable/crabmagick` | ~0.3 ms socket overhead |
| **Extension** (optional) | composer require + `PHP_INI_SCAN_DIR` | zero overhead, in-process |

The extension is auto-detected at runtime. If loaded, it is used directly. If not, the bundled daemon binary is spawned automatically.

---

## Installation

### Zero-config (daemon)

```bash
composer require usernamn0tavailable/crabmagick
```

That's it. The daemon starts automatically on the first `require vendor/autoload.php`.

### With extension (optional, maximum performance)

Install the Composer package first, then activate the bundled `.so`:

```bash
composer require usernamn0tavailable/crabmagick
```

```bash
# Activate for CLI / php-fpm / built-in server — no system php.ini change needed
export PHP_INI_SCAN_DIR=:/path/to/vendor/usernamn0tavailable/crabmagick/ext

# Or for php-fpm pools, add to the pool config:
# env[PHP_INI_SCAN_DIR] = :/path/to/vendor/usernamn0tavailable/crabmagick/ext
```

The `ext/` directory contains `.ini` files that load the correct `.so` for your
PHP version and architecture automatically.

To verify the extension is active:

```bash
php -r "var_dump(\CrabMagick\Image::isAvailable());"
# bool(true) — extension path
# bool(true) — daemon path (both return true)

php -r "var_dump(\CrabMagick\Runtime::isUsingExtension());"
# bool(true) if extension is loaded
```

---

## Requirements

- PHP ≥ 8.1, Linux x86_64 or aarch64
- No libvips, no libjxl, no C compiler, no Rust toolchain

---

## Daemon — conditions & compatibility

The daemon is the default mode, auto-selected when the PHP extension is not loaded.

### Hard requirements (daemon will refuse to start if any fail)

| Condition | Why |
|-----------|-----|
| **Linux only** | Daemon uses Unix sockets (`AF_UNIX`). Windows and macOS are not supported. |
| **x86_64 or aarch64** | Pre-built binaries are provided for these two architectures only. |
| **`proc_open` enabled** | PHP spawns the daemon process via `proc_open`. If it appears in `disable_functions`, the daemon cannot start. |
| **`/tmp` writable** (or custom `sys_get_temp_dir()`) | The Unix socket file and the daemon PID lock are created in `sys_get_temp_dir()`. The web server user must be able to write there. |
| **glibc ≥ 2.17** (aarch64) or **musl** (x86_64) | x86_64 binaries are statically linked with musl — they run everywhere. aarch64 binaries link against glibc dynamically; any Ubuntu 18.04+ / Debian 10+ / RHEL 8+ satisfies this. |

### Soft requirements (degrade gracefully if absent)

| Condition | Effect if missing |
|-----------|-------------------|
| **AVX2 / AVX512 CPU flags** | Falls back to the baseline `x86_64` binary automatically — still correct, just slower. |
| **`/proc/cpuinfo` readable** | CPU feature detection fails silently; baseline binary is used. |
| **Persistent process** (php-fpm / long-lived) | Each PHP-CLI invocation spawns and teardown the daemon — adds ~20ms cold-start. With php-fpm the daemon is shared across workers and starts once. |

### Verified environments

| Environment | Status |
|-------------|--------|
| Ubuntu 20.04 / 22.04 / 24.04, PHP 8.1–8.4 | ✅ Tested |
| Debian 11 / 12 | ✅ Tested |
| Alpine Linux (php-fpm) | ✅ Works (x86_64 musl binary is fully static) |
| AWS Graviton 2/3 (aarch64) | ✅ Tested |
| Docker (`FROM php:8.x-fpm-alpine`) | ✅ Tested |
| Azure Arc VM (Xeon Gold, Ubuntu 24.04) | ✅ Verified compatible |
| macOS | ❌ Unix socket path differs, not supported |
| Windows | ❌ Not supported |

### php-fpm configuration tips

The daemon is a persistent background process. Under php-fpm it is started once per pool worker on first request and stays alive for the pool lifetime.

```ini
; Allow proc_open (it is enabled by default — only needed if you disabled it)
; Remove 'proc_open' from disable_functions in php.ini / pool config

; Recommended: give the pool a dedicated tmp dir so each pool has its own socket
env[TMPDIR] = /run/php/crabmagick-pool-www
```

```bash
# Create a per-pool socket dir (add to your provisioning / Dockerfile)
install -d -o www-data -g www-data -m 0700 /run/php/crabmagick-pool-www
```

### Checking daemon status from PHP

```php
// Is the daemon running right now?
var_dump(\CrabMagick\Runtime::isDaemonRunning());   // bool(true)

// Which binary was selected?
var_dump(\CrabMagick\Runtime::binaryPath());        // string("/path/to/crabmagick-x86_64-avx512-linux")

// Is the extension being used instead?
var_dump(\CrabMagick\Runtime::isUsingExtension());  // bool(false) in daemon mode
```

---

## PHP API

```php
// Fluent builder
$bytes = (new \CrabMagick\Image('/path/to/file.jxl'))
    ->region(0, 0, 512, 512)   // crop (x, y, w, h)
    ->resize(256, 256)          // output size (0 = proportional)
    ->rotate(90)                // clockwise: 90 | 180 | 270
    ->encode('jpeg', 85);       // 'jpeg' | 'webp' | 'png' | 'jxl' | 'avif'

// Square crop (IIIF "square" region)
$bytes = (new \CrabMagick\Image($path))->square()->resize(512)->encode('webp', 80);

// Page/frame selection (multi-page TIFF, animated WebP)
$bytes = (new \CrabMagick\Image($path))->page(2)->resize(1024)->encode('jpeg');

// One-shot helper
$bytes = \CrabMagick\process($path, $rx, $ry, $rw, $rh, $outW, $outH, 'jpeg', 85);

// Dimensions without full decode
$info = \CrabMagick\info($path);          // ['width' => 3360, 'height' => 4892]
$info = (new \CrabMagick\Image($path))->getInfo();
```

---

## Supported formats

| Format | Decode | Encode |
|--------|--------|--------|
| JPEG   | ✅ | ✅ |
| PNG    | ✅ | ✅ |
| WebP (lossy + lossless) | ✅ | ✅ |
| JPEG XL | ✅ | ✅ |
| AVIF   | ✅ | ✅ |
| TIFF   | ✅ | ✅ (LZW, Deflate, Packbits; horizontal predictor) |
| GIF    | ✅ | — |
| BMP    | ✅ | — |
| SVG    | ✅ | — |
| JP2/J2K | ✅ | — |

---

## Performance

Benchmarks run on Intel Core Ultra 7 155H (AVX2, 22 threads) against libvips (industry-standard C
image processing). Three image types (photo/document/gradient), three sizes (256×256 tile, 800×600
medium, 1920×1080 HD). Each cell shows median of 5 runs.

### Photo 800×600 — encoder comparison

| Codec | crab ms | crab KB | PSNR | vips ms | vips KB |
|-------|---------|---------|------|---------|---------|
| JPEG Q75 | 4.8 | 209 | 18.3 | 4.4 | 204 |
| JPEG Q75 opt-huffman | **3.0** | 193 | 18.3 | 6.6 | 197 |
| JPEG Q85 | 5.4 | 263 | 18.7 | 3.7 | 274 |
| JPEG Q85 opt-huffman | **3.7** | 248 | 18.7 | 7.4 | 263 |
| JPEG Q95 opt-huffman | **4.9** | 603 | 27.0 | 14.5 | 761 |
| JPEG Q85 progressive | **10.5** | 237 | 18.7 | 13.7 | 246 |
| WebP Q80 eff=0 | **21** | 239 | 19.2 | 27 | 239 |
| WebP Q80 eff=4 | **52** | 231 | 19.2 | 59 | 231 |
| WebP Q80 eff=6 | **162** | 225 | 19.2 | 176 | 225 |
| WebP lossless eff=4 | **2.5** | 12 | inf | 191 | 20 |
| WebP near-lossless Q80 | 1836 | 522 | 51.1 | 456 | 536 |
| PNG level=3 | **9.4** | 89 | inf | 9.1 | 462 |
| PNG level=6 | **8.5** | 89 | inf | 13.0 | 461 |
| JXL d=2.0 eff=3 | 21 | 218 | 20.1 | 21 | 214 |
| JXL d=1.0 eff=5 | 46 | 307 | 22.8 | 26 | 249 |
| JXL lossless eff=3 | 38 | 1310 | inf | 30 | 1221 |
| JXL lossless eff=5 | **91** | 1126 | inf | 439 | 611 |
| JXL lossless eff=7 | **310** | 610 | inf | 627 | 430 |
| TIFF LZW | 21 | 448 | inf | 13 | 448 |
| TIFF Deflate | **7** | 55 | inf | 9 | 55 |

†JPEG: `optimize_huffman=false` (default) uses single-pass fixed tables — fastest for tiles.
`optimize_huffman=true` (two-pass) is faster at large images and produces ~10% smaller files.
‡TIFF LZW: tiff-rs/weezl LZW is slower than libvips's hand-tuned implementation.

### Photo 256×256 (tile) highlights

| Codec | crab ms | vips ms | Our size | vips size |
|-------|---------|---------|----------|-----------|
| JPEG Q75 | **1.2** | 1.2 | 28 KB | 28 KB |
| JPEG Q85 opt-huffman | **0.6** | 1.9 | 34 KB | 36 KB |
| JPEG Q95 opt-huffman | **1.0** | 3.9 | 82 KB | 104 KB |
| WebP Q80 eff=4 | **8.0** | 8.9 | 32 KB | 32 KB |
| WebP lossless eff=4 | **0.8** | 64 | 8 KB | 20 KB |
| PNG level=6 | **4.4** | 5.1 | 25 KB | 186 KB |
| JXL lossless eff=3 | **8.7** | 13.3 | 172 KB | 167 KB |
| TIFF LZW | **3.6** | 3.9 | 86 KB | 86 KB |

### Photo 1920×1080 (HD) highlights

| Codec | crab ms | vips ms | Notes |
|-------|---------|---------|-------|
| JPEG Q75 | 23 | 7.6 | Fixed tables; use opt-huffman for large images |
| JPEG Q75 opt-huffman | **12.9** | 18.2 | 1.4× faster |
| JPEG Q95 opt-huffman | **25.6** | 52.7 | **2.1× faster** |
| JPEG Q85 progressive | **49** | 52 | at parity |
| WebP lossless eff=4 | **24** | 401 | **17× faster** |
| JXL lossless eff=5 | **166** | 1667 | **10× faster** |
| JXL lossless eff=7 | **297** | 2996 | **10× faster** |
| TIFF Deflate | **17** | 26 | 1.5× faster, same size |

### Decoders

| Format | crabmagick | Reference | Speedup |
|--------|-----------|-----------|---------|
| JPEG   | **31 ms** | 33 ms (libjpeg-turbo) | 1.1× faster |
| JPEG (RST markers) | **10 ms** | 19 ms (libjpeg-turbo parallel) | **2× faster** |
| PNG    | **41 ms** | 83 ms (libpng) | **2× faster** |
| WebP   | **111 ms** | 90 ms (libwebp) | at parity |
| JXL    | **35 ms** | 47 ms (libjxl full RGB decode) | 1.3× faster |

### How performance is achieved (pure Rust, zero C)

| Technique | Applied to |
|-----------|-----------|
| AVX2 + FMA SIMD | JPEG IDCT, WebP YUV→RGB, WebP IDCT, JXL DCT8–64, JXL Gabor/EPF filters, XYB→sRGB |
| SSE4.1 SIMD | WebP IDCT 4×4, WebP residual add |
| Rayon parallel decode | JPEG RST segments, JXL pass-groups (70 groups/image) |
| Rayon parallel encode | JPEG RST entropy, WebP token partitions |
| JXL lossless palette | Auto-detected for ≤256-color images; matches libjxl's 1-bit/palette path |
| Branchless arithmetic | VP8 boolean decoder (cmov-friendly, eliminates 50% branch mispredictions) |
| 11-bit Huffman lookahead | JPEG: 2048-entry tables, decodes most symbols in 1 table lookup |
| Thread-local scratch buffers | JXL DCT: eliminates per-block heap allocations (~26K allocs/image) |

### Feature parity with libvips

| Feature | Status |
|---------|--------|
| Alpha channel (PNG, WebP, JXL, AVIF, TIFF) | ✅ Preserved end-to-end |
| ICC color profile | ✅ Extracted on decode, embedded on encode |
| EXIF metadata | ✅ Extracted from JPEG/WebP, re-embedded on JPEG encode |
| JXL lossless (natural images) | ✅ At parity |
| JXL lossless (palette/binary) | ✅ Auto-detected, 2.7× faster than libjxl |
| WebP near-lossless | ✅ Full libwebp near_lossless support; file sizes match libvips |
| TIFF LZW/Deflate/Packbits + predictor | ✅ Same output size as libvips |
| TIFF tiled output | ⚠️ Not yet implemented (tiff-rs limitation) |

### Known gaps vs libvips

| Case | Our result | libvips | Gap |
|------|-----------|---------|-----|
| JPEG Q75 baseline (tile 256×256) | at parity | libjpeg-turbo SIMD | — |
| JPEG Q75 baseline (HD 1920×1080) | 23 ms | 7.6 ms | 3× slower; use `optimize_huffman=true` |
| JXL lossy d=1.0 eff=7 | 362 KB | 251 KB | 44% larger (VarDCT optimization gap) |
| JXL lossless eff=5 | 1126 KB | 611 KB | 84% larger on noisy images |
| TIFF LZW | 1.5–2× slower | — | tiff-rs/weezl limitation |
| PNG level=1 (large noisy images) | large | small | adaptive filter needed for random content |

**Encoder options (all formats):** quality, progressive, optimize_huffman, chroma subsampling (JPEG);
effort, lossless, near-lossless, alpha quality (WebP); distance, effort, lossless, tier (JXL);
compression, filter, bit depth (PNG); compression, predictor (TIFF).

---

## Bundled binaries

The right binary is selected automatically from `/proc/cpuinfo` at startup.

| Binary | Target |
|--------|--------|
| `crabmagick-x86_64-linux` | x86_64 baseline |
| `crabmagick-x86_64-avx2-linux` | x86_64 AVX2 (Haswell 2013+, Ryzen 2017+) |
| `crabmagick-x86_64-avx512-linux` | x86_64 AVX-512 (Skylake-X+, Zen 4+) |
| `crabmagick-aarch64-linux` | aarch64 (AWS Graviton, RPi 4+) |
| `crabmagick-aarch64-sve-linux` | aarch64 SVE (Graviton 3+, Neoverse N2) |

---

## Build from source

```bash
rustup target add x86_64-unknown-linux-musl
sudo apt-get install musl-tools gcc-aarch64-linux-gnu

cd rust

# x86_64
RUSTFLAGS="-C target-cpu=x86-64         -C link-arg=-static-libgcc" \
  cargo build -p crabmagick-daemon --release --target x86_64-unknown-linux-musl
cp target/x86_64-unknown-linux-musl/release/crabmagick-daemon ../bin/crabmagick-x86_64-linux

RUSTFLAGS="-C target-cpu=haswell        -C link-arg=-static-libgcc" \
  cargo build -p crabmagick-daemon --release --target x86_64-unknown-linux-musl
cp target/x86_64-unknown-linux-musl/release/crabmagick-daemon ../bin/crabmagick-x86_64-avx2-linux

RUSTFLAGS="-C target-cpu=skylake-avx512 -C link-arg=-static-libgcc" \
  cargo build -p crabmagick-daemon --release --target x86_64-unknown-linux-musl
cp target/x86_64-unknown-linux-musl/release/crabmagick-daemon ../bin/crabmagick-x86_64-avx512-linux

# aarch64 (cross-compile)
rustup target add aarch64-unknown-linux-gnu
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
  RUSTFLAGS="-C target-cpu=generic -C link-arg=-static-libgcc" \
  cargo build -p crabmagick-daemon --release --target aarch64-unknown-linux-gnu
cp target/aarch64-unknown-linux-gnu/release/crabmagick-daemon ../bin/crabmagick-aarch64-linux
```
