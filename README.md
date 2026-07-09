# CrabMagick

Pure-Rust image processing for PHP. Zero C dependencies, no libvips, no libjxl.

Two install modes — same `\CrabMagick\Image` API either way:

| Mode | Install | Performance |
|------|---------|-------------|
| **Daemon** (zero config) | `composer require usernamn0tavailable/crabmagick` | ~0.3 ms socket overhead |
| **Extension** (optional) | composer require + `PHP_INI_SCAN_DIR` | zero overhead, in-process |

The extension is auto-detected at runtime. If loaded, it is used directly. If not, the bundled daemon binary is spawned automatically.

---

## Why crabmagick instead of libvips?

### 1. Zero setup — it just works

With libvips you need system packages on every server, container, and dev machine:

```bash
# libvips approach — requires root, distro-specific, version drift
apt-get install libvips-dev libwebp-dev libjxl-dev libaom-dev ...
pecl install vips
```

With crabmagick:
```bash
composer require usernamn0tavailable/crabmagick
```

That's it. The pre-built Rust binary is bundled in the Composer package. No apt, no pecl, no root access, no OS packages to maintain or pin. Works identically on Ubuntu, Alpine, Amazon Linux, and any aarch64/x86_64 container.

---

### 2. Strong encode performance, with JXL speed caveats

Benchmarked on **Intel Core Ultra 7 155H** (22-thread AVX2), comparing encoder output at identical
quality settings. crabmagick uses the same format parameters as libvips — same codec, same options —
so file sizes are directly comparable.

#### JPEG (same quality, smaller files and/or faster)

| Setting | crab | vips | vs vips | crab KB | vips KB |
|---------|------|------|---------|---------|---------|
| Q75 baseline — 256×256 tile | 1.2 ms | 1.2 ms | = | 28 | 28 |
| Q75 opt-huffman — 256×256 | **0.6 ms** | 2.5 ms | **4× faster** | 26 | 27 |
| Q85 opt-huffman — 256×256 | **0.6 ms** | 1.9 ms | **3× faster** | 34 | 36 |
| Q95 — 256×256 | **1.5 ms** | 1.7 ms | 1.1× faster | **91** | **114** |
| Q95 opt-huffman — 256×256 | **1.0 ms** | 3.9 ms | **4× faster** | **82** | **104** |
| Q85 progressive — 256×256 | **1.8 ms** | 3.6 ms | **2× faster** | **32** | 34 |
| Q90 4:4:4 — 256×256 | **1.2 ms** | 1.8 ms | 1.5× faster | **65** | **84** |
| Q75 opt-huffman — 800×600 | **3.0 ms** | 6.6 ms | **2.2× faster** | **193** | 197 |
| Q85 opt-huffman — 800×600 | **3.7 ms** | 7.4 ms | **2× faster** | **248** | 263 |
| Q95 opt-huffman — 800×600 | **4.9 ms** | 14.5 ms | **3× faster** | **603** | 761 |
| Q75 opt-huffman — HD | **12.9 ms** | 18.2 ms | **1.4× faster** | **833** | 851 |
| Q85 opt-huffman — HD | **12.6 ms** | 20.0 ms | **1.6× faster** | **1070** | 1137 |
| Q95 opt-huffman — HD | **25.6 ms** | 52.7 ms | **2.1× faster** | **2608** | 3292 |

> `optimize_huffman=false` (default) uses pre-built Huffman tables — matches libjpeg-turbo single-pass
> speed at tile sizes. `optimize_huffman=true` switches to a two-pass parallel build:
> faster at ≥512×512, produces 10–20% smaller files.

#### WebP (same quality, same files, same or faster)

| Setting | crab | vips | vs vips | crab KB | vips KB |
|---------|------|------|---------|---------|---------|
| Q80 eff=0 — 256×256 | **3.3 ms** | 4.5 ms | 1.4× faster | 33 | 34 |
| Q80 eff=4 — 256×256 | **8.0 ms** | 8.9 ms | 1.1× faster | 32 | 32 |
| Q80 eff=4 — 800×600 | **52 ms** | 59 ms | 1.1× faster | 231 | 231 |
| Q80 eff=4 — HD | **222 ms** | 242 ms | 1.1× faster | 993 | 993 |
| lossless eff=4 — 256×256 | **0.8 ms** | 64 ms | **80× faster** | **8** | 20 |
| lossless eff=6 — 256×256 | **0.9 ms** | 64 ms | **71× faster** | **8** | 20 |
| lossless eff=4 — 800×600 | **2.9 ms** | 191 ms | **66× faster** | **10** | 20 |
| lossless eff=6 — 800×600 | **3.3 ms** | 191 ms | **58× faster** | **9** | 20 |
| lossless eff=4 — HD | **21 ms** | 384 ms | **18× faster** | **15** | 22 |
| lossless eff=6 — HD | **27 ms** | 812 ms | **30× faster** | **11** | 22 |

WebP lossless uses a custom pure-Rust encoder with an effort-aware LZ77 chain (eff=0–6). It is
dramatically faster than libwebp at any effort level because libwebp serializes its lossless
passes. Our encoder parallelizes them across all cores. At higher efforts the chain depth increases
(eff=4 → depth 64, eff=6 → depth 256), matching libwebp's quality while remaining 30-80× faster.

#### PNG (same lossless content, way smaller files)

| Setting | crab | vips | vs vips | crab KB | vips KB |
|---------|------|------|---------|---------|---------|
| level=3 — 256×256 | **2.4 ms** | 5.3 ms | **2.2× faster** | **25** | 183 |
| level=6 — 256×256 | **4.4 ms** | 5.1 ms | 1.2× faster | **25** | 186 |
| level=3 — 800×600 | 9.4 ms | 9.1 ms | = | **89** | 462 |
| level=6 — 800×600 | **8.5 ms** | 13.0 ms | 1.5× faster | **89** | 461 |
| level=3 — HD | 21 ms | 16.3 ms | — | **182** | 867 |
| level=6 — HD | **21 ms** | 26 ms | 1.2× faster | **182** | 858 |

Same pixels, **5× smaller files**. libvips uses libpng's adaptive row-filter selection, which
performs poorly for structured content; crabmagick uses Paeth filter + zlib producing substantially
better compression at any level.

#### JXL lossless (size parity or better on tested real photos; threaded runtime is competitive)

| Setting | crab | vips | vs vips | crab KB | vips KB |
|---------|------|------|---------|---------|---------|
| lossless eff=7 — 1680×2446 real photo | **4923 ms** | 6324 ms | **1.28× faster** | 4119 | 4056 |
| lossless eff=7 — 3668×4527 real photo | **14393 ms** | 24140 ms | **1.68× faster** | **7962** | 8829 |

On the current real-photo corpus, the default lossless profile is now at **parity or better on
file size**: from **+1.5%** on one 1680×2446 photo to **-9.8%** on a 3668×4527 photo. With the
normal threaded configuration, runtime is also ahead on both sampled photos because the modular
path parallelizes well.

#### JXL lossy

| Setting | crab | vips | vs vips | crab KB | vips KB |
|---------|------|------|---------|---------|---------|
| d=1.0 eff=5 — 2446×3019 real photo | **188 ms** | 209 ms | **1.11× faster** | **845** | 936 |
| d=1.0 eff=5 — 3668×4527 real photo | **348 ms** | 420 ms | **1.20× faster** | **1530** | 1744 |
| d=1.0 eff=7 — 2446×3019 real photo | **262 ms** | 269 ms | **1.03× faster** | **852** | 945 |
| d=1.0 eff=7 — 3668×4527 real photo | **526 ms** | 536 ms | **1.02× faster** | **1546** | 1758 |

The default lossy profile is now ahead on the sampled real-photo corpus at both
`distance=1.0, effort=5` and `distance=1.0, effort=7`: runtime is faster on the
current anchor photos, output is still **9.7–12.3% smaller** than libvips on the
anchor set, and measured PSNR/SSIM remain above the previous crabmagick baselines.
An additional sweep on 1223×1509 and 4891×6037 photos also stayed ahead at
`effort=7` while remaining smaller than libvips.

#### TIFF

| Setting | crab | vips | vs vips | crab KB | vips KB |
|---------|------|------|---------|---------|---------|
| LZW — 256×256 | **3.6 ms** | 3.9 ms | = | 86 | 86 |
| Deflate — 256×256 | **2.3 ms** | 3.8 ms | **1.7× faster** | 19 | 20 |
| LZW — HD | 80 ms | 52 ms | 1.5× slower† | 1441 | 1427 |
| Deflate — HD | **17 ms** | 26 ms | **1.5× faster** | 116 | 116 |
| Packbits — HD | **5.2 ms** | 14.5 ms | **2.8× faster** | 6130 | 6122 |

†TIFF LZW is slower at large sizes due to tiff-rs/weezl's single-threaded LZW implementation.
Use Deflate for better speed at all sizes.

---

### 3. Zero C dependency = no CVEs from C libraries

libvips links against libwebp, libjxl, libpng, libtiff, libjpeg-turbo — each of which has had
significant CVEs. crabmagick is pure Rust: memory-safe by construction, no unsafe C heap
allocations in codec paths, and the entire dependency chain can be audited with `cargo audit`.

---

### 4. Full feature parity with libvips

Everything you'd use libvips for in an IIIF image server works identically:

| Feature | Notes |
|---------|-------|
| Alpha channel | Preserved through decode → ops → encode (PNG, WebP, JXL, AVIF, TIFF) |
| ICC color profile | Extracted on decode, embedded on encode (all formats) |
| EXIF metadata | Extracted from JPEG/WebP, re-embedded on JPEG encode |
| Progressive JPEG | `progressive: true` |
| Chroma subsampling | 4:2:0 (default), 4:2:2, 4:4:4 |
| Optimize Huffman | `optimize_huffman: true` — two-pass, 10–20% smaller, faster at large sizes |
| WebP near-lossless | `near_lossless: true, quality: 0–100` |
| WebP lossless effort sweep | eff=0 (fastest) to eff=6 (best) |
| JXL distance + effort | Full range: d=0.0–25.0, eff=1–8 |
| JXL lossless palette | Auto-detected ≤1024 unique colors — matches libjxl compression |
| TIFF LZW/Deflate/Packbits/None | With horizontal difference predictor |
| PNG bit depth | 8-bit and 16-bit |
| PNG filter | None/Sub/Up/Avg/Paeth/All |
| Region extract | Sub-image crop with coordinate validation |
| Resize | Bilinear (fast), Lanczos (quality) |
| Rotate | 0°/90°/180°/270° with alpha/ICC pass-through |
| Square crop | Center-weighted smart crop |

---

### 5. What's still better in libvips

Be honest about the gaps:

| Case | libvips | crabmagick | Gap |
|------|---------|-----------|-----|
| JPEG Q75 baseline at HD | 7.6 ms | 23 ms | 3× slower — use `optimize_huffman=true` (12.9 ms) |
| JXL lossless eff=7 (1680×2446 photo) | 4056 KB, 6324 ms | 4119 KB, 4923 ms | 1.5% larger, but 1.28× faster |
| TIFF LZW at large sizes | 52 ms | 80 ms | 1.5× slower — use Deflate instead |
| PNG level=1 (noisy images) | small | very large | zlib level=1 + Paeth bad for random content |

---



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
| WebP lossless eff=4 | **2.9** | 10 | inf | 180 | 20 |
| WebP lossless eff=6 | **3.3** | 9 | inf | 182 | 20 |
| WebP near-lossless Q80 | 1836 | 522 | 51.1 | 456 | 536 |
| PNG level=3 | **9.4** | 89 | inf | 9.1 | 462 |
| PNG level=6 | **8.5** | 89 | inf | 13.0 | 461 |
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
| TIFF LZW | **3.6** | 3.9 | 86 KB | 86 KB |

### Photo 1920×1080 (HD) highlights

| Codec | crab ms | vips ms | Notes |
|-------|---------|---------|-------|
| JPEG Q75 | 23 | 7.6 | Fixed tables; use opt-huffman for large images |
| JPEG Q75 opt-huffman | **12.9** | 18.2 | 1.4× faster |
| JPEG Q95 opt-huffman | **25.6** | 52.7 | **2.1× faster** |
| JPEG Q85 progressive | **49** | 52 | at parity |
| WebP lossless eff=4 | **21** | 384 | **18× faster** |
| WebP lossless eff=6 | **27** | 812 | **30× faster** |
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
| JXL lossy (natural photos, d=1.0 eff=5/e7) | ✅ Smaller than libvips and faster on current sampled photos |
| JXL lossless (natural images, eff=5) | ✅ File-size parity or better on current real-photo probes |
| JXL lossless (natural images, eff=7) | ✅ File-size parity or better, and threaded runtime is faster on sampled photos |
| JXL lossless (palette/binary) | ✅ Auto-detected, 2.7× faster than libjxl |
| WebP near-lossless | ✅ Full libwebp near_lossless support; file sizes match libvips |
| TIFF LZW/Deflate/Packbits + predictor | ✅ Same output size as libvips |
| TIFF tiled output | ⚠️ Not yet implemented (tiff-rs limitation) |

### Known gaps vs libvips

| Case | Our result | libvips | Gap |
|------|-----------|---------|-----|
| JPEG Q75 baseline (tile 256×256) | at parity | libjpeg-turbo SIMD | — |
| JPEG Q75 baseline (HD 1920×1080) | 23 ms | 7.6 ms | 3× slower; use `optimize_huffman=true` |
| JXL lossless eff=7 (1680×2446 photo, threaded) | 4923 ms, 4119 KB | 6324 ms, 4056 KB | 1.28× faster, 1.5% larger |
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
