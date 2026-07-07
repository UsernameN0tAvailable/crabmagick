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
| WebP   | ✅ | ✅ |
| JXL    | ✅ | ✅ |
| AVIF   | ✅ | ✅ |
| TIFF   | ✅ | — |
| GIF    | ✅ | — |
| BMP    | ✅ | — |
| SVG    | ✅ | — |
| JP2/J2K | ✅ | — |

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
