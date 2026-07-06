# crabmagick

Pure-Rust image processing for PHP. Zero C dependencies, no libvips, no libjxl.

Two install modes — same `\Crabmagick\Image` API either way:

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
php -r "var_dump(\Crabmagick\Image::isAvailable());"
# bool(true) — extension path
# bool(true) — daemon path (both return true)

php -r "var_dump(\Crabmagick\Runtime::isUsingExtension());"
# bool(true) if extension is loaded
```

---

## Requirements

- PHP ≥ 8.1, Linux x86_64 or aarch64
- No libvips, no libjxl, no C compiler, no Rust toolchain

---

## PHP API

```php
// Fluent builder
$bytes = (new \Crabmagick\Image('/path/to/file.jxl'))
    ->region(0, 0, 512, 512)   // crop (x, y, w, h)
    ->resize(256, 256)          // output size (0 = proportional)
    ->rotate(90)                // clockwise: 90 | 180 | 270
    ->encode('jpeg', 85);       // 'jpeg' | 'webp' | 'png' | 'jxl' | 'avif'

// Square crop (IIIF "square" region)
$bytes = (new \Crabmagick\Image($path))->square()->resize(512)->encode('webp', 80);

// Page/frame selection (multi-page TIFF, animated WebP)
$bytes = (new \Crabmagick\Image($path))->page(2)->resize(1024)->encode('jpeg');

// One-shot helper
$bytes = \Crabmagick\process($path, $rx, $ry, $rw, $rh, $outW, $outH, 'jpeg', 85);

// Dimensions without full decode
$info = \Crabmagick\info($path);          // ['width' => 3360, 'height' => 4892]
$info = (new \Crabmagick\Image($path))->getInfo();
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
