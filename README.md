# crabmagick

Pure-Rust image processing for PHP with a bundled C-ABI shared library loaded at runtime through PHP FFI. No PHP headers at build time, no `extension=` line in `php.ini`, and the same Linux `.so` works across PHP 8.1–8.x.

## What's inside

| Path | Role |
|---|---|
| `rust/crates/crabmagick-core` | Decode / crop / resize / encode pipeline |
| `rust/crates/crabmagick-ffi` | Plain C-ABI `cdylib` loaded by PHP FFI |
| `rust/crates/crabmagick-php` | Legacy Zend extension crate, kept for optional extension-based builds |
| `ffi/oxipix.h` | Stable C header for the native ABI |
| `src/Runtime.php` | Loads the bundled shared library via `FFI::cdef()` |
| `src/Image.php` | Fluent PHP API |
| `src/bootstrap.php` | Composer autoload bootstrap that resolves and loads the native library |
| `src/Installer.php` | Composer post-install hook that generates `crabmagick.ini` |

## Why FFI

- No PHP-version coupling in the native binary
- No PHP headers or `php-dev` package needed in CI
- No `extension=` activation step
- Minimal runtime dependencies: the release build is configured for `panic = "abort"` and uses static `libgcc` / `libm` link args to avoid `libgcc_s.so.1` and `libm.so.6`

## Installation

```bash
composer require usernamn0tavailable/crabmagick
```

Requirements:

- PHP 8.1+
- `ext-ffi`
- Linux x86_64 or aarch64

## Activation

`composer install` or `composer update` writes `crabmagick.ini` with:

```ini
ffi.enable = true
```

Then activate with one of these options:

```bash
# Option 1 — PHP_INI_SCAN_DIR (no sudo)
PHP_INI_SCAN_DIR=:/path/to/vendor/usernamn0tavailable/crabmagick php ...

# Option 2 — symlink the generated ini
sudo ln -s /path/to/vendor/usernamn0tavailable/crabmagick/crabmagick.ini \
           /etc/php/cli/conf.d/30-crabmagick.ini
```

No `extension=` line is required. `src/bootstrap.php` finds the bundled `ext/crabmagick-*-linux.so` for the current CPU and loads it through FFI.

## Binary selection

At runtime and during install, crabmagick picks the best bundled binary for the host:

| CPU | Selected binary |
|---|---|
| x86_64 with AVX-512 | `crabmagick-x86_64-avx512-linux.so` |
| x86_64 with AVX2 | `crabmagick-x86_64-avx2-linux.so` |
| generic x86_64 | `crabmagick-x86_64-linux.so` |
| aarch64 with SVE | `crabmagick-aarch64-sve-linux.so` |
| generic aarch64 | `crabmagick-aarch64-linux.so` |

## PHP API

```php
$bytes = (new \Crabmagick\Image('/path/to/file.jxl'))
    ->region(100, 200, 512, 512)
    ->resize(256, 256)
    ->page(0)
    ->rotate(90)
    ->square()
    ->encode('webp', 82);

$bytes = \Crabmagick\process($path, $rx, $ry, $rw, $rh, $outW, $outH, 'jpeg', 85);

$info = \Crabmagick\info($path); // ['width' => 3360, 'height' => 4892]

\Crabmagick\Image::isAvailable(); // bool
```

Supported output formats: `jpeg`, `webp`, `png`, `jxl`, `avif`.

## Performance

Benchmarked on a 3360×4892 JXL source with 512×512 output tiles:

| Request type | Time |
|---|---|
| Cold tile (crop-during-decode) | ~29 ms |
| Full image 800px | ~570 ms |

libvips takes ~540 ms for the same cold tile. The speedup comes from JXL group-level crop-during-decode.

## Build

```bash
cd rust
RUSTFLAGS='-C link-arg=-static-libgcc -C link-arg=-static-libm' \
cargo build -p crabmagick-ffi --release
cp target/release/libcrabmagick.so ../ext/crabmagick-x86_64-linux.so
ldd ../ext/crabmagick-x86_64-linux.so
```

Target-specific builds used by CI:

```bash
# AVX2
RUSTFLAGS='-C target-cpu=haswell -C link-arg=-static-libgcc -C link-arg=-static-libm' \
cargo build -p crabmagick-ffi --release

# AVX-512
RUSTFLAGS='-C target-cpu=skylake-avx512 -C link-arg=-static-libgcc -C link-arg=-static-libm' \
cargo build -p crabmagick-ffi --release
```

## Supported platforms

- `x86_64`
- `x86_64-avx2`
- `x86_64-avx512`
- `aarch64`
- `aarch64-sve`

The native ABI is defined in `ffi/oxipix.h`. Other language runtimes can reuse the same shared library without any PHP-specific build step.
