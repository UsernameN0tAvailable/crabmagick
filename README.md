# crabmagick

Pure-Rust libvips replacement for PHP — self-contained Composer package with a bundled native extension. Zero C dependencies, optimized for cached JXL-backed image requests.

## What's inside

| Crate | Role |
|---|---|
| `crates/crabmagick-core` | Decode / crop / resize / encode pipeline + two-level in-process cache |
| `crates/crabmagick-php` | `ext-php-rs` PHP extension exposing `\Crabmagick\Image` |
| `ext/` | Pre-built `.so` binaries per PHP version + arch |
| `src/bootstrap.php` | Auto-loaded by Composer — tries `dl()`, otherwise hints `PHP_INI_SCAN_DIR` |
| `src/Installer.php` | Composer post-install hook — generates `crabmagick.ini` with absolute path |

## Installation

```bash
composer require usernamn0tavailable/crabmagick
```

After install, activate the extension **without touching system php.ini**:

```bash
# Option 1 — PHP_INI_SCAN_DIR (no sudo)
PHP_INI_SCAN_DIR=:/path/to/vendor/usernamn0tavailable/crabmagick php -S 0.0.0.0:8088 -t public/

# Option 2 — symlink into conf.d
sudo ln -s $(pwd)/vendor/usernamn0tavailable/crabmagick/crabmagick.ini \
           /etc/php/8.4/cli/conf.d/30-crabmagick.ini
```

`crabmagick.ini` is generated automatically by `composer install` with the correct absolute path.

## PHP API

```php
// Fluent builder
$bytes = (new \Crabmagick\Image('/path/to/file.jxl'))
    ->region(100, 200, 512, 512)   // crop (triggers crop-during-decode)
    ->resize(256, 256)
    ->rotate(90)
    ->encode('webp', 82);          // 'jpeg' | 'webp' | 'png' | 'jxl'

// One-shot
$bytes = \Crabmagick\process($path, $rx, $ry, $rw, $rh, $outW, $outH, 'jpeg', 85);

// Dimensions (reads JXL header only, no full decode)
$info = \Crabmagick\info($path); // ['width' => 3360, 'height' => 4892]

// Extension check
\Crabmagick\Image::isAvailable(); // bool
```

## Performance

Benchmarked on 3360×4892 JXL source, 512×512 output tile:

| Request type | Time |
|---|---|
| Cold tile (crop-during-decode) | **~29 ms** |
| Tile cache hit (same crop, different size) | **~2 ms** |
| Output cache hit (exact repeat) | **< 0.01 ms** |
| Full image 800px (unavoidable full decode) | **~570 ms** |

libvips takes ~540 ms for the same cold tile. The 19× speedup comes from JXL group-level crop-during-decode via `jxl-oxide`'s `set_image_region`.

## Build

Requires PHP 8.x headers (`php-dev` package) on the build machine.

```bash
cd rust
cargo build -p crabmagick-php --release
# → target/release/libcrabmagick.so
cp target/release/libcrabmagick.so \
   ../ext/crabmagick-php8.4-x86_64-linux.so
```

### Adding a new architecture

Build on the target machine (or cross-compile), then commit the binary:

```bash
cp target/release/libcrabmagick.so \
   ../ext/crabmagick-php8.4-aarch64-linux.so
git add ext/crabmagick-php8.4-aarch64-linux.so
```

`bootstrap.php` and `Installer.php` detect arch via `php_uname('m')` and select the correct file automatically.

## Bundled decoders / encoders

All pure Rust, zero C dependencies:

- JXL encode/decode — `jxl-encoder` + `jxl-oxide`
- JPEG encode/decode — `zenjpeg`
- WebP encode/decode — `fast-webp`
- PNG — via `image` crate
- JP2/J2K — `zen-jp2` (optional feature `jp2`)
- Resize — `fast_image_resize` (SIMD)

