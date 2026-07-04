# oxipix

Pure-Rust image processing for PHP: Composer package + native extension, built around JXL decode, fast resize, and cached output encoding.

## Workspace

- `rust/crates/oxipix-core` — decode/crop/resize/encode pipeline + two-level cache
- `rust/crates/oxipix-php` — `ext-php-rs` extension exposing `Oxipix\\Image`
- `src/ImageProcessor.php` — small PHP wrapper around the native extension
- `ffi/oxipix.h` — C header for future FFI fallback integration

## Build

```bash
cd rust
cargo build -p oxipix-php --release
```

Then load the generated `oxipix.so` in PHP and use the wrapper:

```php
<?php

use Oxipix\ImageProcessor;

$img = new ImageProcessor('/path/to/file.jxl');
$data = $img->region(0, 0, 512, 512)->resize(256, 256)->encode('webp', 82);
```
