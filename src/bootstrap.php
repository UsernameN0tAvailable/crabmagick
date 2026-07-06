<?php

declare(strict_types=1);

/**
 * Loaded automatically via Composer's autoload.files on every
 * require vendor/autoload.php.
 *
 * Finds the pre-built crabmagick shared library for the current
 * architecture and loads it via PHP FFI.
 *
 * Requirements:
 *   - ffi.enable = true  (set by crabmagick.ini; activate with PHP_INI_SCAN_DIR)
 *   - ext-ffi            (bundled with PHP ≥ 7.4)
 *
 * No php.ini extension= line needed. No sudo. No PHP-version-specific binaries.
 */
(static function (): void {
    if (\Crabmagick\Runtime::isLoaded()) {
        return;
    }

    if (!extension_loaded('ffi')) {
        trigger_error(
            '[crabmagick] The PHP FFI extension is not available. '
            . 'Make sure ext-ffi is compiled in (it ships with PHP ≥ 7.4 by default).',
            E_USER_NOTICE,
        );
        return;
    }

    $extDir = __DIR__ . '/../ext';
    $arch = php_uname('m');
    $so = self_crabmagick_resolve($extDir, $arch);

    if ($so === null) {
        trigger_error(
            '[crabmagick] No pre-built binary found for arch ' . $arch . '. '
            . 'See https://github.com/UsernameN0tAvailable/crabmagick for supported platforms.',
            E_USER_NOTICE,
        );
        return;
    }

    try {
        \Crabmagick\Runtime::load($so);
    } catch (\Throwable $e) {
        trigger_error('[crabmagick] Failed to load native library: ' . $e->getMessage(), E_USER_WARNING);
    }
})();

function self_crabmagick_resolve(string $extDir, string $arch): ?string
{
    $variant = self_crabmagick_detect_variant($arch);
    foreach (array_unique(array_filter([$variant, $arch])) as $v) {
        $path = "{$extDir}/crabmagick-{$v}-linux.so";
        if (file_exists($path)) {
            return $path;
        }
    }
    return null;
}

function self_crabmagick_detect_variant(string $arch): ?string
{
    if ($arch === 'x86_64') {
        $cpuinfo = @file_get_contents('/proc/cpuinfo') ?: '';
        if (preg_match('/\bflags\b.*\bavx512f\b/m', $cpuinfo)) {
            return 'x86_64-avx512';
        }
        if (preg_match('/\bflags\b.*\bavx2\b/m', $cpuinfo)) {
            return 'x86_64-avx2';
        }
        return null;
    }

    if ($arch === 'aarch64' || $arch === 'arm64') {
        $cpuinfo = @file_get_contents('/proc/cpuinfo') ?: '';
        if (preg_match('/\bFeatures\b.*\bsve\b/mi', $cpuinfo)) {
            return 'aarch64-sve';
        }
        return null;
    }

    return null;
}
