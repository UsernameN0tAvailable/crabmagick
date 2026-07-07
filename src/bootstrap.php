<?php

declare(strict_types=1);

/**
 * Loaded automatically via Composer's autoload.files on every
 * `require vendor/autoload.php`.
 *
 * Finds the pre-built crabmagick-daemon binary for the current architecture,
 * starts it in the background if it is not already running, and registers
 * the socket path with Runtime so all \CrabMagick\Image calls work.
 *
 * Zero configuration required. No php.ini changes. No sudo.
 */
(static function (): void {
    if (\CrabMagick\Runtime::isReady()) {
        return;
    }

    // ── Extension fast-path ───────────────────────────────────────────────────
    // If the CrabMagick PHP extension is loaded it exposes crabmagick_process()
    // and \CrabMagick\Image natively (in-process, zero socket overhead).
    // Skip daemon spawn entirely — the autoloader will resolve \CrabMagick\Image
    // to the native class and never load our PHP shim.
    if (function_exists('crabmagick_process')) {
        \CrabMagick\Runtime::setUsingExtension();
        return;
    }

    // ── Pre-flight checks ─────────────────────────────────────────────────────

    // Unix sockets + pre-built Linux binaries: Linux only.
    if (PHP_OS_FAMILY !== 'Linux') {
        trigger_error(
            '[CrabMagick] Unsupported OS "' . PHP_OS_FAMILY . '". '
            . 'Pre-built daemon binaries are Linux-only. '
            . 'See https://github.com/UsernameN0tAvailable/crabmagick for build instructions.',
            E_USER_WARNING,
        );
        return;
    }

    // proc_open is required to spawn the daemon. It is sometimes listed in
    // disable_functions on shared hosting or hardened FPM pools.
    if (!function_exists('proc_open') || self_crabmagick_is_disabled('proc_open')) {
        trigger_error(
            '[CrabMagick] proc_open() is disabled (disable_functions). '
            . 'The daemon cannot be spawned automatically. '
            . 'Start it manually: bin/crabmagick-<arch>-linux --socket /tmp/crabmagick.sock',
            E_USER_WARNING,
        );
        return;
    }

    // A writable temp dir is needed for the Unix socket file.
    $tmpDir = sys_get_temp_dir();
    if (!is_writable($tmpDir)) {
        trigger_error(
            '[CrabMagick] Temp directory "' . $tmpDir . '" is not writable. '
            . 'Set TMPDIR to a writable path or start the daemon manually.',
            E_USER_WARNING,
        );
        return;
    }

    // ── Find binary ───────────────────────────────────────────────────────────

    $arch   = php_uname('m');
    $binDir = __DIR__ . '/../bin';

    $bin = self_crabmagick_find_binary($binDir, $arch);
    if ($bin === null) {
        trigger_error(
            '[CrabMagick] No pre-built daemon binary found for arch "' . $arch . '". '
            . 'See https://github.com/UsernameN0tAvailable/crabmagick for supported platforms.',
            E_USER_WARNING,
        );
        return;
    }

    // ── Spawn (or attach to existing) daemon ──────────────────────────────────

    $uid        = function_exists('posix_getuid') ? posix_getuid() : getmypid();
    $socketPath = $tmpDir . '/crabmagick-' . $uid . '.sock';

    // If socket already exists, register and trust it (another worker started it).
    if (file_exists($socketPath)) {
        \CrabMagick\Runtime::setSocketPath($socketPath);
        return;
    }

    // Spawn daemon detached from the current process.
    $descriptors = [
        0 => ['file', '/dev/null', 'r'],
        1 => ['file', '/dev/null', 'w'],
        2 => ['file', '/dev/null', 'w'],
    ];
    $proc = @proc_open(
        [$bin, '--socket', $socketPath],
        $descriptors,
        $pipes,
        null,
        null,
        ['bypass_shell' => true],
    );
    if ($proc === false) {
        trigger_error('[CrabMagick] Failed to spawn daemon binary: ' . $bin, E_USER_WARNING);
        return;
    }
    // Don't wait — let it run in the background.
    proc_close($proc);

    // Wait up to 2 s for the socket to appear.
    $deadline = microtime(true) + 2.0;
    while (!file_exists($socketPath) && microtime(true) < $deadline) {
        usleep(5_000);
    }

    if (!file_exists($socketPath)) {
        trigger_error(
            '[CrabMagick] Daemon did not create socket at "' . $socketPath . '" within 2 s. '
            . 'Run the binary manually to see its error output: ' . $bin,
            E_USER_WARNING,
        );
        return;
    }

    \CrabMagick\Runtime::setSocketPath($socketPath);
})();

// ── Helpers ───────────────────────────────────────────────────────────────────

function self_crabmagick_is_disabled(string $fn): bool
{
    $disabled = array_map('trim', explode(',', (string) ini_get('disable_functions')));
    return in_array($fn, $disabled, true);
}

function self_crabmagick_find_binary(string $binDir, string $arch): ?string
{
    $variant = self_crabmagick_detect_variant($arch);
    foreach (array_unique(array_filter([$variant, "{$arch}-linux"])) as $suffix) {
        $path = "{$binDir}/crabmagick-{$suffix}";
        if (is_executable($path)) {
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
            return 'x86_64-avx512-linux';
        }
        if (preg_match('/\bflags\b.*\bavx2\b/m', $cpuinfo)) {
            return 'x86_64-avx2-linux';
        }
        return 'x86_64-linux';
    }
    if ($arch === 'aarch64' || $arch === 'arm64') {
        $cpuinfo = @file_get_contents('/proc/cpuinfo') ?: '';
        if (preg_match('/\bFeatures\b.*\bsve\b/mi', $cpuinfo)) {
            return 'aarch64-sve-linux';
        }
        return 'aarch64-linux';
    }
    return null;
}
