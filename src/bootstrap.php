<?php

declare(strict_types=1);

/**
 * Loaded automatically via Composer's autoload.files on every
 * `require vendor/autoload.php`.
 *
 * Finds the pre-built crabmagick-daemon binary for the current architecture,
 * starts it in the background if it is not already running, and registers
 * the socket path with Runtime so all \Crabmagick\Image calls work.
 *
 * Zero configuration required. No php.ini changes. No sudo.
 */
(static function (): void {
    if (\Crabmagick\Runtime::isReady()) {
        return;
    }

    $arch   = php_uname('m');
    $binDir = __DIR__ . '/../bin';

    $bin = self_crabmagick_find_binary($binDir, $arch);
    if ($bin === null) {
        trigger_error(
            '[crabmagick] No pre-built daemon binary found for arch ' . $arch . '. '
            . 'See https://github.com/UsernameN0tAvailable/crabmagick for supported platforms.',
            E_USER_WARNING,
        );
        return;
    }

    $uid        = function_exists('posix_getuid') ? posix_getuid() : getmypid();
    $socketPath = sys_get_temp_dir() . '/crabmagick-' . $uid . '.sock';

    // If socket already exists, register and trust it (another worker started it).
    if (file_exists($socketPath)) {
        \Crabmagick\Runtime::setSocketPath($socketPath);
        return;
    }

    // Spawn daemon detached from the current process.
    $cmd = escapeshellarg($bin) . ' --socket ' . escapeshellarg($socketPath);
    $descriptors = [
        0 => ['file', '/dev/null', 'r'],
        1 => ['file', '/dev/null', 'w'],
        2 => ['file', '/dev/null', 'w'],
    ];
    $proc = @proc_open($cmd, $descriptors, $pipes, null, null, ['bypass_shell' => true]);
    if ($proc === false) {
        trigger_error('[crabmagick] Failed to spawn daemon: ' . $cmd, E_USER_WARNING);
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
        trigger_error('[crabmagick] Daemon did not create socket in time: ' . $socketPath, E_USER_WARNING);
        return;
    }

    \Crabmagick\Runtime::setSocketPath($socketPath);
})();

// ── Helpers ───────────────────────────────────────────────────────────────────

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
