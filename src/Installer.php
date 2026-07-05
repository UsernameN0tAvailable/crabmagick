<?php

declare(strict_types=1);

/**
 * Composer post-install/post-update hook.
 *
 * Resolves the best pre-built .so for the current platform and writes
 * crabmagick.ini with the absolute extension path.
 *
 * Selection order:
 *   1. "extra.crabmagick.variant" in the root composer.json (manual override)
 *   2. Auto-detect: probes /proc/cpuinfo for CPU features (avx2, etc.)
 *   3. Falls back to the generic arch binary (x86_64, aarch64)
 *
 * Override example in your project's composer.json:
 *   "extra": { "crabmagick": { "variant": "x86_64-avx2" } }
 *
 * To activate without touching system php.ini:
 *   PHP_INI_SCAN_DIR=:/path/to/vendor/usernamn0tavailable/crabmagick php ...
 * Or symlink:
 *   sudo ln -s .../crabmagick.ini /etc/php/X.Y/cli/conf.d/30-crabmagick.ini
 */

use Composer\Script\Event;

class CrabmagickInstaller
{
    public static function postInstall(Event $event): void
    {
        $io         = $event->getIO();
        $packageDir = realpath(__DIR__ . '/..') ?: (__DIR__ . '/..');
        $phpVer     = PHP_MAJOR_VERSION . '.' . PHP_MINOR_VERSION;

        $soPath = self::resolveSo($event, $packageDir, $phpVer, $io);
        if ($soPath === null) {
            return;
        }

        file_put_contents("{$packageDir}/crabmagick.ini", "extension={$soPath}\n");

        $io->write("<info>[crabmagick] Selected binary: {$soPath}</info>");
        $io->write('<info>[crabmagick] To activate — pick one option:</info>');
        $io->write("<info>  1. Scan dir (no sudo): PHP_INI_SCAN_DIR=:{$packageDir} php ...</info>");
        $io->write("<info>  2. Symlink:            sudo ln -s {$packageDir}/crabmagick.ini /etc/php/{$phpVer}/cli/conf.d/30-crabmagick.ini</info>");
    }

    /**
     * Resolve the best .so path for this platform.
     *
     * Candidate names tried in order:
     *   crabmagick-php{ver}-{variant}-linux.so   (variant = override or detected)
     *   crabmagick-php{ver}-{arch}-linux.so       (generic fallback)
     */
    private static function resolveSo(Event $event, string $packageDir, string $phpVer, $io): ?string
    {
        $arch    = php_uname('m'); // x86_64 | aarch64 | arm64
        $variant = self::resolveVariant($event, $arch, $io);

        $candidates = array_unique(array_filter([$variant, $arch]));
        foreach ($candidates as $v) {
            $name = "crabmagick-php{$phpVer}-{$v}-linux.so";
            $path = realpath("{$packageDir}/ext/{$name}");
            if ($path !== false) {
                return $path;
            }
        }

        $tried = implode(', ', array_map(
            fn($v) => "ext/crabmagick-php{$phpVer}-{$v}-linux.so",
            $candidates
        ));
        $io->writeError("<warning>[crabmagick] No pre-built binary found. Tried: {$tried}</warning>");
        return null;
    }

    /**
     * Returns the variant string to try first, in order:
     *   1. "extra.crabmagick.variant" from root composer.json
     *   2. CPU feature detection from /proc/cpuinfo
     *   3. null (fall through to generic arch)
     */
    private static function resolveVariant(Event $event, string $arch, $io): ?string
    {
        // 1. Manual override via root composer.json extra
        $extra   = $event->getComposer()->getPackage()->getExtra();
        $override = $extra['crabmagick']['variant'] ?? null;
        if (is_string($override) && $override !== '') {
            $io->write("<info>[crabmagick] Using variant override: {$override}</info>");
            return $override;
        }

        // 2. Auto-detect CPU features
        return self::detectVariant($arch);
    }

    /**
     * Detect the best variant for this CPU.
     * Returns e.g. "x86_64-avx2" or null (use generic).
     */
    private static function detectVariant(string $arch): ?string
    {
        if ($arch !== 'x86_64') {
            return null; // aarch64 etc — only one variant currently
        }

        $cpuinfo = @file_get_contents('/proc/cpuinfo') ?: '';
        if (str_contains($cpuinfo, ' avx2 ') || str_contains($cpuinfo, "\tavx2\t")
            || preg_match('/\bflags\b.*\bavx2\b/m', $cpuinfo)) {
            return 'x86_64-avx2';
        }

        return null;
    }
}
