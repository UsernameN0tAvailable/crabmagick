<?php

declare(strict_types=1);

namespace CrabMagick;

use Composer\Script\Event;

class Installer
{
    public static function postInstall(Event $event): void
    {
        $io         = $event->getIO();
        $packageDir = realpath(__DIR__ . '/..') ?: (__DIR__ . '/..');
        $arch       = php_uname('m');
        $bin        = self::findBinary($packageDir, $arch);

        if ($bin === null) {
            $io->writeError(
                '<warning>[CrabMagick] No pre-built daemon binary found for arch ' . $arch . '. '
                . 'See https://github.com/UsernameN0tAvailable/crabmagick for supported platforms.</warning>'
            );
            return;
        }

        // Ensure the binary is executable (git may have dropped the bit).
        if (!is_executable($bin)) {
            chmod($bin, 0755);
        }

        $io->write("<info>[CrabMagick] Daemon binary ready: {$bin}</info>");
        $io->write('<info>[CrabMagick] No php.ini changes needed — the daemon starts automatically.</info>');
    }

    private static function findBinary(string $packageDir, string $arch): ?string
    {
        $variant = self::detectVariant($arch);
        foreach (array_unique(array_filter([$variant, "{$arch}-linux"])) as $suffix) {
            $path = "{$packageDir}/bin/crabmagick-{$suffix}";
            if (file_exists($path)) {
                return $path;
            }
        }
        return null;
    }

    private static function detectVariant(string $arch): ?string
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
}
