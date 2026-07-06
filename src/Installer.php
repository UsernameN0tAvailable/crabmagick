<?php

declare(strict_types=1);

namespace Crabmagick;

use Composer\Script\Event;

class CrabmagickInstaller
{
    public static function postInstall(Event $event): void
    {
        $io = $event->getIO();
        $packageDir = realpath(__DIR__ . '/..') ?: (__DIR__ . '/..');

        $soPath = self::resolveSo($packageDir);
        if ($soPath === null) {
            $io->writeError('<warning>[crabmagick] No pre-built binary found for this platform. '
                . 'See https://github.com/UsernameN0tAvailable/crabmagick for supported platforms.</warning>');
            return;
        }

        file_put_contents("{$packageDir}/crabmagick.ini", "ffi.enable = true\n");

        $io->write("<info>[crabmagick] Selected binary: {$soPath}</info>");
        $io->write('<info>[crabmagick] Activation — pick one option:</info>');
        $io->write("<info>  1. PHP_INI_SCAN_DIR (no sudo): PHP_INI_SCAN_DIR=:{$packageDir} php ...</info>");
        $io->write("<info>  2. Symlink into conf.d:       sudo ln -s {$packageDir}/crabmagick.ini /etc/php/cli/conf.d/30-crabmagick.ini</info>");
        $io->write('<info>     (crabmagick.ini only sets ffi.enable=true — the binary is bundled and found automatically)</info>');
    }

    private static function resolveSo(string $packageDir): ?string
    {
        $arch = php_uname('m');
        $variant = self::detectVariant($arch);

        foreach (array_unique(array_filter([$variant, $arch])) as $v) {
            $path = realpath("{$packageDir}/ext/crabmagick-{$v}-linux.so");
            if ($path !== false) {
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
}
