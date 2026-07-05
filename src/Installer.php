<?php

declare(strict_types=1);

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

    private static function resolveSo(Event $event, string $packageDir, string $phpVer, $io): ?string
    {
        $arch    = php_uname('m');
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

    private static function resolveVariant(Event $event, string $arch, $io): ?string
    {
        $extra = $event->getComposer()->getPackage()->getExtra();
        $override = $extra['crabmagick']['variant'] ?? null;
        if (is_string($override) && $override !== '') {
            $io->write("<info>[crabmagick] Using variant override: {$override}</info>");
            return $override;
        }

        return self::detectVariant($arch);
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
