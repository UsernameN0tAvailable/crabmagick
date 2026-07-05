<?php

declare(strict_types=1);

/**
 * Composer post-install/post-update hook.
 *
 * Generates an extension ini snippet (crabmagick.ini) inside this package
 * directory containing the absolute path to the correct pre-built .so.
 *
 * Usage (automatic — driven by composer.json scripts):
 *   composer install   → generates vendor/.../crabmagick/crabmagick.ini
 *
 * To activate without touching system php.ini:
 *   PHP_INI_SCAN_DIR=:/path/to/vendor/usernamn0tavailable/crabmagick php ...
 *
 * Or to install globally:
 *   sudo ln -s /path/to/vendor/.../crabmagick.ini /etc/php/X.Y/cli/conf.d/30-crabmagick.ini
 */

use Composer\Script\Event;

class CrabmagickInstaller
{
    public static function postInstall(Event $event): void
    {
        $io = $event->getIO();

        $packageDir = __DIR__ . '/..';
        $phpVer     = PHP_MAJOR_VERSION . '.' . PHP_MINOR_VERSION;
        $arch       = php_uname('m');
        $soName     = "crabmagick-php{$phpVer}-{$arch}-linux.so";
        $soPath     = realpath("{$packageDir}/ext/{$soName}");

        if ($soPath === false) {
            $io->writeError(
                "<warning>[crabmagick] No pre-built binary found for PHP {$phpVer} / {$arch}. "
                . "Looked for ext/{$soName}</warning>"
            );
            return;
        }

        $ini = "extension={$soPath}\n";
        file_put_contents("{$packageDir}/crabmagick.ini", $ini);

        $io->write("<info>[crabmagick] Extension binary: {$soPath}</info>");
        $io->write('<info>[crabmagick] To activate — pick one option:</info>');
        $io->write("<info>  1. Scan dir (no sudo): PHP_INI_SCAN_DIR=:{$packageDir} php ...</info>");
        $io->write("<info>  2. Symlink:            sudo ln -s {$packageDir}/crabmagick.ini /etc/php/{$phpVer}/cli/conf.d/30-crabmagick.ini</info>");
        $io->write("<info>  3. Append to php.ini:  echo \"extension={$soPath}\" | sudo tee -a /etc/php/{$phpVer}/cli/php.ini</info>");
    }
}
