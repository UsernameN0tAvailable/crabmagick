<?php

declare(strict_types=1);

namespace Crabmagick;

/**
 * One-shot decode → crop/resize/rotate → encode.
 *
 * @param string $format 'jpeg' | 'webp' | 'png' | 'jxl' | 'avif'
 * @return string Raw encoded image bytes
 */
function process(
    string $path,
    int $regionX = 0, int $regionY = 0, int $regionW = 0, int $regionH = 0,
    int $outW = 0, int $outH = 0,
    string $format = 'jpeg', int $quality = 85,
): string {
    return Runtime::process($path, $regionX, $regionY, $regionW, $regionH, $outW, $outH, $format, $quality);
}

/**
 * Read image dimensions without full decode.
 *
 * @return array{width:int, height:int}
 */
function info(string $path, int $page = 0): array
{
    return Runtime::info($path, $page);
}
