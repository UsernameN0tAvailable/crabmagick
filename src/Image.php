<?php

declare(strict_types=1);

namespace CrabMagick;

/**
 * Fluent builder for image processing via the CrabMagick daemon.
 *
 * The daemon is started automatically when this package is loaded via
 * `require vendor/autoload.php`. No php.ini changes required.
 *
 * Example:
 *   $bytes = (new \CrabMagick\Image('/path/to/file.jxl'))
 *       ->region(0, 0, 512, 512)
 *       ->resize(256, 256)
 *       ->encode('jpeg', 85);
 */
class Image
{
    private int  $regionX      = 0;
    private int  $regionY      = 0;
    private int  $regionW      = 0;
    private int  $regionH      = 0;
    private int  $outW         = 0;
    private int  $outH         = 0;
    private int  $page         = 0;
    private int  $rotation     = 0;
    private bool $squareRegion = false;

    public function __construct(private readonly string $path) {}

    /** Set a rectangular source region (x, y, w, h). */
    public function region(int $x, int $y, int $w, int $h): static
    {
        $this->regionX = $x;
        $this->regionY = $y;
        $this->regionW = $w;
        $this->regionH = $h;
        return $this;
    }

    /** Crop the largest centred square (IIIF "square" region). */
    public function square(): static
    {
        $this->squareRegion = true;
        return $this;
    }

    /** Set the output dimensions. Pass 0 to derive proportionally. */
    public function resize(int $w, int $h = 0): static
    {
        $this->outW = $w;
        $this->outH = $h;
        return $this;
    }

    /** Select a page/frame (multi-page TIFF, PDF, animated WebP). */
    public function page(int $page): static
    {
        $this->page = $page;
        return $this;
    }

    /** Rotate clockwise: 90, 180, or 270 degrees. */
    public function rotate(int $degrees): static
    {
        $this->rotation = $degrees;
        return $this;
    }

    /**
     * Encode and return the processed image as a raw byte string.
     *
     * @param string $format  'jpeg' | 'webp' | 'png' | 'jxl' | 'avif'
     * @param int    $quality 0–100
     * @return string         Raw encoded image bytes
     */
    public function encode(string $format = 'jpeg', int $quality = 85): string
    {
        return Runtime::process(
            $this->path,
            $this->regionX, $this->regionY, $this->regionW, $this->regionH,
            $this->outW, $this->outH,
            $format, $quality,
            $this->page, $this->rotation, $this->squareRegion,
        );
    }

    /**
     * Read image dimensions without full decode.
     *
     * @return array{width:int, height:int}
     */
    public function getInfo(): array
    {
        return Runtime::info($this->path, $this->page);
    }

    /** Returns true if the CrabMagick daemon is running and reachable. */
    public static function isAvailable(): bool
    {
        return Runtime::isReady();
    }

    /**
     * Losslessly repackage this JPEG into a JXL container without pixel decode/re-encode.
     *
     * Equivalent to `cjxl --lossless_jpeg=1`. The source JPEG's DCT coefficients are
     * preserved verbatim — zero quality loss. Typical file size reduction: 15–30%.
     * The original JPEG bytes can be recovered from the output with `djxl --pixels_to_jpeg`.
     *
     * @return string Raw JXL bytes
     * @throws \RuntimeException if the source file is not a valid JPEG
     */
    public function transcodeToJxl(): string
    {
        return Runtime::transcodeJpeg($this->path);
    }
}
