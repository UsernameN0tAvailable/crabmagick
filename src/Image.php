<?php

declare(strict_types=1);

namespace Crabmagick;

/**
 * Fluent builder for image processing via the crabmagick native library.
 *
 * Loaded at runtime through PHP FFI — no extension= in php.ini required.
 * Activate by pointing PHP_INI_SCAN_DIR at this package directory, which
 * contains a crabmagick.ini that sets ffi.enable=true.
 */
class Image
{
    private int $regionX = 0;
    private int $regionY = 0;
    private int $regionW = 0;
    private int $regionH = 0;
    private int $outW = 0;
    private int $outH = 0;
    private int $page = 0;
    private int $rotation = 0;
    private bool $squareRegion = false;

    public function __construct(private readonly string $path) {}

    public function region(int $x, int $y, int $w, int $h): static
    {
        $this->regionX = $x;
        $this->regionY = $y;
        $this->regionW = $w;
        $this->regionH = $h;
        return $this;
    }

    public function resize(int $w, int $h = 0): static
    {
        $this->outW = $w;
        $this->outH = $h;
        return $this;
    }

    public function page(int $page): static
    {
        $this->page = $page;
        return $this;
    }

    public function rotate(int $degrees): static
    {
        $this->rotation = $degrees;
        return $this;
    }

    public function square(): static
    {
        $this->squareRegion = true;
        return $this;
    }

    /**
     * Encode and return the processed image as a raw byte string.
     *
     * @param string $format 'jpeg'|'webp'|'png'|'jxl'|'avif'
     * @param int    $quality 0–100
     */
    public function encode(string $format = 'jpeg', int $quality = 85): string
    {
        return Runtime::process(
            $this->path,
            $this->regionX,
            $this->regionY,
            $this->regionW,
            $this->regionH,
            $this->outW,
            $this->outH,
            $format,
            $quality,
            $this->page,
            $this->rotation,
            $this->squareRegion,
        );
    }

    /** @return array{width:int, height:int} */
    public function getInfo(): array
    {
        return Runtime::info($this->path);
    }

    /** Returns true if the crabmagick native library is loaded and ready. */
    public static function isAvailable(): bool
    {
        return Runtime::isLoaded();
    }
}
