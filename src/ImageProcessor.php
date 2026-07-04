<?php

declare(strict_types=1);

namespace Oxipix;

final class ImageProcessor
{
    private ?\Oxipix\Image $inner = null;

    public function __construct(private string $path)
    {
        self::assertExtension();
        $this->inner = new \Oxipix\Image($path);
    }

    public function region(int $x, int $y, int $w, int $h): static
    {
        $this->inner?->region($x, $y, $w, $h);
        return $this;
    }

    public function resize(int $w, int $h = 0): static
    {
        $this->inner?->resize($w, $h);
        return $this;
    }

    public function encode(string $format = 'jpeg', int $quality = 85): string
    {
        return $this->inner?->encode($format, $quality) ?? '';
    }

    public function getInfo(): array
    {
        return $this->inner?->getInfo() ?? [];
    }

    public static function isAvailable(): bool
    {
        return \extension_loaded('oxipix');
    }

    private static function assertExtension(): void
    {
        if (!self::isAvailable()) {
            throw new \RuntimeException('oxipix extension not loaded. Add extension=oxipix.so to php.ini');
        }
    }
}
