<?php

declare(strict_types=1);

namespace Crabmagick;

/**
 * Holds the FFI instance and exposes static helpers used by Image and the
 * top-level \Crabmagick\process() / \Crabmagick\info() functions.
 *
 * Populated by bootstrap.php at autoload time.
 */
final class Runtime
{
    private static ?\FFI $ffi = null;

    /** C declarations passed to FFI::cdef(). Must match ffi/oxipix.h exactly. */
    private const DECLS = <<<'C'
        typedef struct {
            uint32_t region_x;
            uint32_t region_y;
            uint32_t region_w;
            uint32_t region_h;
            uint32_t out_w;
            uint32_t out_h;
            uint8_t  quality;
            int      format;
            uint32_t page;
            uint16_t rotation;
            uint8_t  square_region;
        } oxipix_request;

        typedef struct {
            uint32_t width;
            uint32_t height;
        } oxipix_image_info;

        int   oxipix_get_info(const char *path, oxipix_image_info *info, char **error_message);
        int   oxipix_process(const char *path, const oxipix_request *request, uint8_t **out_data, size_t *out_len, char **error_message);
        void  oxipix_free(void *ptr);
    C;

    private const FORMAT_MAP = [
        'jpg' => 0,
        'jpeg' => 0,
        'webp' => 1,
        'png' => 2,
        'jxl' => 3,
        'avif' => 4,
    ];

    public static function load(string $soPath): void
    {
        if (self::$ffi !== null) {
            return;
        }

        $ffi = \FFI::cdef(self::DECLS, $soPath);
        self::$ffi = $ffi;
    }

    public static function isLoaded(): bool
    {
        return self::$ffi !== null;
    }

    /** @return array{width:int, height:int} */
    public static function info(string $path): array
    {
        $ffi = self::require();
        $info = $ffi->new('oxipix_image_info');
        $err = $ffi->new('char*');

        $rc = $ffi->oxipix_get_info($path, \FFI::addr($info), \FFI::addr($err));
        if ($rc !== 0) {
            throw new \RuntimeException('[crabmagick] ' . self::takeError($ffi, $err));
        }

        return ['width' => (int) $info->width, 'height' => (int) $info->height];
    }

    public static function process(
        string $path,
        int $regionX,
        int $regionY,
        int $regionW,
        int $regionH,
        int $outW,
        int $outH,
        string $format,
        int $quality,
        int $page = 0,
        int $rotation = 0,
        bool $squareRegion = false,
    ): string {
        $ffi = self::require();

        $req = $ffi->new('oxipix_request');
        $req->region_x = $regionX;
        $req->region_y = $regionY;
        $req->region_w = $regionW;
        $req->region_h = $regionH;
        $req->out_w = $outW;
        $req->out_h = $outH;
        $req->quality = $quality;
        $req->format = self::formatCode($format);
        $req->page = $page;
        $req->rotation = $rotation;
        $req->square_region = $squareRegion ? 1 : 0;

        $outPtr = $ffi->new('uint8_t*');
        $outLen = $ffi->new('size_t');
        $err = $ffi->new('char*');

        $rc = $ffi->oxipix_process(
            $path,
            \FFI::addr($req),
            \FFI::addr($outPtr),
            \FFI::addr($outLen),
            \FFI::addr($err),
        );

        if ($rc !== 0) {
            throw new \RuntimeException('[crabmagick] ' . self::takeError($ffi, $err));
        }

        $bytes = \FFI::string(\FFI::cast('char*', $outPtr), (int) $outLen->cdata);
        $ffi->oxipix_free(\FFI::cast('void*', $outPtr));

        return $bytes;
    }

    private static function require(): \FFI
    {
        if (self::$ffi === null) {
            throw new \RuntimeException(
                '[crabmagick] Native library not loaded. '
                . 'Ensure ffi.enable=true and that PHP_INI_SCAN_DIR includes the crabmagick package directory.'
            );
        }

        return self::$ffi;
    }

    private static function formatCode(string $format): int
    {
        $key = strtolower($format);
        if (!isset(self::FORMAT_MAP[$key])) {
            throw new \InvalidArgumentException("[crabmagick] Unknown format: {$format}");
        }

        return self::FORMAT_MAP[$key];
    }

    private static function takeError(\FFI $ffi, \FFI\CData $err): string
    {
        if (\FFI::isNull($err)) {
            return 'unknown error';
        }

        $msg = \FFI::string($err);
        $ffi->oxipix_free(\FFI::cast('void*', $err));

        return $msg;
    }
}
