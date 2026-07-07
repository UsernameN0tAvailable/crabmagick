<?php

declare(strict_types=1);

namespace CrabMagick;

/**
 * Low-level dispatch layer for CrabMagick.
 *
 * Automatically uses whichever backend is available:
 *   1. Native PHP extension (crabmagick_process / crabmagick_info are defined)
 *      → in-process call, zero socket overhead
 *   2. Unix socket daemon (spawned by bootstrap.php)
 *      → ~0.3 ms per request, zero config required
 *
 * Wire protocol for daemon (little-endian):
 *   Request  PHP → daemon : [u32 json_len][json_bytes]
 *   Response daemon → PHP : [u8 status][u32 payload_len][payload_bytes]
 *   status 0 = success; 1 = error (payload = UTF-8 error string)
 */
final class Runtime
{
    private static ?string $socketPath = null;
    private static bool $usingExtension = false;

    public static function setSocketPath(string $path): void
    {
        self::$socketPath = $path;
    }

    /** Called by bootstrap when the native PHP extension is detected. */
    public static function setUsingExtension(): void
    {
        self::$usingExtension = true;
    }

    public static function isUsingExtension(): bool
    {
        return self::$usingExtension;
    }

    /** True when either the extension is loaded or the daemon socket is live. */
    public static function isReady(): bool
    {
        return self::$usingExtension
            || (self::$socketPath !== null && file_exists(self::$socketPath));
    }

    /**
     * Decode → crop/resize/rotate → encode.
     *
     * @return string Raw encoded image bytes
     * @throws \RuntimeException on error or connection failure
     */
    public static function process(
        string $path,
        int $regionX = 0, int $regionY = 0, int $regionW = 0, int $regionH = 0,
        int $outW = 0, int $outH = 0,
        string $format = 'jpeg', int $quality = 85,
        int $page = 0, int $rotation = 0, bool $square = false,
    ): string {
        if (self::$usingExtension) {
            /** @var string $result */
            $result = \crabmagick_process(
                $path,
                $regionX, $regionY, $regionW, $regionH,
                $outW, $outH,
                $format, $quality,
                $page, $rotation, $square,
            );
            return $result;
        }

        return self::send([
            'cmd'      => 'process',
            'path'     => $path,
            'region_x' => $regionX,
            'region_y' => $regionY,
            'region_w' => $regionW,
            'region_h' => $regionH,
            'out_w'    => $outW,
            'out_h'    => $outH,
            'format'   => $format,
            'quality'  => $quality,
            'page'     => $page,
            'rotation' => $rotation,
            'square'   => $square,
        ]);
    }

    /**
     * Read image dimensions from the file header (fast — no full decode for JXL).
     *
     * @return array{width:int, height:int}
     * @throws \RuntimeException on error or connection failure
     */
    public static function info(string $path, int $page = 0): array
    {
        if (self::$usingExtension) {
            /** @var array{width:int,height:int} $result */
            $result = \crabmagick_info($path, $page);
            return $result;
        }

        $payload = self::send(['cmd' => 'info', 'path' => $path, 'page' => $page]);
        $data    = json_decode($payload, true);
        if (!is_array($data) || !isset($data['width'], $data['height'])) {
            throw new \RuntimeException('[CrabMagick] Malformed info response: ' . $payload);
        }
        return ['width' => (int)$data['width'], 'height' => (int)$data['height']];
    }

    // ── Private ───────────────────────────────────────────────────────────────

    private static function send(array $request): string
    {
        if (self::$socketPath === null) {
            throw new \RuntimeException(
                '[CrabMagick] Daemon socket not initialised. Did you require vendor/autoload.php?'
            );
        }

        $json  = json_encode($request, JSON_THROW_ON_ERROR);
        $frame = pack('V', strlen($json)) . $json;   // u32 LE length-prefix

        $sock = @stream_socket_client('unix://' . self::$socketPath, $errno, $errstr, 2.0);
        if ($sock === false) {
            throw new \RuntimeException(
                "[CrabMagick] Cannot connect to daemon socket: {$errstr} ({$errno})"
            );
        }
        stream_set_timeout($sock, 30);

        fwrite($sock, $frame);

        // Response header: u8 status + u32 payload_len = 5 bytes
        $header = self::readExact($sock, 5);
        $parts  = unpack('Cstatus/Vlen', $header);

        $payloadLen = (int)$parts['len'];
        $payload    = $payloadLen > 0 ? self::readExact($sock, $payloadLen) : '';
        fclose($sock);

        if ((int)$parts['status'] !== 0) {
            throw new \RuntimeException('[CrabMagick] Daemon error: ' . $payload);
        }

        return $payload;
    }

    /** Read exactly $n bytes from $sock, or throw on EOF/timeout. */
    private static function readExact($sock, int $n): string
    {
        $buf       = '';
        $remaining = $n;
        while ($remaining > 0) {
            $chunk = fread($sock, $remaining);
            if ($chunk === false || $chunk === '') {
                throw new \RuntimeException('[CrabMagick] Daemon connection closed unexpectedly');
            }
            $buf       .= $chunk;
            $remaining -= strlen($chunk);
        }
        return $buf;
    }
}
