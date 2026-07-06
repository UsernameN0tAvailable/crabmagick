#!/usr/bin/env bash
set -euo pipefail

args=()
for arg in "$@"; do
    case "$arg" in
        -lgcc_s)
            args+=("-Wl,--push-state,-Bstatic" "-lgcc_eh" "-lgcc" "-Wl,--pop-state")
            ;;
        -static-libm)
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

exec cc "${args[@]}"
