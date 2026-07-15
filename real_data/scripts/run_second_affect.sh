#!/usr/bin/env bash
# One-command runner for the 2nd-affect-encoder control.
# Usage: SP=<cachedir> FILM=<film.mp4> [ENCODER=hsemotion|clip] bash run_second_affect.sh
set -e
: "${SP:?set SP to the cache dir (feat_avg_cache.npz, deriv_sub-*.nii.gz, frames/ or FILM)}"
PY="${PY:-$SP/py39/bin/python}"
export SP ENCODER="${ENCODER:-hsemotion}" FPS="${FPS:-6}" DEVICE="${DEVICE:-cpu}" FILM="${FILM:-}"
DIR="$(cd "$(dirname "$0")" && pwd)"
echo "[1/2] extracting 2nd-affect features ($ENCODER)…"; "$PY" "$DIR/extract_affect2.py"
echo "[2/2] running the swap analysis…";                 "$PY" "$DIR/second_affect_swap.py"
