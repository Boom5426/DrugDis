#!/usr/bin/env bash
# Rebuild the frozen split manifests from the substrate config.
#   bash scripts/make_manifests.sh [output_dir]      (default: $DRUGDIS_WORK/manifests)
# make_manifests.py reads the data paths in the config as written; the config
# stores them relative to DRUGDIS_DATA, so it runs from there. The stamp is the
# freeze date recorded in manifests/MANIFEST.json.
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUTDIR="$(mkdir -p "${1:-$WORK/manifests}" && cd "${1:-$WORK/manifests}" && pwd)"
cd "$DRUGDIS_DATA"
$PY "$REPO/drugdis/splits/make_manifests.py" --config "$CFG" --out "$OUTDIR" \
    --stamp 2026-09-05T00:00:00Z
