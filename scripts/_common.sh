# Sourced by every launcher: repository root, interpreter and output roots.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
RUNS="${DRUGDIS_RUNS:-$REPO/runs}"
WORK="${DRUGDIS_WORK:-$REPO/work}"
# The table builders check that a run was trained with the same config file and
# manifest directory they are given (the paths recorded in its results.json).
# DRUGDIS_CONFIG / DRUGDIS_MANIFESTS point them at another copy of the frozen
# config and manifests, e.g. the one a set of existing runs was trained with.
CFG="${DRUGDIS_CONFIG:-$REPO/configs/substrate_config.frozen.json}"
MAN="${DRUGDIS_MANIFESTS:-$REPO/manifests}"
: "${DRUGDIS_DATA:?set DRUGDIS_DATA to the processed data directory (Hugging Face dataset Boom5426/DrugDis)}"
