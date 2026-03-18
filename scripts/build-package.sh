#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${1:-$PROJECT_ROOT/dist}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv is required to build andromeda-trust-lib." >&2
    echo "[ERROR] Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"
uv build --out-dir "$OUTPUT_DIR"

echo "Built andromeda-trust-lib artifacts in: $OUTPUT_DIR"
ls -1 "$OUTPUT_DIR"/andromeda_trust_lib-*
