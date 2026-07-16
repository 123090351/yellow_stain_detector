#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/data/greya/yellow_stain_detector}"
DATA_DIR="${CODE_SERVER_DATA_DIR:-/data/greya/.code-server}"

if [[ -z "${PASSWORD:-}" ]]; then
  echo "PASSWORD environment variable is required" >&2
  exit 1
fi

mkdir -p "$WORKSPACE_DIR" "$DATA_DIR/user-data" "$DATA_DIR/extensions"

exec /opt/code-server/bin/code-server "$WORKSPACE_DIR" \
  --bind-addr "0.0.0.0:${CODE_SERVER_PORT:-8080}" \
  --auth password \
  --disable-telemetry \
  --user-data-dir "$DATA_DIR/user-data" \
  --extensions-dir "$DATA_DIR/extensions"
