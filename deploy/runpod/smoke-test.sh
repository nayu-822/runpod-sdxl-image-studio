#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly IMAGE_STUDIO_VENV="${IMAGE_STUDIO_VENV:-/opt/image-studio-venv}"
readonly PYTHON_BIN="${IMAGE_STUDIO_VENV}/bin/python"

fail() {
    printf '%s\n' "[image-studio-smoke] ERROR: $*" >&2
    exit 1
}

[[ -x "$PYTHON_BIN" ]] || fail "Image Studio Python environment is unavailable"
[[ -x /start.sh ]] || fail "RunPod base /start.sh is unavailable"
[[ -f "$PROJECT_ROOT/pyproject.toml" ]] || fail "pyproject.toml is unavailable"
[[ -f "$PROJECT_ROOT/alembic.ini" ]] || fail "alembic.ini is unavailable"
[[ -d "$PROJECT_ROOT/alembic" ]] || fail "alembic directory is unavailable"
[[ -d "$PROJECT_ROOT/workflows" ]] || fail "workflow directory is unavailable"
[[ -x "$PROJECT_ROOT/deploy/runpod/bootstrap.sh" ]] || fail "bootstrap.sh is not executable"

version_output="$($PYTHON_BIN --version 2>&1)"
"$PYTHON_BIN" -c 'import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or newer is required")
' || fail "unsupported Python version: ${version_output}"

"$PYTHON_BIN" -c 'import runpod_sdxl_image_studio
import gradio
major = int(gradio.__version__.split(".", 1)[0])
if major >= 6:
    raise SystemExit("Gradio 6 or newer is not supported")
' || fail "Image Studio import or Gradio version check failed"

command -v rclone >/dev/null 2>&1 || fail "rclone is unavailable"
rclone version >/dev/null 2>&1 || fail "rclone could not report its version"
if ! rclone help flags --name use-json-log | grep -F -- "--use-json-log" >/dev/null; then
    fail "rclone does not support --use-json-log"
fi
if ! rclone help flags --name stats | grep -F -- "--stats Duration" >/dev/null; then
    fail "rclone does not support --stats"
fi
if rclone help flags --name stats | grep -F -- "--stats-one-line-json" >/dev/null; then
    fail "unsupported --stats-one-line-json flag is available instead of the JSON log mode"
fi

temporary_root="$(mktemp -d)" || fail "temporary directory could not be created"
trap 'rm -rf -- "$temporary_root"' EXIT

IMAGE_STUDIO_DATA_DIR="$temporary_root/data" \
IMAGE_STUDIO_DATABASE_URL="sqlite:///$temporary_root/image_studio.sqlite3" \
"$PYTHON_BIN" - "$PROJECT_ROOT" <<'PY' || fail "database migration smoke test failed"
import os
import sqlite3
import sys
from pathlib import Path

from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.db.migration_runner import upgrade_database

project_root = Path(sys.argv[1])
settings = Settings(
    _env_file=None,
    data_dir=Path(os.environ["IMAGE_STUDIO_DATA_DIR"]),
    database_url=os.environ["IMAGE_STUDIO_DATABASE_URL"],
    workflow_dir=project_root / "workflows",
)
upgrade_database(settings, project_root=project_root)
database_path = Path(settings.database_url.removeprefix("sqlite:///"))
with sqlite3.connect(database_path) as connection:
    result = connection.execute("PRAGMA integrity_check").fetchone()
if result != ("ok",):
    raise SystemExit("SQLite integrity check failed")
PY

printf '%s\n' "[image-studio-smoke] checks passed"
