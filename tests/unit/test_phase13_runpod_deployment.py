from __future__ import annotations

import base64
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPOSITORY_ROOT / "Dockerfile"
DOCKERIGNORE = REPOSITORY_ROOT / ".dockerignore"
BOOTSTRAP = REPOSITORY_ROOT / "deploy" / "runpod" / "bootstrap.sh"
SMOKE_TEST = REPOSITORY_ROOT / "deploy" / "runpod" / "smoke-test.sh"
TEMPLATE = REPOSITORY_ROOT / "deploy" / "runpod" / "template.env.example"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _bash() -> str:
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("bash is required for RunPod deployment shell tests")
    return executable


def _run_bash(
    script: str, *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    merged_environment = os.environ.copy()
    if env is not None:
        merged_environment.update(env)
    return subprocess.run(
        [_bash(), "-c", script, "phase13-test", *arguments],
        cwd=REPOSITORY_ROOT,
        env=merged_environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_dockerfile_is_pinned_and_keeps_the_base_entrypoint() -> None:
    dockerfile = _read(DOCKERFILE)

    assert "FROM runpod/comfyui:1.4.4-cuda12.8" in dockerfile
    assert "runpod/comfyui:latest" not in dockerfile
    assert "ENTRYPOINT" not in dockerfile
    assert 'CMD ["/opt/image-studio/deploy/runpod/bootstrap.sh"]' in dockerfile
    assert 'python3.12 -m venv "$IMAGE_STUDIO_VENV"' in dockerfile
    assert 'pip" install --no-cache-dir -e "$IMAGE_STUDIO_ROOT"' in dockerfile
    assert "SHA256SUMS" in dockerfile
    assert "sha256sum --check --status" in dockerfile
    assert "EXPOSE 7860" in dockerfile
    assert "EXPOSE 8188" not in dockerfile
    assert "COPY *.safetensors" not in dockerfile
    assert "COPY . " not in dockerfile
    assert "http://127.0.0.1:7860/" in dockerfile


def test_dockerignore_excludes_runtime_state_and_models_but_keeps_examples() -> None:
    dockerignore = _read(DOCKERIGNORE)

    for entry in (
        ".git",
        ".env",
        "*.sqlite3",
        "rclone.conf",
        ".venv",
        "*.safetensors",
        "*.ckpt",
        "data",
        "outputs",
    ):
        assert entry in dockerignore
    assert "!.env.example" in dockerignore
    assert "deploy/runpod/template.env.example" not in dockerignore


def test_template_uses_local_comfyui_and_only_exposes_the_app_port() -> None:
    template = _read(TEMPLATE)

    required_lines = (
        "IMAGE_STUDIO_ENV=production",
        "IMAGE_STUDIO_HOST=0.0.0.0",
        "IMAGE_STUDIO_PORT=7860",
        "COMFYUI_BASE_URL=http://127.0.0.1:8188",
        "COMFYUI_WS_URL=ws://127.0.0.1:8188/ws",
        "COMFYUI_OUTPUT_DIR=/workspace/runpod-slim/ComfyUI/output",
        "IMAGE_STUDIO_WORKFLOW_DIR=/opt/image-studio/workflows",
        "RCLONE_CONFIG=/run/image-studio/rclone.conf",
        "IMAGE_STUDIO_RCLONE_CONFIG_B64={{ RUNPOD_SECRET_image_studio_rclone_config_b64 }}",
        "IMAGE_STUDIO_AUTO_TERMINATE_ENABLED=false",
        "IMAGE_STUDIO_BOOTSTRAP_COMFYUI_TIMEOUT_SECONDS=900",
        "IMAGE_STUDIO_BOOTSTRAP_SHUTDOWN_GRACE_SECONDS=30",
    )
    for line in required_lines:
        assert line in template
    assert "RUNPOD_API_KEY" not in template
    assert "RUNPOD_POD_ID" not in template


def test_bootstrap_and_smoke_scripts_are_syntactically_valid_and_safe() -> None:
    if os.name != "nt":
        for script in (BOOTSTRAP, SMOKE_TEST):
            result = subprocess.run(
                [_bash(), "-n", str(script.relative_to(REPOSITORY_ROOT))],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stderr

    bootstrap = _read(BOOTSTRAP)
    assert "http://127.0.0.1:8188/system_stats" in bootstrap
    assert "kill -TERM" in bootstrap
    assert "shutdown_process" in bootstrap
    assert "mktemp" in bootstrap
    assert "chmod 0600" in bootstrap
    assert "mv -f" in bootstrap
    assert "rclone listremotes" in bootstrap
    for forbidden in (
        "curl | bash",
        "git clone",
        "pip install",
        "rclone sync",
        "rclone config show",
        'cat "$RCLONE_CONFIG"',
        'echo "$IMAGE_STUDIO_RCLONE_CONFIG_B64"',
    ):
        assert forbidden not in bootstrap


@pytest.mark.skipif(os.name == "nt", reason="runtime bash materialization is tested on Linux")
def test_valid_rclone_secret_is_atomically_materialized_with_private_mode() -> None:
    encoded = base64.b64encode(b"[drive]\ntype = drive\n").decode("ascii")
    result = _run_bash(
        """
set -Eeuo pipefail
root="$(mktemp -d)"
trap 'rm -rf -- "$root"' EXIT
export RCLONE_CONFIG="$root/rclone.conf"
export IMAGE_STUDIO_RCLONE_CONFIG_B64
source "$1"
materialize_rclone_config
printf 'mode=%s\\n' "$(stat -c '%a' "$RCLONE_CONFIG")"
printf 'content=%s\\n' "$(base64 -w0 "$RCLONE_CONFIG")"
""",
        "deploy/runpod/bootstrap.sh",
        env={"IMAGE_STUDIO_RCLONE_CONFIG_B64": encoded},
    )

    assert result.returncode == 0, result.stderr
    assert "mode=600" in result.stdout
    assert f"content={encoded}" in result.stdout
    assert encoded not in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="runtime bash materialization is tested on Linux")
@pytest.mark.parametrize("secret", ["", "not-base64!"])
def test_invalid_rclone_secret_fails_without_materializing_config(secret: str) -> None:
    result = _run_bash(
        """
set -Eeuo pipefail
root="$(mktemp -d)"
trap 'rm -rf -- "$root"' EXIT
export RCLONE_CONFIG="$root/rclone.conf"
export IMAGE_STUDIO_RCLONE_CONFIG_B64="$2"
source "$1"
if materialize_rclone_config; then
    exit 2
fi
""",
        "deploy/runpod/bootstrap.sh",
        secret,
    )

    assert result.returncode != 0
    assert "rclone config secret" in result.stderr
    assert secret not in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="runtime bash materialization is tested on Linux")
def test_existing_rclone_config_symlink_is_rejected() -> None:
    encoded = base64.b64encode(b"[drive]\ntype = drive\n").decode("ascii")
    result = _run_bash(
        """
set -Eeuo pipefail
root="$(mktemp -d)"
trap 'rm -rf -- "$root"' EXIT
touch "$root/target"
ln -s "$root/target" "$root/rclone.conf"
export RCLONE_CONFIG="$root/rclone.conf"
export IMAGE_STUDIO_RCLONE_CONFIG_B64="$2"
source "$1"
if materialize_rclone_config; then
    exit 2
fi
""",
        "deploy/runpod/bootstrap.sh",
        encoded,
    )

    assert result.returncode != 0
    assert "must not be a symlink" in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="runtime bash materialization is tested on Linux")
def test_rclone_config_write_failure_is_fail_fast() -> None:
    encoded = base64.b64encode(b"[drive]\ntype = drive\n").decode("ascii")
    result = _run_bash(
        """
set -Eeuo pipefail
root="$(mktemp -d)"
trap 'rm -rf -- "$root"' EXIT
printf '%s' 'not a directory' > "$root/not-a-directory"
export RCLONE_CONFIG="$root/not-a-directory/rclone.conf"
export IMAGE_STUDIO_RCLONE_CONFIG_B64="$2"
source "$1"
if materialize_rclone_config; then
    exit 2
fi
""",
        "deploy/runpod/bootstrap.sh",
        encoded,
    )

    assert result.returncode != 0
    assert "rclone config" in result.stderr
