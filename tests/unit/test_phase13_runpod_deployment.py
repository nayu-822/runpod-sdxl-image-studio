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
    try:
        return subprocess.run(
            [_bash(), "-c", script, "phase13-test", *arguments],
            cwd=REPOSITORY_ROOT,
            env=merged_environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError("Phase 13 bash test exceeded its 10-second timeout") from exc


def test_dockerfile_is_pinned_and_keeps_the_base_entrypoint() -> None:
    dockerfile = _read(DOCKERFILE)

    assert "FROM runpod/comfyui:1.4.4-cuda12.8" in dockerfile
    assert "ARG IMAGE_STUDIO_RCLONE_VERSION=1.74.2" in dockerfile
    assert "ARG RCLONE_VERSION" not in dockerfile
    assert "${RCLONE_VERSION}" not in dockerfile
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
    assert "--start-period=30m" in dockerfile


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
        "IMAGE_STUDIO_BOOTSTRAP_COMFYUI_MONITOR_INTERVAL_SECONDS=5",
        "IMAGE_STUDIO_BOOTSTRAP_COMFYUI_FAILURE_THRESHOLD=12",
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
    assert "bootstrap_now_seconds" in bootstrap
    assert "deadline_seconds" in bootstrap
    assert '--max-time "$max_time_seconds"' in bootstrap
    assert "probe_comfyui" in bootstrap
    assert "check_comfyui_liveness" in bootstrap
    assert "IMAGE_STUDIO_BOOTSTRAP_COMFYUI_MONITOR_INTERVAL_SECONDS" in bootstrap
    assert "IMAGE_STUDIO_BOOTSTRAP_COMFYUI_FAILURE_THRESHOLD" in bootstrap
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
    if secret:
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


@pytest.mark.skipif(os.name == "nt", reason="runtime bootstrap supervision is tested on Linux")
def test_bootstrap_retries_comfyui_readiness_before_starting_image_studio() -> None:
    result = _run_bash(
        """
set -Eeuo pipefail
root="$(mktemp -d)"
trap 'rm -rf -- "$root"' EXIT
mkdir -p "$root/venv/bin"
printf '%s\n' '#!/usr/bin/env bash' 'exit 0' > "$root/venv/bin/runpod-sdxl-image-studio"
chmod +x "$root/venv/bin/runpod-sdxl-image-studio"
export IMAGE_STUDIO_ROOT="$root"
export IMAGE_STUDIO_VENV="$root/venv"
source "$1"
BASE_PROCESS_PID=123
attempts=0
kill() {
    if [[ "$1" == "-0" && "$2" == "123" ]]; then
        return 0
    fi
    return 1
}
sleep() { :; }
probe_comfyui() {
    attempts=$((attempts + 1))
    if ((attempts < 3)); then
        return 1
    fi
    return 0
}
wait_for_comfyui
start_image_studio
wait "$IMAGE_STUDIO_PROCESS_PID"
printf 'attempts=%s app=started\n' "$attempts"
""",
        "deploy/runpod/bootstrap.sh",
    )

    assert result.returncode == 0, result.stderr
    assert "attempts=3 app=started" in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="runtime bootstrap supervision is tested on Linux")
def test_bootstrap_comfyui_timeout_stops_base_before_image_studio_start() -> None:
    result = _run_bash(
        """
set -Eeuo pipefail
source "$1"
export IMAGE_STUDIO_BOOTSTRAP_COMFYUI_TIMEOUT_SECONDS=2
export IMAGE_STUDIO_BOOTSTRAP_SHUTDOWN_GRACE_SECONDS=0
BASE_PROCESS_PID=123
fake_now=100
bootstrap_now_seconds() { printf '%s\n' "$fake_now"; }
kill() {
    if [[ "$1" == "-0" && "$2" == "123" ]]; then
        return 0
    fi
    if [[ "$1" == "-TERM" || "$1" == "-KILL" ]]; then
        printf '%s:%s\n' "$1" "$2"
        return 0
    fi
    return 1
}
sleep() { :; }
probe_comfyui() {
    fake_now=$((fake_now + 1))
    return 1
}
if wait_for_comfyui; then
    printf 'app-started\n'
    exit 2
fi
""",
        "deploy/runpod/bootstrap.sh",
    )

    assert result.returncode != 0
    assert "-TERM:123" in result.stdout
    assert "-KILL:123" in result.stdout
    assert "app-started" not in result.stdout
    assert "ComfyUI readiness timed out" in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="runtime bootstrap supervision is tested on Linux")
def test_bootstrap_startup_timeout_uses_wall_clock_and_bounds_probe_timeout() -> None:
    result = _run_bash(
        """
set -Eeuo pipefail
source "$1"
export IMAGE_STUDIO_BOOTSTRAP_COMFYUI_TIMEOUT_SECONDS=4
export IMAGE_STUDIO_BOOTSTRAP_SHUTDOWN_GRACE_SECONDS=0
BASE_PROCESS_PID=123
fake_now=100
probe_calls=0
probe_timeout=0
sleep_calls=0
bootstrap_now_seconds() { printf '%s\n' "$fake_now"; }
report_counts() {
    printf 'probes=%s timeout=%s sleeps=%s now=%s\n' \
        "$probe_calls" "$probe_timeout" "$sleep_calls" "$fake_now"
}
trap report_counts EXIT
kill() {
    if [[ "$1" == "-0" && "$2" == "123" ]]; then
        return 0
    fi
    if [[ "$1" == "-TERM" || "$1" == "-KILL" ]]; then
        return 0
    fi
    return 1
}
sleep() {
    sleep_calls=$((sleep_calls + 1))
    fake_now=$((fake_now + 1))
}
probe_comfyui() {
    probe_calls=$((probe_calls + 1))
    probe_timeout="$1"
    printf 'probe_timeout=%s\n' "$probe_timeout"
    fake_now=$((fake_now + 5))
    return 1
}
wait_for_comfyui
""",
        "deploy/runpod/bootstrap.sh",
    )

    assert result.returncode != 0
    assert "probe_timeout=4" in result.stdout
    assert "probes=1 timeout=4 sleeps=0 now=105" in result.stdout
    assert "ComfyUI readiness timed out" in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="runtime bootstrap supervision is tested on Linux")
def test_bootstrap_timeout_zero_allows_one_bounded_readiness_probe() -> None:
    result = _run_bash(
        """
set -Eeuo pipefail
source "$1"
export IMAGE_STUDIO_BOOTSTRAP_COMFYUI_TIMEOUT_SECONDS=0
export IMAGE_STUDIO_BOOTSTRAP_SHUTDOWN_GRACE_SECONDS=0
BASE_PROCESS_PID=123
fake_now=100
probe_calls=0
probe_timeout=0
bootstrap_now_seconds() { printf '%s\n' "$fake_now"; }
report_counts() {
    printf 'probes=%s timeout=%s sleeps=%s\n' \
        "$probe_calls" "$probe_timeout" "$sleep_calls"
}
trap report_counts EXIT
kill() {
    if [[ "$1" == "-0" && "$2" == "123" ]]; then
        return 0
    fi
    if [[ "$1" == "-TERM" || "$1" == "-KILL" ]]; then
        return 0
    fi
    return 1
}
sleep_calls=0
sleep() { sleep_calls=$((sleep_calls + 1)); }
probe_comfyui() {
    probe_calls=$((probe_calls + 1))
    probe_timeout="$1"
    printf 'probe_timeout=%s\n' "$probe_timeout"
    return 1
}
wait_for_comfyui
""",
        "deploy/runpod/bootstrap.sh",
    )

    assert result.returncode != 0
    assert "probe_timeout=1" in result.stdout
    assert "probes=1 timeout=1" in result.stdout
    assert "sleeps=0" in result.stdout
    assert "ComfyUI readiness timed out" in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="runtime bootstrap supervision is tested on Linux")
def test_temporary_comfyui_probe_failure_is_tolerated_and_success_resets_count() -> None:
    result = _run_bash(
        """
set -Eeuo pipefail
source "$1"
export IMAGE_STUDIO_BOOTSTRAP_COMFYUI_FAILURE_THRESHOLD=3
attempts=0
probe_comfyui() {
    attempts=$((attempts + 1))
    if ((attempts == 1)); then
        return 1
    fi
    return 0
}
check_comfyui_liveness
[[ "$COMFYUI_FAILURE_COUNT" == "1" ]]
check_comfyui_liveness
printf 'attempts=%s failures=%s\n' "$attempts" "$COMFYUI_FAILURE_COUNT"
""",
        "deploy/runpod/bootstrap.sh",
    )

    assert result.returncode == 0, result.stderr
    assert "attempts=2 failures=0" in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="runtime bootstrap supervision is tested on Linux")
def test_persistent_comfyui_probe_failure_stops_image_studio_then_base() -> None:
    result = _run_bash(
        """
set -Eeuo pipefail
source "$1"
export IMAGE_STUDIO_BOOTSTRAP_COMFYUI_MONITOR_INTERVAL_SECONDS=1
export IMAGE_STUDIO_BOOTSTRAP_COMFYUI_FAILURE_THRESHOLD=2
export IMAGE_STUDIO_BOOTSTRAP_SHUTDOWN_GRACE_SECONDS=0
IMAGE_STUDIO_PROCESS_PID=401
BASE_PROCESS_PID=402
app_alive=1
base_alive=1
probe_calls=0
events=""
kill() {
    if [[ "$1" == "-0" ]]; then
        if [[ "$2" == "401" ]]; then
            ((app_alive == 1))
            return
        fi
        if [[ "$2" == "402" ]]; then
            ((base_alive == 1))
            return
        fi
        return 1
    fi
    events="${events}${1}:${2};"
    if [[ "$2" == "401" ]]; then app_alive=0; fi
    if [[ "$2" == "402" ]]; then base_alive=0; fi
    return 0
}
sleep() { :; }
probe_comfyui() {
    probe_calls=$((probe_calls + 1))
    return 1
}
status=0
monitor_processes || status=$?
printf 'probes=%s events=%s\n' "$probe_calls" "$events"
exit "$status"
""",
        "deploy/runpod/bootstrap.sh",
    )

    assert result.returncode != 0
    assert "probes=2 events=-TERM:401;-KILL:401;-TERM:402;-KILL:402;" in result.stdout
    assert "ComfyUI failed its liveness probe persistently" in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="runtime bootstrap supervision is tested on Linux")
@pytest.mark.parametrize("dead_process", ["image", "base"])
def test_unexpected_child_exit_stops_the_sibling_and_returns_nonzero(dead_process: str) -> None:
    result = _run_bash(
        """
set -Eeuo pipefail
source "$1"
export IMAGE_STUDIO_BOOTSTRAP_COMFYUI_MONITOR_INTERVAL_SECONDS=1
export IMAGE_STUDIO_BOOTSTRAP_COMFYUI_FAILURE_THRESHOLD=2
export IMAGE_STUDIO_BOOTSTRAP_SHUTDOWN_GRACE_SECONDS=0
IMAGE_STUDIO_PROCESS_PID=501
BASE_PROCESS_PID=502
app_alive=1
base_alive=1
events=""
if [[ "$2" == "image" ]]; then app_alive=0; else base_alive=0; fi
kill() {
    if [[ "$1" == "-0" ]]; then
        if [[ "$2" == "501" ]]; then
            ((app_alive == 1))
            return
        fi
        if [[ "$2" == "502" ]]; then
            ((base_alive == 1))
            return
        fi
        return 1
    fi
    events="${events}${1}:${2};"
    return 0
}
wait() { return 17; }
sleep() { :; }
probe_comfyui() { return 0; }
status=0
monitor_processes || status=$?
printf 'events=%s\n' "$events"
exit "$status"
""",
        "deploy/runpod/bootstrap.sh",
        dead_process,
    )

    assert result.returncode != 0
    assert "exited unexpectedly" in result.stderr
    if dead_process == "image":
        assert "events=-TERM:502;-KILL:502;" in result.stdout
    else:
        assert "events=-TERM:501;-KILL:501;" in result.stdout
