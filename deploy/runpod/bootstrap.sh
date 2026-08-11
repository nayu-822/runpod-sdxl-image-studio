#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

readonly IMAGE_STUDIO_ROOT="${IMAGE_STUDIO_ROOT:-/opt/image-studio}"
readonly IMAGE_STUDIO_VENV="${IMAGE_STUDIO_VENV:-/opt/image-studio-venv}"
readonly COMFYUI_READY_URL="http://127.0.0.1:8188/system_stats"
readonly IMAGE_STUDIO_APP="${IMAGE_STUDIO_VENV}/bin/runpod-sdxl-image-studio"

BASE_PROCESS_PID=""
IMAGE_STUDIO_PROCESS_PID=""
SHUTDOWN_REQUESTED=0
COMFYUI_FAILURE_COUNT=0

log_message() {
    printf '%s\n' "[image-studio-bootstrap] $*" >&2
}

is_true() {
    case "${1:-}" in
        true|TRUE|1|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

shutdown_process() {
    local pid="${1:-}"
    local process_name="$2"
    local grace_seconds="${IMAGE_STUDIO_BOOTSTRAP_SHUTDOWN_GRACE_SECONDS:-30}"

    [[ -n "$pid" ]] || return 0
    if ! [[ "$grace_seconds" =~ ^[0-9]+$ ]]; then
        grace_seconds=30
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    kill -TERM "$pid" 2>/dev/null || true
    for ((second = 0; second < grace_seconds; second++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done

    log_message "$process_name did not stop during the shutdown grace period"
    kill -KILL "$pid" 2>/dev/null || true
}

fatal() {
    log_message "ERROR: $*"
    shutdown_process "$IMAGE_STUDIO_PROCESS_PID" "Image Studio" || true
    shutdown_process "$BASE_PROCESS_PID" "RunPod base process" || true
    exit 1
}

materialize_rclone_config() {
    local config_path="${RCLONE_CONFIG:-/run/image-studio/rclone.conf}"
    local config_parent
    local temporary_path

    [[ "$config_path" == /* ]] || fatal "RCLONE_CONFIG must be an absolute path"
    config_parent="$(dirname -- "$config_path")"
    if [[ -L "$config_parent" ]]; then
        fatal "rclone config parent must not be a symlink"
    fi
    mkdir -p -- "$config_parent" || fatal "rclone config directory could not be created"
    chmod 0700 -- "$config_parent" || fatal "rclone config directory permissions failed"

    if [[ -L "$config_path" ]]; then
        fatal "rclone config path must not be a symlink"
    fi

    if [[ -v IMAGE_STUDIO_RCLONE_CONFIG_B64 ]]; then
        [[ -n "${IMAGE_STUDIO_RCLONE_CONFIG_B64}" ]] || fatal "rclone config secret is empty"
        temporary_path="$(mktemp "${config_parent}/.rclone.conf.XXXXXX")" \
            || fatal "temporary rclone config could not be created"
        if ! printf '%s' "$IMAGE_STUDIO_RCLONE_CONFIG_B64" \
            | python3 -c 'import base64, sys
raw = b"".join(sys.stdin.buffer.read().split())
if not raw:
    raise SystemExit("empty base64 input")
try:
    decoded = base64.b64decode(raw, validate=True)
except Exception as exc:
    raise SystemExit("invalid base64 input") from exc
if not decoded:
    raise SystemExit("empty decoded config")
sys.stdout.buffer.write(decoded)
' > "$temporary_path"; then
            rm -f -- "$temporary_path"
            fatal "rclone config secret could not be decoded"
        fi
        chmod 0600 -- "$temporary_path" || {
            rm -f -- "$temporary_path"
            fatal "rclone config permissions failed"
        }
        if [[ -L "$config_path" ]]; then
            rm -f -- "$temporary_path"
            fatal "rclone config path became a symlink"
        fi
        mv -f -- "$temporary_path" "$config_path" || {
            rm -f -- "$temporary_path"
            fatal "rclone config could not be installed"
        }
        chmod 0600 -- "$config_path" || fatal "rclone config permissions failed"
        unset IMAGE_STUDIO_RCLONE_CONFIG_B64
    fi

    export RCLONE_CONFIG="$config_path"
}

remote_configuration_required() {
    is_true "${IMAGE_STUDIO_STATE_SYNC_ENABLED:-false}" \
        || is_true "${IMAGE_STUDIO_REMOTE_MODEL_ENABLED:-false}" \
        || [[ -n "${RCLONE_REMOTE:-}" ]]
}

verify_rclone_remote() {
    local remote_name="${RCLONE_REMOTE:-}"
    local configured_remotes

    remote_configuration_required || return 0
    command -v rclone >/dev/null 2>&1 || fatal "rclone is not available"
    [[ -n "$remote_name" ]] || fatal "RCLONE_REMOTE is required for remote features"
    [[ -n "${RCLONE_CONFIG:-}" ]] || fatal "RCLONE_CONFIG is required for remote features"
    [[ -f "$RCLONE_CONFIG" ]] || fatal "rclone config is unavailable"

    if ! configured_remotes="$(rclone listremotes --config "$RCLONE_CONFIG" 2>/dev/null)"; then
        fatal "rclone remote verification failed"
    fi
    if ! printf '%s\n' "$configured_remotes" | grep --fixed-strings --line-regexp \
        --quiet "${remote_name}:"; then
        fatal "configured rclone remote is unavailable"
    fi
}

start_runpod_base() {
    [[ -x /start.sh ]] || fatal "RunPod base /start.sh is unavailable"
    /start.sh &
    BASE_PROCESS_PID=$!
}

wait_for_comfyui() {
    local timeout_seconds="${IMAGE_STUDIO_BOOTSTRAP_COMFYUI_TIMEOUT_SECONDS:-900}"

    [[ "$timeout_seconds" =~ ^[0-9]+$ ]] || fatal "ComfyUI timeout must be an integer"
    command -v curl >/dev/null 2>&1 || fatal "curl is not available"
    log_message "waiting for ComfyUI readiness"
    for ((second = 0; second <= timeout_seconds; second++)); do
        if probe_comfyui; then
            log_message "ComfyUI is ready"
            return 0
        fi
        if ! kill -0 "$BASE_PROCESS_PID" 2>/dev/null; then
            fatal "RunPod base process exited before ComfyUI became ready"
        fi
        sleep 1
    done
    fatal "ComfyUI readiness timed out"
}

probe_comfyui() {
    curl --fail --silent --max-time 5 --output /dev/null "$COMFYUI_READY_URL"
}

start_image_studio() {
    [[ -x "$IMAGE_STUDIO_APP" ]] || fatal "Image Studio executable is unavailable"
    cd -- "$IMAGE_STUDIO_ROOT"
    "$IMAGE_STUDIO_APP" &
    IMAGE_STUDIO_PROCESS_PID=$!
    log_message "Image Studio started"
}

handle_shutdown_signal() {
    SHUTDOWN_REQUESTED=1
    shutdown_process "$IMAGE_STUDIO_PROCESS_PID" "Image Studio" || true
    shutdown_process "$BASE_PROCESS_PID" "RunPod base process" || true
}

validate_comfyui_monitor_configuration() {
    local monitor_interval="${IMAGE_STUDIO_BOOTSTRAP_COMFYUI_MONITOR_INTERVAL_SECONDS:-5}"
    local failure_threshold="${IMAGE_STUDIO_BOOTSTRAP_COMFYUI_FAILURE_THRESHOLD:-12}"

    [[ "$monitor_interval" =~ ^[1-9][0-9]*$ ]] \
        || fatal "ComfyUI monitor interval must be a positive integer"
    [[ "$failure_threshold" =~ ^[1-9][0-9]*$ ]] \
        || fatal "ComfyUI failure threshold must be a positive integer"
}

check_comfyui_liveness() {
    local failure_threshold="${IMAGE_STUDIO_BOOTSTRAP_COMFYUI_FAILURE_THRESHOLD:-12}"

    if probe_comfyui; then
        COMFYUI_FAILURE_COUNT=0
        return 0
    fi

    COMFYUI_FAILURE_COUNT=$((COMFYUI_FAILURE_COUNT + 1))
    if ((COMFYUI_FAILURE_COUNT >= failure_threshold)); then
        return 1
    fi
    return 0
}

monitor_processes() {
    local process_status
    local monitor_interval="${IMAGE_STUDIO_BOOTSTRAP_COMFYUI_MONITOR_INTERVAL_SECONDS:-5}"

    validate_comfyui_monitor_configuration

    while :; do
        if ! kill -0 "$IMAGE_STUDIO_PROCESS_PID" 2>/dev/null; then
            if [[ "$SHUTDOWN_REQUESTED" == 1 ]]; then
                return 0
            fi
            if wait "$IMAGE_STUDIO_PROCESS_PID"; then
                process_status=0
            else
                process_status=$?
            fi
            log_message "Image Studio exited unexpectedly (status ${process_status})"
            shutdown_process "$BASE_PROCESS_PID" "RunPod base process" || true
            return 1
        fi
        if ! kill -0 "$BASE_PROCESS_PID" 2>/dev/null; then
            if [[ "$SHUTDOWN_REQUESTED" == 1 ]]; then
                return 0
            fi
            if wait "$BASE_PROCESS_PID"; then
                process_status=0
            else
                process_status=$?
            fi
            log_message "RunPod base process exited unexpectedly (status ${process_status})"
            shutdown_process "$IMAGE_STUDIO_PROCESS_PID" "Image Studio" || true
            return 1
        fi
        if ! check_comfyui_liveness; then
            log_message "ERROR: ComfyUI failed its liveness probe persistently"
            shutdown_process "$IMAGE_STUDIO_PROCESS_PID" "Image Studio" || true
            shutdown_process "$BASE_PROCESS_PID" "RunPod base process" || true
            return 1
        fi
        sleep "$monitor_interval"
    done
}

main() {
    trap handle_shutdown_signal TERM INT
    materialize_rclone_config
    verify_rclone_remote
    start_runpod_base
    wait_for_comfyui
    start_image_studio
    monitor_processes
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
