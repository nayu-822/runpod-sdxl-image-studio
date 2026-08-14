"""UX-facing generation preflight checks.

The queue worker keeps its own final validation.  This service is deliberately
read-only with respect to Generation, Job, and Queue rows.  Optional error
telemetry is persisted only while the lifecycle mutation gate admits it.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.adapters.storage.disk_usage import (
    DiskUsageAdapterProtocol,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation_settings import (
    CURRENT_WORKFLOW_TEMPLATE_VERSION,
    LEGACY_WORKFLOW_TEMPLATE_VERSION,
    GenerationSettings,
)
from runpod_sdxl_image_studio.domain.preflight import (
    PreflightIssue,
    PreflightResult,
    PreflightSeverity,
)
from runpod_sdxl_image_studio.domain.system_status import ComfyUIStatus, ErrorSeverity
from runpod_sdxl_image_studio.services.pod_lifecycle_service import (
    LifecycleGate,
    PodLifecycleWorkBlockedError,
)

logger = logging.getLogger(__name__)


class ComfyUIStatusProvider(Protocol):
    async def get_status(self) -> ComfyUIStatus: ...


class ErrorEventRecorder(Protocol):
    def record(
        self,
        *,
        category: str,
        severity: ErrorSeverity | str,
        error_code: str,
        summary: str,
        generation_id: UUID | None = None,
        job_id: UUID | None = None,
        retryable: bool = False,
        details: str | None = None,
        created_at: datetime | None = None,
    ) -> object: ...


DriveStatusProvider = Callable[[], Awaitable[object]]


class GenerationPreflightService:
    """Run shared capability, workflow, and local capacity checks."""

    def __init__(
        self,
        comfyui_service: ComfyUIStatusProvider,
        settings: Settings,
        *,
        disk_usage_adapter: DiskUsageAdapterProtocol | None = None,
        workflow_template: Mapping[str, object] | None = None,
        workflow_templates: Mapping[str, Mapping[str, object]] | None = None,
        drive_status_provider: DriveStatusProvider | None = None,
        error_recorder: ErrorEventRecorder | None = None,
        now_factory: Callable[[], datetime] | None = None,
        work_gate: LifecycleGate | None = None,
        state_changed_callback: Callable[[], None] | None = None,
    ) -> None:
        self._comfyui_service = comfyui_service
        self._settings = settings
        self._disk_usage_adapter = disk_usage_adapter
        self._workflow_template = workflow_template or {}
        self._workflow_templates = dict(workflow_templates or {})
        if workflow_template is not None:
            self._workflow_templates.setdefault("sdxl_txt2img", workflow_template)
        self._drive_status_provider = drive_status_provider
        self._error_recorder = error_recorder
        self._now_factory = now_factory or (lambda: datetime.now(UTC))
        self._work_gate = work_gate
        self._state_changed_callback = state_changed_callback

    async def check(
        self,
        generation_settings: GenerationSettings,
        *,
        uses_upscaler: bool = False,
        upscaler_name: str | None = None,
    ) -> PreflightResult:
        """Return a typed result without creating any persistent generation rows."""

        checked_at = _utc(self._now_factory())
        errors: list[PreflightIssue] = []
        warnings: list[PreflightIssue] = []

        status = await self._get_status(errors)
        capabilities = status.capabilities if status is not None else None
        if status is not None and not status.is_connected:
            errors.append(
                _error(
                    "comfyui_unavailable",
                    status.message or "ComfyUI is not connected",
                )
            )
        if capabilities is None:
            errors.append(
                _error(
                    "comfyui_capabilities_unavailable",
                    "ComfyUI capabilities could not be read",
                )
            )
        else:
            effective_uses_upscaler = uses_upscaler or generation_settings.final_upscale
            effective_upscaler_name = (
                upscaler_name
                if upscaler_name is not None
                else generation_settings.final_upscale_model
            )
            self._check_capabilities(
                capabilities,
                generation_settings,
                uses_upscaler=effective_uses_upscaler,
                upscaler_name=effective_upscaler_name,
                errors=errors,
            )
            self._check_required_nodes(
                capabilities,
                generation_settings,
                workflow_template_id=generation_settings.workflow_template_id,
                errors=errors,
            )

        self._check_disk(errors, warnings)
        await self._check_drive(warnings)

        if status is not None:
            warnings.extend(
                _warning("comfyui_warning", warning) for warning in status.warnings[:10]
            )
        result = PreflightResult(
            is_ready=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
            checked_at=checked_at,
        )
        self._record_issues(result)
        return result

    async def check_upscale(
        self,
        method: str,
        *,
        upscaler_name: str | None,
        source_settings: GenerationSettings | None = None,
    ) -> PreflightResult:
        """Run workflow-specific checks before an image or latent upscale enqueue."""

        checked_at = _utc(self._now_factory())
        errors: list[PreflightIssue] = []
        warnings: list[PreflightIssue] = []
        status = await self._get_status(errors)
        capabilities = status.capabilities if status is not None else None
        if status is not None and not status.is_connected:
            errors.append(
                _error("comfyui_unavailable", status.message or "ComfyUI is not connected")
            )
        if capabilities is None:
            errors.append(
                _error(
                    "comfyui_capabilities_unavailable",
                    "ComfyUI capabilities could not be read",
                )
            )
        elif method == "image":
            _require_choice(
                upscaler_name,
                capabilities.upscale_models,
                "upscaler_missing",
                "Selected upscaler is not available",
                errors,
            )
            self._check_required_nodes(
                capabilities,
                None,
                workflow_template_id="sdxl_image_upscale",
                errors=errors,
            )
        elif method == "latent":
            if source_settings is None:
                errors.append(
                    _error(
                        "upscale_source_settings_missing",
                        "Latent upscale requires the parent Generation snapshot",
                    )
                )
            else:
                self._check_capabilities(
                    capabilities,
                    source_settings,
                    uses_upscaler=False,
                    upscaler_name=None,
                    errors=errors,
                )
            self._check_required_nodes(
                capabilities,
                source_settings,
                workflow_template_id="sdxl_latent_upscale",
                errors=errors,
            )
        else:
            errors.append(_error("upscale_method_invalid", "Upscale method is not supported"))

        self._check_disk(errors, warnings)
        await self._check_drive(warnings)
        if status is not None:
            warnings.extend(
                _warning("comfyui_warning", warning) for warning in status.warnings[:10]
            )
        result = PreflightResult(
            is_ready=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
            checked_at=checked_at,
        )
        self._record_issues(result)
        return result

    async def run(
        self,
        generation_settings: GenerationSettings,
        **kwargs: object,
    ) -> PreflightResult:
        """Alias used by callers that model preflight as a command."""

        return await self.check(generation_settings, **kwargs)  # type: ignore[arg-type]

    async def validate(
        self,
        generation_settings: GenerationSettings,
        **kwargs: object,
    ) -> PreflightResult:
        """Compatibility alias for validation-oriented application code."""

        return await self.check(generation_settings, **kwargs)  # type: ignore[arg-type]

    async def _get_status(self, errors: list[PreflightIssue]) -> ComfyUIStatus | None:
        try:
            return await self._comfyui_service.get_status()
        except Exception:  # noqa: BLE001 - safe preflight boundary
            logger.warning("Generation preflight could not read ComfyUI status", exc_info=True)
            errors.append(_error("comfyui_status_failed", "ComfyUI status could not be read"))
            return None

    def _check_capabilities(
        self,
        capabilities: ComfyUICapabilities,
        generation_settings: GenerationSettings,
        *,
        uses_upscaler: bool,
        upscaler_name: str | None,
        errors: list[PreflightIssue],
    ) -> None:
        _require_choice(
            generation_settings.checkpoint_name,
            capabilities.checkpoints,
            "checkpoint_missing",
            "Selected checkpoint is not available",
            errors,
        )
        _require_choice(
            generation_settings.sampler_name,
            capabilities.samplers,
            "sampler_missing",
            "Selected sampler is not available",
            errors,
        )
        _require_choice(
            generation_settings.scheduler_name,
            capabilities.schedulers,
            "scheduler_missing",
            "Selected scheduler is not available",
            errors,
        )
        if generation_settings.vae_name is not None:
            _require_choice(
                generation_settings.vae_name,
                capabilities.vaes,
                "vae_missing",
                "Selected VAE is not available",
                errors,
            )
        for lora in generation_settings.loras:
            _require_choice(
                lora.name,
                capabilities.loras,
                "lora_missing",
                f"Selected LoRA is not available: {lora.name}",
                errors,
            )
        if uses_upscaler:
            _require_choice(
                upscaler_name,
                capabilities.upscale_models,
                "upscaler_missing",
                "Selected upscaler is not available",
                errors,
            )

    def _check_required_nodes(
        self,
        capabilities: ComfyUICapabilities,
        generation_settings: GenerationSettings | None,
        *,
        workflow_template_id: str,
        errors: list[PreflightIssue],
    ) -> None:
        required = _required_node_classes(
            self._workflow_templates.get(
                workflow_template_id,
                self._workflow_template if workflow_template_id == "sdxl_txt2img" else {},
            ),
            workflow_template_id=workflow_template_id,
        )
        if generation_settings is not None and generation_settings.vae_name is not None:
            required.add("VAELoader")
        if generation_settings is not None and generation_settings.loras:
            required.add("LoraLoader")
        if generation_settings is not None and generation_settings.hires_fix:
            if generation_settings.workflow_template_version == LEGACY_WORKFLOW_TEMPLATE_VERSION:
                required.add("LatentUpscale")
            elif generation_settings.workflow_template_version == CURRENT_WORKFLOW_TEMPLATE_VERSION:
                required.update({"ImageScaleBy", "VAEEncode"})
            else:
                errors.append(
                    _error(
                        "workflow_version_unsupported",
                        "Hires.fix workflow version is not supported",
                    )
                )
        if generation_settings is not None and generation_settings.final_upscale:
            required.update({"UpscaleModelLoader", "ImageUpscaleWithModel"})
        missing = sorted(required.difference(capabilities.available_node_classes))
        if missing:
            errors.append(
                _error(
                    "required_node_missing",
                    "Required ComfyUI node types are missing: " + ", ".join(missing),
                )
            )

    def _check_disk(
        self,
        errors: list[PreflightIssue],
        warnings: list[PreflightIssue],
    ) -> None:
        if self._disk_usage_adapter is None:
            return
        try:
            usage = self._disk_usage_adapter.usage(self._settings.data_dir)
        except Exception:  # noqa: BLE001 - fail closed for a generation guard
            errors.append(_error("disk_usage_unavailable", "Local disk capacity could not be read"))
            return
        if usage.free_bytes < self._settings.min_free_disk_bytes:
            errors.append(
                _error(
                    "disk_space_critical",
                    "Local disk free space is below the configured minimum",
                )
            )
        elif usage.free_bytes < self._settings.warning_free_disk_bytes:
            warnings.append(
                _warning(
                    "disk_space_low",
                    "Local disk free space is getting low",
                )
            )

    async def _check_drive(self, warnings: list[PreflightIssue]) -> None:
        if self._drive_status_provider is None:
            return
        try:
            result = await self._drive_status_provider()
        except Exception:  # noqa: BLE001 - Drive is never a generation hard stop
            warnings.append(
                _warning("drive_connection_unavailable", "Google Drive status could not be read")
            )
            return
        connected = (
            bool(getattr(result, "connected", False))
            or getattr(getattr(result, "status", None), "value", None) == "connected"
        )
        status_value = getattr(getattr(result, "status", None), "value", None)
        configured_value = getattr(result, "configured", None)
        configured = (
            bool(configured_value)
            if configured_value is not None
            else status_value != "not_configured"
        )
        if not configured:
            warnings.append(
                _warning(
                    "drive_not_configured",
                    "Google Drive is not configured; generation can continue and sync can be "
                    "enabled later",
                )
            )
        elif not connected:
            warnings.append(
                _warning(
                    "drive_not_connected",
                    "Google Drive is not connected; generation can continue and sync can retry",
                )
            )

    def _record_issues(self, result: PreflightResult) -> None:
        if self._error_recorder is None or not result.issues:
            return
        persisted = False
        try:
            mutation = (
                self._work_gate.admit_persistent_mutation()
                if self._work_gate is not None
                else nullcontext()
            )
            with mutation:
                for issue in result.issues:
                    try:
                        self._error_recorder.record(
                            category="generation_preflight",
                            severity=(
                                ErrorSeverity.ERROR if issue.is_error else ErrorSeverity.WARNING
                            ),
                            error_code=issue.code,
                            summary=issue.message,
                            retryable=issue.is_error,
                            created_at=result.checked_at,
                        )
                        persisted = True
                    except Exception:  # noqa: BLE001 - telemetry must not block enqueue UX
                        logger.warning("Preflight error history could not be saved", exc_info=True)
                if persisted and self._state_changed_callback is not None:
                    try:
                        self._state_changed_callback()
                    except Exception:  # noqa: BLE001 - backup notification must not block UX
                        logger.warning("Preflight state change notification failed")
        except PodLifecycleWorkBlockedError:
            logger.info("Preflight error history skipped after lifecycle drain")
        except Exception:  # noqa: BLE001 - telemetry must not block enqueue UX
            logger.warning("Preflight error history could not be saved", exc_info=True)


def _require_choice(
    selected: str | None,
    available: Sequence[str],
    code: str,
    message: str,
    errors: list[PreflightIssue],
) -> None:
    if selected is None or selected not in available:
        errors.append(_error(code, message))


_DEFAULT_REQUIRED_NODES = {
    "sdxl_image_upscale": {
        "LoadImage",
        "UpscaleModelLoader",
        "ImageUpscaleWithModel",
        "ImageScale",
        "SaveImage",
    },
    "sdxl_latent_upscale": {
        "LoadImage",
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "VAEEncode",
        "LatentUpscale",
        "KSampler",
        "VAEDecode",
        "SaveImage",
    },
}


def _required_node_classes(
    template: Mapping[str, object],
    *,
    workflow_template_id: str = "sdxl_txt2img",
) -> set[str]:
    value = template.get("required_node_classes")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return set(
            _DEFAULT_REQUIRED_NODES.get(
                workflow_template_id,
                {
                    "CheckpointLoaderSimple",
                    "CLIPTextEncode",
                    "EmptyLatentImage",
                    "KSampler",
                    "VAEDecode",
                    "SaveImage",
                },
            )
        )
    return {item for item in value if isinstance(item, str) and item}


def _error(code: str, message: str) -> PreflightIssue:
    return PreflightIssue(code, message, PreflightSeverity.ERROR)


def _warning(code: str, message: str) -> PreflightIssue:
    return PreflightIssue(code, message, PreflightSeverity.WARNING)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["GenerationPreflightService"]
