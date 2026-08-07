"""Parse the application's schema-versioned sidecar JSON safely."""

from __future__ import annotations

import hashlib
import json
import ntpath
import posixpath
from dataclasses import dataclass

from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.domain.metadata_import import (
    MetadataFieldResolution,
    MetadataFieldStatus,
    MetadataImportCandidate,
    MetadataRawSource,
    MetadataSourceKind,
)

SUPPORTED_SIDECAR_SCHEMA_VERSION = 1


class SidecarMetadataError(ValueError):
    """Safe parsing error carrying a stable import error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SidecarMetadataResult:
    candidate: MetadataImportCandidate
    raw_source: MetadataRawSource
    warnings: tuple[str, ...]


def migrate_import_schema_v1(payload: dict[str, object]) -> dict[str, object]:
    """Validate the explicit Phase 6 import schema converter entry point."""

    if payload.get("schema_version") != SUPPORTED_SIDECAR_SCHEMA_VERSION:
        raise SidecarMetadataError(
            "metadata_import_unsupported_schema", "unsupported sidecar schema version"
        )
    return dict(payload)


def parse_sidecar_metadata(
    payload: bytes | str | bytearray,
    *,
    source_image_sha256: str | None = None,
    max_raw_bytes: int = 4_000_000,
) -> SidecarMetadataResult:
    raw_bytes = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    if len(raw_bytes) > max_raw_bytes:
        raise SidecarMetadataError("metadata_import_too_large", "sidecar exceeds the size limit")
    raw_text = raw_bytes.decode("utf-8", errors="strict")
    try:
        parsed = json.loads(raw_text)
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SidecarMetadataError(
            "metadata_import_invalid_json", "sidecar JSON is invalid"
        ) from exc
    if not isinstance(parsed, dict):
        raise SidecarMetadataError("metadata_import_invalid_json", "sidecar JSON must be an object")
    parsed = migrate_import_schema_v1(parsed)
    raw_source = MetadataRawSource(
        kind=MetadataSourceKind.APP_SIDECAR,
        raw_text=raw_text,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    warnings: list[str] = []
    image = parsed.get("image")
    if isinstance(image, dict):
        declared_hash = image.get("sha256")
        if (
            isinstance(declared_hash, str)
            and source_image_sha256 is not None
            and declared_hash.lower() != source_image_sha256.lower()
        ):
            warnings.append("metadata_import_sidecar_hash_mismatch")
    candidate = _candidate_from_settings(parsed.get("settings"), warnings)
    return SidecarMetadataResult(candidate, raw_source, tuple(dict.fromkeys(warnings)))


def _candidate_from_settings(
    raw_settings: object,
    warnings: list[str],
) -> MetadataImportCandidate:
    if not isinstance(raw_settings, dict):
        return MetadataImportCandidate(
            source_kind=MetadataSourceKind.APP_SIDECAR,
            unresolved_fields=("settings",),
            warnings=("metadata_import_unresolved",),
        )
    source = raw_settings
    positive = _string(source.get("positive_prompt"))
    negative = _string(source.get("negative_prompt"))
    seed = _int(source.get("seed"))
    width = _int(source.get("width"))
    height = _int(source.get("height"))
    steps = _int(source.get("steps"))
    cfg_scale = _float(source.get("cfg_scale"))
    sampler = _string(source.get("sampler_name"))
    scheduler = _string(source.get("scheduler_name"))
    checkpoint = _safe_model_name(source.get("checkpoint_name"))
    vae = _safe_model_name(source.get("vae_name")) if source.get("vae_name") is not None else None
    loras, lora_ok = _loras(source.get("loras"))
    unresolved = [
        name
        for name, value in (
            ("positive_prompt", positive),
            ("negative_prompt", negative),
            ("seed", seed),
            ("width", width),
            ("height", height),
            ("steps", steps),
            ("cfg_scale", cfg_scale),
            ("sampler_name", sampler),
            ("scheduler_name", scheduler),
            ("checkpoint_name", checkpoint),
        )
        if value is None
    ]
    if not lora_ok:
        unresolved.append("loras")
    resolutions = tuple(
        MetadataFieldResolution(
            field_name=field,
            status=(
                MetadataFieldStatus.UNRESOLVED
                if field in unresolved
                else MetadataFieldStatus.RESOLVED
            ),
            value=value,
        )
        for field, value in (
            ("positive_prompt", positive),
            ("negative_prompt", negative),
            ("seed", seed),
            ("width", width),
            ("height", height),
            ("steps", steps),
            ("cfg_scale", cfg_scale),
            ("sampler_name", sampler),
            ("scheduler_name", scheduler),
            ("checkpoint_name", checkpoint),
            ("vae_name", vae),
            ("loras", loras),
        )
    )
    if source.get("workflow_template_id") is not None:
        warnings.append("metadata_import_workflow_ignored")
    return MetadataImportCandidate(
        source_kind=MetadataSourceKind.APP_SIDECAR,
        positive_prompt=positive,
        negative_prompt=negative,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        scheduler_name=scheduler,
        checkpoint_name=checkpoint,
        vae_name=vae,
        loras=loras,
        unresolved_fields=tuple(dict.fromkeys(unresolved)),
        warnings=tuple(dict.fromkeys(warnings)),
        resolutions=resolutions,
    )


def _loras(value: object) -> tuple[tuple[LoraSetting, ...], bool]:
    if value is None:
        return (), True
    if not isinstance(value, (list, tuple)):
        return (), False
    result: list[LoraSetting] = []
    try:
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                return (), False
            result.append(
                LoraSetting(
                    name=_safe_model_name(item.get("name")) or "",
                    model_strength=_float(item.get("model_strength")) or 0.0,
                    clip_strength=_float(item.get("clip_strength")) or 0.0,
                    order=_int(item.get("order")) if _int(item.get("order")) is not None else index,
                )
            )
    except ValueError:
        return (), False
    return tuple(result), True


def _safe_model_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace("\\", "/")
    if (
        not normalized
        or "\x00" in normalized
        or posixpath.isabs(normalized)
        or ntpath.isabs(value)
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        return None
    return normalized


def _string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


__all__ = [
    "SUPPORTED_SIDECAR_SCHEMA_VERSION",
    "SidecarMetadataError",
    "SidecarMetadataResult",
    "migrate_import_schema_v1",
    "parse_sidecar_metadata",
]
