"""Parse the application's schema-versioned sidecar JSON safely."""

from __future__ import annotations

import hashlib
import json
import math
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
    # Prompt text is user data, not a model identifier.  Preserve it exactly,
    # including an intentionally empty value and surrounding whitespace.
    positive = _prompt(source, "positive_prompt")
    negative = _prompt(source, "negative_prompt")
    seed = _int(source.get("seed"))
    width = _int(source.get("width"))
    height = _int(source.get("height"))
    steps = _int(source.get("steps"))
    cfg_scale = _float(source.get("cfg_scale"))
    sampler = _string(source.get("sampler_name"))
    scheduler = _string(source.get("scheduler_name"))
    checkpoint = _safe_model_name(source.get("checkpoint_name"))
    raw_vae = source.get("vae_name")
    vae = _safe_model_name(raw_vae) if raw_vae is not None else None
    vae_ok = "vae_name" in source and (raw_vae is None or vae is not None)
    loras, lora_ok = _loras(source.get("loras"), present="loras" in source)
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
    if not vae_ok:
        unresolved.append("vae_name")
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


def _loras(value: object, *, present: bool) -> tuple[tuple[LoraSetting, ...], bool]:
    # An omitted key is different from an explicitly empty collection.  The
    # latter is a fully resolved "no LoRA" choice.
    if not present:
        return (), False
    if not isinstance(value, (list, tuple)):
        return (), False
    result: list[LoraSetting] = []
    try:
        for item in value:
            if not isinstance(item, dict):
                return (), False
            required = {"name", "model_strength", "clip_strength", "order"}
            if not required.issubset(item):
                return (), False
            name = _safe_model_name(item.get("name"))
            model_strength = _float(item.get("model_strength"))
            clip_strength = _float(item.get("clip_strength"))
            order = _order(item.get("order"))
            if name is None or model_strength is None or clip_strength is None or order is None:
                return (), False
            result.append(
                LoraSetting(
                    name=name,
                    model_strength=model_strength,
                    clip_strength=clip_strength,
                    order=order,
                )
            )
    except ValueError:
        return (), False
    if len({item.name for item in result}) != len(result):
        return (), False
    if len({item.order for item in result}) != len(result):
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


def _prompt(source: dict[str, object], key: str) -> str | None:
    value = source.get(key)
    return value if isinstance(value, str) else None


def _int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _order(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


__all__ = [
    "SUPPORTED_SIDECAR_SCHEMA_VERSION",
    "SidecarMetadataError",
    "SidecarMetadataResult",
    "migrate_import_schema_v1",
    "parse_sidecar_metadata",
]
