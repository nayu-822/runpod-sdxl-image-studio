"""Read only known PNG metadata fields without trusting executable workflow data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from runpod_sdxl_image_studio.domain.metadata_import import MetadataRawSource, MetadataSourceKind


@dataclass(frozen=True)
class PngMetadataResult:
    prompt: dict[str, object] | None
    workflow: object | None
    raw_sources: tuple[MetadataRawSource, ...]
    warnings: tuple[str, ...]


def parse_png_metadata(
    image_bytes: bytes,
    *,
    max_raw_bytes: int = 4_000_000,
) -> PngMetadataResult:
    """Extract only ComfyUI's known ``prompt`` and ``workflow`` chunks."""

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in {"PNG", "WEBP"}:
                return PngMetadataResult(None, None, (), ("metadata_import_invalid_image",))
            info = dict(image.info)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("image metadata could not be read") from exc

    raw_sources: list[MetadataRawSource] = []
    warnings: list[str] = []
    prompt: dict[str, object] | None = None
    workflow: object | None = None
    for key, source_value in (("prompt", info.get("prompt")), ("workflow", info.get("workflow"))):
        if source_value is None:
            continue
        raw_text = _raw_text(source_value)
        encoded = raw_text.encode("utf-8")
        if len(encoded) > max_raw_bytes:
            warnings.append("metadata_import_raw_too_large")
            continue
        kind = MetadataSourceKind.COMFYUI_PROMPT if key == "prompt" else MetadataSourceKind.WORKFLOW
        raw_sources.append(
            MetadataRawSource(
                kind=kind,
                raw_text=raw_text,
                sha256=hashlib.sha256(encoded).hexdigest(),
            )
        )
        if key == "prompt":
            try:
                parsed = source_value if isinstance(source_value, dict) else json.loads(raw_text)
            except (TypeError, ValueError, json.JSONDecodeError):
                warnings.append("metadata_import_parse_failed")
                continue
            if isinstance(parsed, dict):
                prompt = parsed
            else:
                warnings.append("metadata_import_invalid_json")
        else:
            workflow = source_value
    return PngMetadataResult(prompt, workflow, tuple(raw_sources), tuple(dict.fromkeys(warnings)))


def _raw_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = ["PngMetadataResult", "parse_png_metadata"]
