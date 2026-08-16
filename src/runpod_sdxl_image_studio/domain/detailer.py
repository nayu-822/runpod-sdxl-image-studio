"""Typed Detailer settings and the repository-supported Detailer registry."""

from __future__ import annotations

import ntpath
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FACE_DEFAULT_DETECTOR_MODEL = "bbox/face_yolov8m.pt"
MAX_DETAILER_SEED = 2**63 - 1
FACE_DEFAULT_POSITIVE_PROMPT = (
    "beautiful face, cute face, detailed face, clean facial features, "
    "detailed eyes, beautiful eyes, highly detailed eyes, detailed iris, "
    "detailed pupils, sharp pupils, symmetrical eyes, eye highlights, "
    "sparkling eyes, glossy eyes, detailed eyelashes, long eyelashes, "
    "anime face, anime coloring, clean lineart, soft cel shading"
)
FACE_DEFAULT_NEGATIVE_PROMPT = (
    "bad face, deformed face, blurry face, bad eyes, deformed eyes, "
    "asymmetrical eyes, cross-eyed, misaligned eyes, extra eyes, "
    "extra pupils, multiple pupils, blurry eyes, bad pupils, malformed pupils"
)


class DetailerKind(StrEnum):
    """Detailer stages supported by the application registry."""

    FACE = "face"


class DetailerSettings(BaseModel):
    """Validated settings for one Detailer stage.

    The values intentionally mirror the stable API inputs used by the
    repository-controlled FaceDetailer graph.  Frontend-only widget values,
    SAM model links, and arbitrary node inputs are not part of this model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: DetailerKind = DetailerKind.FACE
    enabled: bool = True
    order: int = Field(default=0, ge=0)
    detector_model: str = Field(default=FACE_DEFAULT_DETECTOR_MODEL, min_length=1)
    positive_prompt: str = Field(default=FACE_DEFAULT_POSITIVE_PROMPT, max_length=10_000)
    negative_prompt: str = Field(default=FACE_DEFAULT_NEGATIVE_PROMPT, max_length=10_000)
    guide_size: int = Field(default=768, gt=0, le=8192)
    guide_size_for: bool = True
    max_size: int = Field(default=1024, gt=0, le=16384)
    seed: int = Field(default=0, ge=0, le=MAX_DETAILER_SEED)
    steps: int = Field(default=20, ge=1, le=150)
    cfg_scale: float = Field(default=5.0, ge=0.0, le=30.0)
    sampler_name: str = Field(default="euler_ancestral", min_length=1, max_length=200)
    scheduler_name: str = Field(default="normal", min_length=1, max_length=200)
    denoise: float = Field(default=0.22, ge=0.0, le=1.0)
    feather: int = Field(default=5, ge=0, le=512)
    noise_mask: bool = True
    force_inpaint: bool = True
    bbox_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    bbox_dilation: int = Field(default=10, ge=0, le=2048)
    bbox_crop_factor: float = Field(default=2.0, gt=0.0, le=20.0)
    sam_detection_hint: str = Field(default="center-1", min_length=1, max_length=100)
    sam_dilation: int = Field(default=0, ge=0, le=2048)
    sam_threshold: float = Field(default=0.93, ge=0.0, le=1.0)
    sam_bbox_expansion: int = Field(default=0, ge=0, le=2048)
    sam_mask_hint_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # The repository-controlled Impact Pack graph currently accepts only the
    # boolean-string value.  Do not advertise values that object_info rejects.
    sam_mask_hint_use_negative: Literal["False"] = "False"
    drop_size: int = Field(default=10, ge=0, le=4096)
    wildcard: str = Field(default="", max_length=10_000)
    cycle: int = Field(default=1, ge=1, le=100)
    inpaint_model: bool = False
    noise_mask_feather: int = Field(default=20, ge=0, le=512)
    tiled_encode: bool = False
    tiled_decode: bool = False

    @field_validator("detector_model")
    @classmethod
    def validate_detector_model(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if (
            not normalized
            or posixpath.isabs(normalized)
            or ntpath.isabs(normalized)
            or any(part in {"", ".", ".."} for part in normalized.split("/"))
        ):
            raise ValueError("detector_model must be a safe relative path")
        return normalized


@dataclass(frozen=True)
class DetailerDefinition:
    """Repository-controlled node mapping and default settings for a kind."""

    kind: DetailerKind
    node_class: str
    detector_provider_class: str
    default_settings: DetailerSettings


class DetailerRegistry:
    """Small immutable registry that keeps kind-specific workflow facts out of the adapter."""

    def __init__(
        self,
        definitions: Mapping[DetailerKind, DetailerDefinition] | None = None,
    ) -> None:
        source = definitions or {
            DetailerKind.FACE: DetailerDefinition(
                kind=DetailerKind.FACE,
                node_class="FaceDetailer",
                detector_provider_class="UltralyticsDetectorProvider",
                default_settings=DetailerSettings(),
            )
        }
        self._definitions = MappingProxyType(dict(source))

    def get(self, kind: DetailerKind) -> DetailerDefinition | None:
        return self._definitions.get(kind)

    def require(self, kind: DetailerKind) -> DetailerDefinition:
        definition = self.get(kind)
        if definition is None:
            raise ValueError(f"unsupported Detailer kind: {kind}")
        return definition

    def default_settings(self, kind: DetailerKind) -> DetailerSettings:
        return self.require(kind).default_settings.model_copy(deep=True)


DEFAULT_DETAILER_REGISTRY = DetailerRegistry()


__all__ = [
    "DEFAULT_DETAILER_REGISTRY",
    "FACE_DEFAULT_DETECTOR_MODEL",
    "FACE_DEFAULT_NEGATIVE_PROMPT",
    "FACE_DEFAULT_POSITIVE_PROMPT",
    "DetailerDefinition",
    "DetailerKind",
    "DetailerRegistry",
    "DetailerSettings",
]
