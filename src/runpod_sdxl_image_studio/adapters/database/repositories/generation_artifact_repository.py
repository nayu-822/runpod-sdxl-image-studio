"""Compatibility module for the generation artifact repository."""

from .generation_repository import (
    GenerationArtifactRepository,
    GenerationArtifactRepositoryProtocol,
    GenerationRepositoryError,
)

__all__ = [
    "GenerationArtifactRepository",
    "GenerationArtifactRepositoryProtocol",
    "GenerationRepositoryError",
]
