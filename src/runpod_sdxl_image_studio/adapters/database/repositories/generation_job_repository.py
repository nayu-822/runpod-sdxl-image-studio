"""Compatibility module for the generation job repository."""

from .generation_repository import (
    GenerationJobRepository,
    GenerationJobRepositoryProtocol,
    GenerationRepositoryError,
)

__all__ = [
    "GenerationJobRepository",
    "GenerationJobRepositoryProtocol",
    "GenerationRepositoryError",
]
