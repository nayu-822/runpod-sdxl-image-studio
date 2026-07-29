"""Transactional completion repository compatibility module."""

from .generation_repository import (
    GenerationCompletionRepository,
    GenerationCompletionRepositoryProtocol,
    GenerationRepositoryError,
)

__all__ = [
    "GenerationCompletionRepository",
    "GenerationCompletionRepositoryProtocol",
    "GenerationRepositoryError",
]
