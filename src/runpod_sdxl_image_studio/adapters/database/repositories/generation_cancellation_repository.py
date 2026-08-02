"""Atomic cancellation repository facade."""

from .generation_repository import (
    GenerationCancellationRepository,
    GenerationCancellationRepositoryProtocol,
    GenerationRepositoryError,
)

__all__ = [
    "GenerationCancellationRepository",
    "GenerationCancellationRepositoryProtocol",
    "GenerationRepositoryError",
]
