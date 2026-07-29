"""Transactional queue repository compatibility module."""

from .generation_repository import (
    GenerationQueueRepository,
    GenerationQueueRepositoryProtocol,
    GenerationRepositoryError,
)

__all__ = [
    "GenerationQueueRepository",
    "GenerationQueueRepositoryProtocol",
    "GenerationRepositoryError",
]
