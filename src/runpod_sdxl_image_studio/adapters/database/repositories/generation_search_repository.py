"""Repository boundary dedicated to advanced generation history search."""

from __future__ import annotations

from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationRepository,
    GenerationRepositoryError,
)
from runpod_sdxl_image_studio.domain.generation_history import (
    GenerationHistoryPage,
    GenerationHistoryQuery,
)


class GenerationSearchRepository:
    """Search-only facade that keeps SQL out of the application service."""

    def __init__(self, generation_repository: GenerationRepository) -> None:
        self._generation_repository = generation_repository

    def search(self, query: GenerationHistoryQuery) -> GenerationHistoryPage:
        return self._generation_repository.list_history(query)


__all__ = ["GenerationSearchRepository", "GenerationRepositoryError"]
