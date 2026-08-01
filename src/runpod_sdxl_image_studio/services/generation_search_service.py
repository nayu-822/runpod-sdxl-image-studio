"""Application service for advanced history search."""

from __future__ import annotations

from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationRepositoryError,
    GenerationRepositoryProtocol,
)
from runpod_sdxl_image_studio.domain.generation_history import (
    GenerationHistoryPage,
    GenerationHistoryQuery,
)


class GenerationSearchError(RuntimeError):
    """履歴検索に失敗した。"""


class GenerationSearchService:
    """Typed queryをRepositoryへ渡す検索Application Service。"""

    def __init__(self, repository: GenerationRepositoryProtocol) -> None:
        self._repository = repository

    def search(self, query: GenerationHistoryQuery | None = None) -> GenerationHistoryPage:
        try:
            return self._repository.list_history(query or GenerationHistoryQuery())
        except GenerationRepositoryError as exc:
            raise GenerationSearchError("履歴を検索できませんでした。") from exc
