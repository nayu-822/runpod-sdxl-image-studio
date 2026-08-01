"""Application service for bounded recent settings queries."""

from __future__ import annotations

from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.preset_repository import (
    PresetRepositoryProtocol,
)
from runpod_sdxl_image_studio.domain.generation_history import GenerationHistoryQuery
from runpod_sdxl_image_studio.domain.preset_payload import PresetKind
from runpod_sdxl_image_studio.domain.recent_settings import RecentSettings


class RecentSettingsService:
    """履歴全件をメモリへ読み込まず、DBの上限付き検索で最近値を求める。"""

    def __init__(
        self,
        generation_repository: GenerationRepositoryProtocol,
        preset_repository: PresetRepositoryProtocol,
        *,
        limit: int = 10,
    ) -> None:
        self._generation_repository = generation_repository
        self._preset_repository = preset_repository
        self._limit = min(max(1, limit), 100)

    def get_recent(self) -> RecentSettings:
        page = self._generation_repository.list_history(
            GenerationHistoryQuery(page_size=self._limit, limit=self._limit)
        )
        checkpoints: list[str] = []
        vaes: list[str] = []
        loras: list[str] = []
        for generation in page.generations:
            snapshot = generation.settings_snapshot
            _append_unique(checkpoints, snapshot.checkpoint_name, self._limit)
            if snapshot.vae_name:
                _append_unique(vaes, snapshot.vae_name, self._limit)
            for lora in snapshot.loras:
                _append_unique(loras, lora.name, self._limit)
        return RecentSettings(
            checkpoints=tuple(checkpoints),
            vaes=tuple(vaes),
            loras=tuple(loras),
            generation_presets=tuple(
                item.id
                for item in self._preset_repository.list(
                    kind=PresetKind.GENERATION, limit=self._limit
                )
            ),
            prompt_presets=tuple(
                item.id
                for item in self._preset_repository.list(kind=PresetKind.PROMPT, limit=self._limit)
            ),
            lora_presets=tuple(
                item.id
                for item in self._preset_repository.list(kind=PresetKind.LORA, limit=self._limit)
            ),
            recent_generation_ids=tuple(item.id for item in page.generations),
        )


def _append_unique(values: list[str], value: str, limit: int) -> None:
    if value not in values and len(values) < limit:
        values.append(value)
