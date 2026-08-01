"""View model for recently used generation inputs."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class RecentSettings:
    """最近使った値をカテゴリ別に限定して返すモデル。"""

    checkpoints: tuple[str, ...] = ()
    vaes: tuple[str, ...] = ()
    loras: tuple[str, ...] = ()
    generation_presets: tuple[UUID, ...] = ()
    prompt_presets: tuple[UUID, ...] = ()
    lora_presets: tuple[UUID, ...] = ()
    recent_generation_ids: tuple[UUID, ...] = ()
