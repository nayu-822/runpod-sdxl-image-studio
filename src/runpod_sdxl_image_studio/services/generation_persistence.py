"""Generation persistence dependency contracts."""

from __future__ import annotations

from dataclasses import dataclass

from runpod_sdxl_image_studio.adapters.database.repositories.generation_progress_repository import (
    GenerationProgressRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepositoryProtocol,
    GenerationCompletionRepositoryProtocol,
    GenerationFailureRepositoryProtocol,
    GenerationJobRepositoryProtocol,
    GenerationQueueRepositoryProtocol,
    GenerationRepositoryProtocol,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_start_repository import (
    GenerationStartRepositoryProtocol,
)


@dataclass(frozen=True)
class GenerationPersistenceRepositories:
    """Generation処理で使用する永続化Repositoryの組み合わせ。"""

    generation: GenerationRepositoryProtocol
    job: GenerationJobRepositoryProtocol
    artifact: GenerationArtifactRepositoryProtocol
    start: GenerationStartRepositoryProtocol
    queue: GenerationQueueRepositoryProtocol
    progress: GenerationProgressRepositoryProtocol
    completion: GenerationCompletionRepositoryProtocol
    failure: GenerationFailureRepositoryProtocol


__all__ = ["GenerationPersistenceRepositories"]
