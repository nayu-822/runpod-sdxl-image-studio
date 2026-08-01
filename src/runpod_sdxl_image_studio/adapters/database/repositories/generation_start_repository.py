"""Atomic creation of a pending Generation and its GenerationJob."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.adapters.database.engine import session_scope
from runpod_sdxl_image_studio.adapters.database.models import (
    GenerationJobModel,
    GenerationLoraModel,
    GenerationModel,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationRepositoryError,
    _generation_domain,
    _job_domain,
)
from runpod_sdxl_image_studio.domain.generation import Generation, GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.job import GenerationJob


class GenerationStartRepositoryProtocol(Protocol):
    """GenerationとJobを原子的にpending作成する契約。"""

    def create_pending(
        self,
        snapshot: GenerationSettingsSnapshot,
        *,
        generation_id: UUID,
        job_id: UUID,
        kind: GenerationKind,
        parent_generation_id: UUID | None,
        created_at: datetime,
    ) -> tuple[Generation, GenerationJob]: ...


class GenerationStartRepository(GenerationStartRepositoryProtocol):
    """Persist a pending Generation and Job in one SQLite transaction."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create_pending(
        self,
        snapshot: GenerationSettingsSnapshot,
        *,
        generation_id: UUID,
        job_id: UUID,
        kind: GenerationKind,
        parent_generation_id: UUID | None,
        created_at: datetime,
    ) -> tuple[Generation, GenerationJob]:
        timestamp = _utc(created_at)
        try:
            validated_snapshot = GenerationSettingsSnapshot.model_validate(snapshot.model_dump())
            with session_scope(self._session_factory) as session:
                if (
                    parent_generation_id is not None
                    and session.get(GenerationModel, str(parent_generation_id)) is None
                ):
                    raise GenerationRepositoryError("parent generation was not found")

                generation_row = GenerationModel(
                    id=str(generation_id),
                    kind=kind.value,
                    status=GenerationStatus.PENDING.value,
                    parent_generation_id=(
                        str(parent_generation_id) if parent_generation_id is not None else None
                    ),
                    settings_snapshot_json=validated_snapshot.to_json(),
                    snapshot_schema_version=validated_snapshot.schema_version,
                    checkpoint_name=validated_snapshot.checkpoint_name,
                    vae_name=validated_snapshot.vae_name,
                    seed=validated_snapshot.seed,
                    width=validated_snapshot.width,
                    height=validated_snapshot.height,
                    positive_prompt_search=validated_snapshot.positive_prompt,
                    negative_prompt_search=validated_snapshot.negative_prompt,
                    workflow_template_id=validated_snapshot.workflow_template_id,
                    workflow_template_version=validated_snapshot.workflow_template_version,
                    favorite=False,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                job_row = GenerationJobModel(
                    id=str(job_id),
                    generation_id=str(generation_id),
                    status=GenerationStatus.PENDING.value,
                    comfy_prompt_id=None,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                session.add(generation_row)
                # 明示的にGenerationを先にflushし、Jobの外部キーを満たす。
                # 2回のflushは同じsession_scope内なので、後続失敗時は両方rollbackされる。
                session.flush()
                session.add(job_row)
                session.add_all(_generation_lora_rows(validated_snapshot, generation_id))
                session.flush()
                return _generation_domain(generation_row), _job_domain(job_row)
        except GenerationRepositoryError:
            raise
        except (IntegrityError, SQLAlchemyError, ValidationError, TypeError, ValueError) as exc:
            raise GenerationRepositoryError("pending generation pair could not be created") from exc


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _generation_lora_rows(
    snapshot: GenerationSettingsSnapshot, generation_id: UUID
) -> list[GenerationLoraModel]:
    return [
        GenerationLoraModel(
            generation_id=str(generation_id),
            lora_name=lora.name,
            order_index=lora.order,
            model_strength=lora.model_strength,
            clip_strength=lora.clip_strength,
        )
        for lora in snapshot.loras
    ]


__all__ = [
    "GenerationStartRepository",
    "GenerationStartRepositoryProtocol",
    "GenerationRepositoryError",
]
