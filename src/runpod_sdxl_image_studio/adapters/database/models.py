"""SQLAlchemy persistence models, kept separate from domain models."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from runpod_sdxl_image_studio.domain.lora_metadata import LoraMetadata


class Base(DeclarativeBase):
    """Alembic metadata root."""


class LoraMetadataModel(Base):
    __tablename__ = "lora_metadata"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    file_name: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_favorite: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    trigger_words_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    recommended_model_strength: Mapped[float | None] = mapped_column(nullable=True)
    recommended_clip_strength: Mapped[float | None] = mapped_column(nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    compatible_models_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    thumbnail_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_missing: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    usage_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    def to_domain(self) -> LoraMetadata:
        return LoraMetadata(
            id=UUID(self.id),
            file_name=self.file_name,
            display_name=self.display_name,
            category=self.category,
            is_favorite=self.is_favorite,
            trigger_words=tuple(json.loads(self.trigger_words_json)),
            recommended_model_strength=self.recommended_model_strength,
            recommended_clip_strength=self.recommended_clip_strength,
            notes=self.notes,
            compatible_models=tuple(json.loads(self.compatible_models_json)),
            thumbnail_path=self.thumbnail_path,
            is_missing=self.is_missing,
            usage_count=self.usage_count,
            last_used_at=_utc(self.last_used_at),
            created_at=_utc(self.created_at) or datetime.now(UTC),
            updated_at=_utc(self.updated_at) or datetime.now(UTC),
        )

    @classmethod
    def from_domain(cls, metadata: LoraMetadata) -> LoraMetadataModel:
        return cls(
            id=str(metadata.id),
            file_name=metadata.file_name,
            display_name=metadata.display_name,
            category=metadata.category,
            is_favorite=metadata.is_favorite,
            trigger_words_json=json.dumps(metadata.trigger_words, ensure_ascii=False),
            recommended_model_strength=metadata.recommended_model_strength,
            recommended_clip_strength=metadata.recommended_clip_strength,
            notes=metadata.notes,
            compatible_models_json=json.dumps(metadata.compatible_models, ensure_ascii=False),
            thumbnail_path=metadata.thumbnail_path,
            is_missing=metadata.is_missing,
            usage_count=metadata.usage_count,
            last_used_at=metadata.last_used_at,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
        )


class GenerationModel(Base):
    __tablename__ = "generations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    parent_generation_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("generations.id", ondelete="RESTRICT"), nullable=True
    )
    retry_of_generation_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("generations.id", ondelete="RESTRICT"), nullable=True
    )
    retry_attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    settings_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    checkpoint_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    vae_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    seed: Mapped[int | None] = mapped_column(nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    positive_prompt_search: Mapped[str | None] = mapped_column(Text, nullable=True)
    negative_prompt_search: Mapped[str | None] = mapped_column(Text, nullable=True)
    workflow_template_id: Mapped[str] = mapped_column(String(200), nullable=False)
    workflow_template_version: Mapped[str] = mapped_column(String(100), nullable=False)
    comfy_prompt_id: Mapped[str | None] = mapped_column(String(100), nullable=True, unique=True)
    favorite: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    user_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class GenerationArtifactModel(Base):
    __tablename__ = "generation_artifacts"
    __table_args__ = (
        UniqueConstraint("generation_id", "artifact_type", "sha256", name="uq_generation_artifact"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    generation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("generations.id", ondelete="CASCADE"), nullable=False
    )
    artifact_type: Mapped[str] = mapped_column(String(32), nullable=False)
    local_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MetadataImportModel(Base):
    """Stored canonical external image and its non-executable metadata preview."""

    __tablename__ = "metadata_imports"
    __table_args__ = (
        CheckConstraint(
            "metadata_status IN ('ready', 'needs_mapping', 'metadata_missing', 'invalid_metadata')",
            name="ck_metadata_import_status",
        ),
        CheckConstraint(
            "metadata_source IN ('comfyui_prompt', 'app_sidecar', 'workflow', 'none')",
            name="ck_metadata_import_source",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    stored_image_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    source_image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    stored_image_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    image_mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    metadata_source: Mapped[str] = mapped_column(String(32), nullable=False)
    metadata_status: Mapped[str] = mapped_column(String(32), nullable=False)
    raw_metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    raw_metadata_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_snapshot_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_snapshot_schema_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    manual_mapping_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    warnings_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class GenerationUpscaleSettingsModel(Base):
    """Immutable request projection for one upscale generation."""

    __tablename__ = "generation_upscale_settings"
    __table_args__ = (
        CheckConstraint("method IN ('image', 'latent')", name="ck_upscale_method"),
        CheckConstraint("sizing_mode IN ('factor', 'dimensions')", name="ck_upscale_sizing_mode"),
        CheckConstraint(
            "(sizing_mode='factor' AND scale_factor IS NOT NULL AND scale_factor > 1) OR "
            "(sizing_mode='dimensions' AND scale_factor IS NULL)",
            name="ck_upscale_sizing_values",
        ),
        CheckConstraint(
            "target_width > 0 AND target_height > 0", name="ck_upscale_target_positive"
        ),
        CheckConstraint(
            "denoise IS NULL OR (denoise >= 0 AND denoise <= 1)", name="ck_upscale_denoise"
        ),
        CheckConstraint(
            "(source_kind='generation_artifact' AND source_artifact_id IS NOT NULL "
            "AND source_import_id IS NULL) OR "
            "(source_kind='metadata_import' AND source_artifact_id IS NULL "
            "AND source_import_id IS NOT NULL)",
            name="ck_upscale_source_kind",
        ),
    )

    generation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("generations.id", ondelete="CASCADE"), primary_key=True
    )
    source_kind: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="generation_artifact",
        server_default="generation_artifact",
    )
    source_artifact_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("generation_artifacts.id", ondelete="RESTRICT"), nullable=True
    )
    source_import_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("metadata_imports.id", ondelete="RESTRICT"), nullable=True
    )
    method: Mapped[str] = mapped_column(String(16), nullable=False)
    sizing_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    scale_factor: Mapped[float | None] = mapped_column(nullable=True)
    target_width: Mapped[int] = mapped_column(Integer, nullable=False)
    target_height: Mapped[int] = mapped_column(Integer, nullable=False)
    upscaler_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    denoise: Mapped[float | None] = mapped_column(nullable=True)
    settings_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class GenerationJobModel(Base):
    __tablename__ = "generation_jobs"
    __table_args__ = (UniqueConstraint("generation_id", name="uq_generation_job_generation"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    generation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("generations.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    comfy_prompt_id: Mapped[str | None] = mapped_column(String(100), nullable=True, unique=True)
    progress_value: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress_maximum: Mapped[int | None] = mapped_column(Integer, nullable=True)
    current_node: Mapped[str | None] = mapped_column(String(200), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class GenerationLoraModel(Base):
    """検索用にsnapshotのLoRA指定を正規化した行。"""

    __tablename__ = "generation_loras"
    __table_args__ = (
        UniqueConstraint("generation_id", "lora_name", name="uq_generation_lora_name"),
    )

    generation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("generations.id", ondelete="CASCADE"), primary_key=True
    )
    lora_name: Mapped[str] = mapped_column(String(500), primary_key=True)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    model_strength: Mapped[float] = mapped_column(nullable=False)
    clip_strength: Mapped[float] = mapped_column(nullable=False)


class PresetModel(Base):
    """保存済みPresetのDB行。"""

    __tablename__ = "presets"
    __table_args__ = (UniqueConstraint("kind", "name", name="uq_preset_kind_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    favorite: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    usage_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class GenerationBatchModel(Base):
    """Persistent batch metadata for one FIFO enqueue operation."""

    __tablename__ = "generation_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    item_count: Mapped[int] = mapped_column(Integer, nullable=False)
    seed_strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    start_seed: Mapped[int | None] = mapped_column(BigInteger(), nullable=True)
    seed_step: Mapped[int] = mapped_column(BigInteger(), nullable=False)
    retry_of_batch_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("generation_batches.id", ondelete="RESTRICT"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        CheckConstraint("item_count > 0", name="ck_generation_batch_item_count_positive"),
        CheckConstraint("seed_step > 0", name="ck_generation_batch_seed_step_positive"),
    )


class GenerationQueueEntryModel(Base):
    """Persistent FIFO sequence and single-worker lease."""

    __tablename__ = "generation_queue_entries"

    sequence: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    generation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("generations.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("generation_jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    batch_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("generation_batches.id", ondelete="CASCADE"), nullable=True
    )
    batch_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    worker_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    submission_state: Mapped[str] = mapped_column(
        String(32), nullable=False, default="ready", server_default="ready"
    )
    submission_token: Mapped[str | None] = mapped_column(String(100), nullable=True)
    submission_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    enqueued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        CheckConstraint("batch_index >= 0", name="ck_generation_queue_batch_index_nonnegative"),
        CheckConstraint(
            "submission_state IN ('ready', 'submitting', 'submitted', 'ambiguous')",
            name="ck_generation_queue_submission_state",
        ),
        UniqueConstraint("batch_id", "batch_index", name="uq_generation_queue_batch_index"),
    )


Index("ix_generations_created_at", GenerationModel.created_at)
Index("ix_generations_status", GenerationModel.status)
Index("ix_generations_kind", GenerationModel.kind)
Index("ix_generations_favorite", GenerationModel.favorite)
Index("ix_generations_parent", GenerationModel.parent_generation_id)
Index("ix_generations_prompt", GenerationModel.comfy_prompt_id)
Index("ix_generations_checkpoint", GenerationModel.checkpoint_name)
Index("ix_generations_vae", GenerationModel.vae_name)
Index("ix_generations_seed", GenerationModel.seed)
Index("ix_generations_resolution", GenerationModel.width, GenerationModel.height)
Index("ix_generation_loras_name", GenerationLoraModel.lora_name)
Index("ix_generation_artifacts_generation", GenerationArtifactModel.generation_id)
Index("ix_generation_upscale_source_artifact", GenerationUpscaleSettingsModel.source_artifact_id)
Index("ix_generation_upscale_source_import", GenerationUpscaleSettingsModel.source_import_id)
Index("ix_generation_upscale_method", GenerationUpscaleSettingsModel.method)
Index("ix_generation_jobs_generation", GenerationJobModel.generation_id)
Index("ix_generation_jobs_prompt", GenerationJobModel.comfy_prompt_id)

Index("ix_lora_metadata_category", LoraMetadataModel.category)
Index("ix_lora_metadata_favorite", LoraMetadataModel.is_favorite)
Index("ix_lora_metadata_missing", LoraMetadataModel.is_missing)
Index("ix_lora_metadata_last_used", LoraMetadataModel.last_used_at)
Index("ix_presets_kind", PresetModel.kind)
Index("ix_presets_name", PresetModel.name)
Index("ix_presets_favorite", PresetModel.favorite)
Index("ix_presets_last_used", PresetModel.last_used_at)
Index("ix_presets_updated", PresetModel.updated_at)
Index("ix_generation_batches_created", GenerationBatchModel.created_at)
Index("ix_generation_queue_batch", GenerationQueueEntryModel.batch_id)
Index("ix_generation_queue_lease", GenerationQueueEntryModel.lease_expires_at)
Index("ix_metadata_imports_created", MetadataImportModel.created_at)
Index("ix_metadata_imports_status", MetadataImportModel.metadata_status)
Index("ix_metadata_imports_source_hash", MetadataImportModel.source_image_sha256)
Index(
    "uq_generations_retry_of_generation",
    GenerationModel.retry_of_generation_id,
    unique=True,
    sqlite_where=GenerationModel.retry_of_generation_id.is_not(None),
)
Index(
    "uq_generation_batches_retry_of_batch",
    GenerationBatchModel.retry_of_batch_id,
    unique=True,
    sqlite_where=GenerationBatchModel.retry_of_batch_id.is_not(None),
)


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
