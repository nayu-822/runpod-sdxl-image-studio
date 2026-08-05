"""Domain models for the persistent single-worker generation queue."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from runpod_sdxl_image_studio.domain.generation import Generation
from runpod_sdxl_image_studio.domain.job import GenerationJob


class BatchSeedStrategy(StrEnum):
    """How seeds are resolved when a batch is enqueued."""

    RANDOM = "random"
    SEQUENTIAL = "sequential"


class SubmissionState(StrEnum):
    """Durable state machine for the non-idempotent ``/prompt`` call."""

    READY = "ready"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    AMBIGUOUS = "ambiguous"


class ReconciliationOutcome(StrEnum):
    """Outcome of checking a prompt that may have survived a restart."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"


class OptionalArtifactRepairOutcome(StrEnum):
    """Outcome of repairing optional artifacts for a completed generation."""

    REPAIRED = "repaired"
    ALREADY_COMPLETE = "already_complete"
    DEFERRED = "deferred"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class OptionalArtifactRepairCandidate:
    """Cursor material for one completed generation missing optional artifacts."""

    generation_id: UUID
    completed_at: datetime


class CancellationOutcome(StrEnum):
    """Typed result of requesting and verifying a ComfyUI cancellation."""

    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class GenerationBatch:
    """Persisted batch metadata; child Generation snapshots remain authoritative."""

    id: UUID
    name: str
    item_count: int
    seed_strategy: BatchSeedStrategy
    start_seed: int | None
    seed_step: int
    retry_of_batch_id: UUID | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class GenerationQueueEntry:
    """Persistent FIFO position and worker lease for one Generation."""

    sequence: int
    generation_id: UUID
    job_id: UUID
    batch_id: UUID | None
    batch_index: int
    worker_id: str | None
    claimed_at: datetime | None
    lease_expires_at: datetime | None
    cancel_requested_at: datetime | None
    submission_state: SubmissionState
    submission_token: str | None
    submission_started_at: datetime | None
    enqueued_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class GenerationQueueItem:
    """Queue entry with the persisted Generation and Job projections."""

    entry: GenerationQueueEntry
    generation: Generation
    job: GenerationJob
    batch: GenerationBatch | None = None


__all__ = [
    "BatchSeedStrategy",
    "CancellationOutcome",
    "GenerationBatch",
    "GenerationQueueEntry",
    "GenerationQueueItem",
    "OptionalArtifactRepairCandidate",
    "OptionalArtifactRepairOutcome",
    "ReconciliationOutcome",
    "SubmissionState",
]
