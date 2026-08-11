from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUIQueueStatus
from runpod_sdxl_image_studio.adapters.runpod.pod_lifecycle import (
    RUNPOD_API_BASE_URL,
    RunPodIdentity,
    RunPodLifecycleAdapter,
    RunPodTerminateError,
    RunPodTerminateResult,
    RunPodTerminateStatus,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.drive_sync import DriveManifestState, DriveSyncStatus
from runpod_sdxl_image_studio.domain.generation import Generation, GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_queue import SubmissionState
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.model_transfer import ModelTransferStatus
from runpod_sdxl_image_studio.domain.pod_lifecycle import AutoTerminateState, PodLifecycleSession
from runpod_sdxl_image_studio.domain.state_sync import StateSyncStatus, StateSyncView
from runpod_sdxl_image_studio.jobs.auto_terminate_worker import AutoTerminateCoordinator
from runpod_sdxl_image_studio.services.drive_sync_service import (
    DriveSyncService,
    DriveSyncServiceError,
)
from runpod_sdxl_image_studio.services.generation_history_service import GenerationHistoryService
from runpod_sdxl_image_studio.services.generation_queue_service import (
    GenerationQueueService,
    GenerationQueueServiceError,
)
from runpod_sdxl_image_studio.services.model_preparation_service import (
    ModelPreparationService,
    ModelPreparationServiceError,
)
from runpod_sdxl_image_studio.services.pod_lifecycle_service import (
    PodLifecycleService,
    PodLifecycleTransitionError,
    PodLifecycleWorkBlockedError,
)
from runpod_sdxl_image_studio.services.upscale_enqueue_service import (
    UpscaleEnqueueError,
    UpscaleEnqueueService,
)


@dataclass
class FakeLifecycleRepository:
    session: PodLifecycleSession | None = None
    sessions: dict[str, PodLifecycleSession] | None = None

    def get_by_pod_id(self, pod_id: str) -> PodLifecycleSession | None:
        if self.sessions is not None:
            return self.sessions.get(pod_id)
        return self.session if self.session and self.session.pod_id == pod_id else None

    def get_or_create(
        self, pod_id: str, *, auto_terminate_enabled: bool, now: datetime | None = None
    ) -> PodLifecycleSession:
        if self.sessions is not None:
            existing = self.sessions.get(pod_id)
            if existing is not None:
                return existing
        elif self.session is not None:
            return self.session
        if self.session is None or self.sessions is not None:
            timestamp = now or datetime.now(UTC)
            session = PodLifecycleSession(
                pod_id=pod_id,
                started_at=timestamp,
                auto_terminate_enabled=auto_terminate_enabled,
                created_at=timestamp,
                updated_at=timestamp,
            )
            self.session = session
            if self.sessions is not None:
                self.sessions[pod_id] = session
        return self.session

    def save(self, session: PodLifecycleSession) -> PodLifecycleSession:
        self.session = session
        if self.sessions is not None:
            self.sessions[session.pod_id] = session
        return session


class FakeGenerationRepository:
    def __init__(self, generations: tuple[Generation, ...]) -> None:
        self.generations = generations

    def list_since(self, started_at: datetime, limit: int = 1000) -> tuple[Generation, ...]:
        return tuple(item for item in self.generations if item.created_at >= started_at)[:limit]

    def list_since_unbounded(self, started_at: datetime) -> tuple[Generation, ...]:
        return tuple(item for item in self.generations if item.created_at >= started_at)


class FakeQueueRepository:
    def __init__(self) -> None:
        self.items: list[object] = []

    def list_queue(self, *, limit: int = 200, **_: object) -> tuple[object, ...]:
        return tuple(self.items[:limit])

    def has_active_generation_work_since(self, started_at: datetime) -> bool:
        del started_at
        active = {
            GenerationStatus.PENDING,
            GenerationStatus.QUEUED,
            GenerationStatus.RUNNING,
        }
        active_submission = {SubmissionState.SUBMITTING, SubmissionState.AMBIGUOUS}
        for item in self.items:
            generation = getattr(item, "generation", None)
            job = getattr(item, "job", None)
            entry = getattr(item, "entry", None)
            generation_status = getattr(generation, "status", None)
            job_status = getattr(job, "status", generation_status)
            if (
                generation_status in active
                or job_status in active
                or job_status != generation_status
                or getattr(entry, "submission_state", None) in active_submission
            ):
                return True
        return False


class FakeModelRepository:
    def __init__(self, counts: dict[ModelTransferStatus, int] | None = None) -> None:
        self.counts = counts or {}

    def status_counts(self) -> dict[ModelTransferStatus, int]:
        return self.counts


class FakeDriveRepository:
    def __init__(
        self,
        record: object,
        counts: dict[DriveSyncStatus, int] | None = None,
        manifest_state: DriveManifestState = DriveManifestState.SYNCED,
    ) -> None:
        self.record = record
        self.counts = counts or {DriveSyncStatus.SYNCED: 1}
        self.manifest_state = manifest_state
        self.manifest_jobs: list[object] = []
        self.missing_generation_id: object | None = None

    def status_counts(self) -> dict[DriveSyncStatus, int]:
        return self.counts

    def get_by_generation(self, generation_id: object) -> object:
        if generation_id == self.missing_generation_id:
            return None
        return self.record

    def list_manifest_jobs(self, limit: int = 50) -> tuple[object, ...]:
        return tuple(self.manifest_jobs[:limit])

    def has_active_manifest_jobs(self) -> bool:
        return any(
            job.status in {DriveSyncStatus.PENDING, DriveSyncStatus.SYNCING}
            for job in self.manifest_jobs
        )

    def manifest_state_for_destination(
        self,
        local_date: str,
        destination: object,
    ) -> DriveManifestState:
        return self.manifest_state


class FakeStateSync:
    def __init__(
        self,
        *,
        status: StateSyncStatus = StateSyncStatus.SYNCED,
        dirty_on_backup: bool = False,
        fail_on_backup: bool = False,
    ) -> None:
        self.enabled = True
        self.backup_in_progress = False
        self.is_clean = True
        self.has_latest_remote_backup = True
        self.status = status
        self.dirty_on_backup = dirty_on_backup
        self.fail_on_backup = fail_on_backup
        self.backup_calls = 0
        self.mark_dirty_calls = 0

    def get_status(self) -> StateSyncView:
        return StateSyncView(self.status)

    async def backup(self, *, wait_for_clean: bool = True) -> StateSyncView:
        self.backup_calls += 1
        if self.dirty_on_backup:
            self.is_clean = False
        if self.fail_on_backup:
            self.status = StateSyncStatus.FAILED
            self.is_clean = False
        return self.get_status()

    def mark_dirty(self) -> None:
        self.mark_dirty_calls += 1
        self.is_clean = False


class FakeRunPod:
    def __init__(self) -> None:
        self.calls = 0
        self.identity_ready = True
        self.pod_id = "pod-current"

    def identity(self) -> RunPodIdentity:
        return RunPodIdentity(self.pod_id, True)

    async def terminate_self(self) -> RunPodTerminateResult:
        self.calls += 1
        return RunPodTerminateResult(RunPodTerminateStatus.TERMINATED)


class FailingRunPod(FakeRunPod):
    def __init__(self, *, ambiguous: bool) -> None:
        super().__init__()
        self.ambiguous = ambiguous

    async def terminate_self(self) -> RunPodTerminateResult:
        self.calls += 1
        raise RunPodTerminateError(
            "runpod_terminate_confirmed_failure"
            if not self.ambiguous
            else "runpod_terminate_ambiguous",
            "termination failed",
            ambiguous=self.ambiguous,
        )


class BlockingDispatchRepository:
    def __init__(self, generation: Generation | None = None) -> None:
        self.persisted = 0
        self.batch_persisted = 0
        self.item: object | None = None
        self.generation = generation
        self.entered = threading.Event()
        self.release = threading.Event()

    def enqueue_single(self, snapshot: object, **_: object) -> object:
        self.persisted += 1
        self.entered.set()
        self.release.wait(timeout=2.0)
        self.item = SimpleNamespace(
            entry=SimpleNamespace(sequence=1),
            generation=self.generation,
            job=SimpleNamespace(status=GenerationStatus.PENDING),
        )
        return self.item

    def enqueue_batch(self, snapshots: object, **_: object) -> object:
        del snapshots
        self.batch_persisted += 1
        raise AssertionError("batch repository must not be called after DRAINING")

    def list_queue(self, **_: object) -> tuple[object, ...]:
        return (self.item,) if self.item is not None else ()


def _generation(status: GenerationStatus = GenerationStatus.COMPLETED) -> Generation:
    timestamp = datetime(2026, 8, 10, 12, tzinfo=UTC)
    snapshot = GenerationSettingsSnapshot(
        positive_prompt="p",
        negative_prompt="n",
        seed=1,
        width=1024,
        height=1024,
        steps=20,
        cfg_scale=5.0,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="model.safetensors",
        vae_name=None,
        workflow_template_id="sdxl_txt2img",
        workflow_template_version="1",
    )
    return Generation(
        id=uuid4(),
        kind=GenerationKind.STANDARD,
        status=status,
        parent_generation_id=None,
        settings_snapshot=snapshot,
        workflow_template_id="sdxl_txt2img",
        workflow_template_version="1",
        comfy_prompt_id="prompt",
        favorite=False,
        user_note=None,
        error_code=None,
        error_summary=None,
        created_at=timestamp,
        started_at=timestamp,
        completed_at=(
            timestamp if status in {GenerationStatus.COMPLETED, GenerationStatus.FAILED} else None
        ),
        updated_at=timestamp,
    )


def _settings() -> GenerationSettings:
    return GenerationSettings(
        positive_prompt="p",
        negative_prompt="n",
        seed=1,
        width=1024,
        height=1024,
        steps=20,
        cfg_scale=5.0,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="model.safetensors",
    )


def _service(
    generations: tuple[Generation, ...] = (),
    *,
    state_changed_callback: object | None = None,
    state_sync: FakeStateSync | None = None,
) -> tuple[
    PodLifecycleService,
    FakeRunPod,
    FakeStateSync,
]:
    generation = generations[0] if generations else _generation()
    record = SimpleNamespace(
        status=DriveSyncStatus.SYNCED,
        remote_image_path="2026-08-10/generated/image.png",
        remote_name="gdrive",
        remote_base_path="studio",
    )
    state_sync = state_sync or FakeStateSync()
    runpod = FakeRunPod()
    service = PodLifecycleService(
        FakeLifecycleRepository(),
        FakeGenerationRepository(generations or (generation,)),
        FakeQueueRepository(),
        FakeModelRepository(),
        FakeDriveRepository(record),
        state_sync,
        runpod,
        settings=SimpleNamespace(auto_terminate_enabled=True),
        comfyui_queue_provider=lambda: _empty_comfy_queue(),
        now_factory=lambda: datetime(2026, 8, 10, 12, tzinfo=UTC),
        state_changed_callback=state_changed_callback,
    )
    service.initialize_session()
    return service, runpod, state_sync


async def _empty_comfy_queue() -> ComfyUIQueueStatus:
    return ComfyUIQueueStatus((), ())


@pytest.mark.asyncio
async def test_readiness_requires_current_completed_generation_and_all_idle_boundaries() -> None:
    service, _, _ = _service()
    service.arm_on_generation_enqueue()
    readiness = await service.check_readiness()
    assert readiness.is_safe
    assert readiness.block_reasons == ()


@pytest.mark.asyncio
async def test_failed_generation_blocks_even_when_another_generation_completed() -> None:
    service, runpod, _ = _service((_generation(), _generation(GenerationStatus.FAILED)))
    service.arm_on_generation_enqueue()
    readiness = await service.check_readiness()
    assert not readiness.is_safe
    assert "generation_failed" in readiness.block_reasons
    assert runpod.calls == 0


@pytest.mark.asyncio
async def test_cancelled_generation_blocks_termination_readiness() -> None:
    service, _, _ = _service((_generation(), _generation(GenerationStatus.CANCELLED)))
    service.arm_on_generation_enqueue()
    readiness = await service.check_readiness()
    assert not readiness.is_safe
    assert "generation_cancelled" in readiness.block_reasons


@pytest.mark.asyncio
async def test_historical_failed_manifest_does_not_block_after_latest_sync() -> None:
    service, runpod, _ = _service()
    service._drive_sync_repository.manifest_jobs.append(  # type: ignore[attr-defined]
        SimpleNamespace(status=DriveSyncStatus.FAILED)
    )
    service.arm_on_generation_enqueue()
    readiness = await service.check_readiness()
    assert readiness.is_safe
    assert "manifest_failed" not in readiness.block_reasons
    assert runpod.calls == 0


@pytest.mark.asyncio
async def test_current_required_failed_manifest_still_blocks_readiness() -> None:
    service, runpod, _ = _service()
    service._drive_sync_repository.manifest_state = DriveManifestState.FAILED  # type: ignore[attr-defined]
    service.arm_on_generation_enqueue()
    readiness = await service.check_readiness()
    assert not readiness.is_safe
    assert "manifest_failed" in readiness.block_reasons
    assert runpod.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "reason"),
    [
        ("comfyui", "comfyui_queue_active"),
        ("model_transfer", "model_transfer_active"),
        ("drive_sync", "drive_sync_active"),
        ("manifest", "manifest_active"),
        ("state_backup", "state_backup_dirty"),
    ],
)
async def test_each_readiness_boundary_fails_closed(
    boundary: str,
    reason: str,
) -> None:
    service, runpod, _ = _service()
    if boundary == "comfyui":

        async def active_comfy_queue() -> ComfyUIQueueStatus:
            return ComfyUIQueueStatus(("prompt",), ())

        service._comfyui_queue_provider = active_comfy_queue  # type: ignore[attr-defined]
    elif boundary == "model_transfer":
        service._model_transfer_repository.counts = {  # type: ignore[attr-defined]
            ModelTransferStatus.DOWNLOADING: 1
        }
    elif boundary == "drive_sync":
        service._drive_sync_repository.counts = {  # type: ignore[attr-defined]
            DriveSyncStatus.SYNCING: 1
        }
    elif boundary == "manifest":
        service._drive_sync_repository.manifest_jobs.append(  # type: ignore[attr-defined]
            SimpleNamespace(status=DriveSyncStatus.SYNCING)
        )
    else:
        service._state_sync_service.is_clean = False  # type: ignore[attr-defined]
    service.arm_on_generation_enqueue()
    readiness = await service.check_readiness()
    assert not readiness.is_safe
    assert reason in readiness.block_reasons
    assert runpod.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("job_status", "submission_state"),
    [
        (GenerationStatus.RUNNING, SubmissionState.SUBMITTED),
        (GenerationStatus.COMPLETED, SubmissionState.SUBMITTING),
    ],
)
async def test_job_and_submission_projections_block_readiness(
    job_status: GenerationStatus,
    submission_state: SubmissionState,
) -> None:
    service, runpod, _ = _service()
    current = service._generation_repository.generations[0]  # type: ignore[attr-defined]
    service._dispatch_queue_repository.items.append(  # type: ignore[attr-defined]
        SimpleNamespace(
            generation=current,
            job=SimpleNamespace(status=job_status),
            entry=SimpleNamespace(submission_state=submission_state),
        )
    )
    service.arm_on_generation_enqueue()
    readiness = await service.check_readiness()
    assert not readiness.is_safe
    assert "generation_work_active" in readiness.block_reasons
    assert runpod.calls == 0


def test_same_pod_reuses_arm_but_new_pod_starts_unarmed() -> None:
    repository = FakeLifecycleRepository(sessions={})
    runpod = FakeRunPod()
    service = PodLifecycleService(
        repository,
        FakeGenerationRepository((_generation(),)),
        FakeQueueRepository(),
        FakeModelRepository(),
        FakeDriveRepository(
            SimpleNamespace(
                status=DriveSyncStatus.SYNCED,
                remote_image_path="2026-08-10/generated/image.png",
                remote_name="gdrive",
                remote_base_path="studio",
            )
        ),
        FakeStateSync(),
        runpod,
        settings=SimpleNamespace(auto_terminate_enabled=True),
        now_factory=lambda: datetime(2026, 8, 10, 12, tzinfo=UTC),
    )
    first = service.initialize_session()
    assert first is not None
    service.arm_on_generation_enqueue()
    same = service.initialize_session()
    assert same is not None
    assert same.id == first.id
    assert same.is_armed

    runpod.pod_id = "pod-new"
    fresh = service.initialize_session()
    assert fresh is not None
    assert fresh.id != first.id
    assert not fresh.is_armed


@pytest.mark.asyncio
async def test_grace_and_final_backup_terminate_once() -> None:
    service, runpod, state_sync = _service()
    service.arm_on_generation_enqueue()
    settings = SimpleNamespace(
        auto_terminate_grace_seconds=0.0,
        auto_terminate_check_interval_seconds=1.0,
    )
    coordinator = AutoTerminateCoordinator(
        service,
        settings,
        now_factory=lambda: datetime(2026, 8, 10, 12, tzinfo=UTC),
    )
    await coordinator.run_once()
    await coordinator.run_once()
    assert runpod.calls == 1
    assert state_sync.backup_calls == 1

    # A second worker tick cannot issue another DELETE after REQUESTING.
    await coordinator.run_once()
    assert runpod.calls == 1


@pytest.mark.asyncio
async def test_waiting_readiness_failure_resets_grace_to_armed() -> None:
    service, _, state_sync = _service()
    service.arm_on_generation_enqueue()
    coordinator = AutoTerminateCoordinator(
        service,
        SimpleNamespace(
            auto_terminate_grace_seconds=30.0,
            auto_terminate_check_interval_seconds=1.0,
        ),
    )

    await coordinator.run_once()
    assert service.session is not None
    assert service.session.status is AutoTerminateState.WAITING
    state_sync.is_clean = False

    readiness = await coordinator.run_once()

    assert readiness is not None and not readiness.is_safe
    assert service.session.status is AutoTerminateState.ARMED


@pytest.mark.asyncio
async def test_ready_readiness_failure_resets_grace_to_armed() -> None:
    service, _, state_sync = _service()
    service.arm_on_generation_enqueue()
    coordinator = AutoTerminateCoordinator(
        service,
        SimpleNamespace(
            auto_terminate_grace_seconds=30.0,
            auto_terminate_check_interval_seconds=1.0,
        ),
    )

    await coordinator.run_once()
    await coordinator.run_once()
    assert service.session is not None
    assert service.session.status is AutoTerminateState.READY
    state_sync.is_clean = False

    readiness = await coordinator.run_once()

    assert readiness is not None and not readiness.is_safe
    assert service.session.status is AutoTerminateState.ARMED


@pytest.mark.asyncio
async def test_grace_restarts_after_real_favorite_and_note_mutations() -> None:
    lifecycle, runpod, state_sync = _service()
    current = lifecycle._generation_repository.generations[0]  # type: ignore[attr-defined]

    class MutableGenerationRepository:
        def __init__(self) -> None:
            self.generation = current

        def get_by_id(self, generation_id: object) -> Generation | None:
            return self.generation if generation_id == self.generation.id else None

        def set_favorite(self, generation_id: object, favorite: bool) -> Generation:
            assert generation_id == self.generation.id
            self.generation = replace(self.generation, favorite=favorite)
            return self.generation

        def update_note(self, generation_id: object, note: str | None) -> Generation:
            assert generation_id == self.generation.id
            self.generation = replace(self.generation, user_note=note)
            return self.generation

    class EmptyArtifacts:
        def list_by_generation(self, generation_id: object) -> tuple[object, ...]:
            del generation_id
            return ()

    history = GenerationHistoryService(
        MutableGenerationRepository(),  # type: ignore[arg-type]
        EmptyArtifacts(),  # type: ignore[arg-type]
        Settings(_env_file=None),
        state_changed_callback=state_sync.mark_dirty,
        work_gate=lifecycle,
    )
    lifecycle.arm_on_generation_enqueue()
    clock = [datetime(2026, 8, 10, 12, tzinfo=UTC)]
    coordinator = AutoTerminateCoordinator(
        lifecycle,
        SimpleNamespace(
            auto_terminate_grace_seconds=5.0,
            auto_terminate_check_interval_seconds=1.0,
        ),
        now_factory=lambda: clock[0],
    )

    await coordinator.run_once()
    history.set_favorite(current.id, True)
    history.update_note(current.id, "kept")
    assert state_sync.mark_dirty_calls == 2
    assert not state_sync.is_clean
    await coordinator.run_once()
    assert lifecycle.session is not None
    assert lifecycle.session.status is AutoTerminateState.ARMED

    state_sync.is_clean = True
    await coordinator.run_once()
    assert lifecycle.session.status is AutoTerminateState.WAITING
    clock[0] += timedelta(seconds=1)
    await coordinator.run_once()
    assert lifecycle.session.status is AutoTerminateState.READY
    clock[0] += timedelta(seconds=5)
    await coordinator.run_once()

    assert lifecycle.session.status is AutoTerminateState.TERMINATION_REQUESTING
    assert runpod.calls == 1
    assert state_sync.backup_calls == 1


@pytest.mark.asyncio
async def test_confirmed_termination_failure_allows_work_and_manual_retry() -> None:
    service, _, _ = _service()
    failing_runpod = FailingRunPod(ambiguous=False)
    service._runpod_adapter = failing_runpod  # type: ignore[attr-defined]
    service.arm_on_generation_enqueue()

    with pytest.raises(RunPodTerminateError):
        await service.manual_drain_backup_and_terminate()
    assert service.session is not None
    assert service.session.status is AutoTerminateState.TERMINATION_FAILED
    error_code = service.session.last_error_code

    repository = BlockingDispatchRepository()
    repository.release.set()
    queue = GenerationQueueService(
        repository,
        Settings(_env_file=None),
        lifecycle_gate=service,
        generation_enqueued_callback=service.arm_on_generation_enqueue,
    )
    queue.enqueue(_settings())
    assert repository.persisted == 1
    assert service.session.status is AutoTerminateState.TERMINATION_FAILED

    coordinator = AutoTerminateCoordinator(
        service,
        SimpleNamespace(
            auto_terminate_grace_seconds=0.0,
            auto_terminate_check_interval_seconds=1.0,
        ),
    )
    await coordinator.run_once()
    assert failing_runpod.calls == 1

    with pytest.raises(RunPodTerminateError):
        await service.manual_drain_backup_and_terminate()
    assert failing_runpod.calls == 2
    assert service.session.status is AutoTerminateState.TERMINATION_FAILED

    updated = service.set_auto_terminate_enabled(False)
    assert updated is not None
    assert updated.status is AutoTerminateState.IDLE
    assert updated.last_error_code == error_code
    updated = service.set_auto_terminate_enabled(True)
    assert updated is not None
    assert updated.status is AutoTerminateState.ARMED
    assert updated.last_error_code == error_code


@pytest.mark.asyncio
async def test_ambiguous_termination_failure_keeps_work_frozen_and_no_duplicate_delete() -> None:
    service, _, _ = _service()
    failing_runpod = FailingRunPod(ambiguous=True)
    service._runpod_adapter = failing_runpod  # type: ignore[attr-defined]
    service.arm_on_generation_enqueue()

    with pytest.raises(RunPodTerminateError):
        await service.manual_drain_backup_and_terminate()
    assert service.session is not None
    assert service.session.status is AutoTerminateState.TERMINATION_AMBIGUOUS
    with pytest.raises(PodLifecycleWorkBlockedError):
        service.ensure_work_allowed()

    repository = BlockingDispatchRepository()
    repository.release.set()
    queue = GenerationQueueService(
        repository,
        Settings(_env_file=None),
        lifecycle_gate=service,
    )
    with pytest.raises(GenerationQueueServiceError):
        queue.enqueue(_settings())
    assert repository.persisted == 0

    readiness = await service.manual_drain_backup_and_terminate()
    assert not readiness.is_safe
    assert failing_runpod.calls == 1


@pytest.mark.asyncio
async def test_manual_terminate_uses_same_drain_backup_path() -> None:
    service, runpod, state_sync = _service()
    service.arm_on_generation_enqueue()
    readiness = await service.drain_backup_and_terminate(require_armed=False)
    assert readiness.is_safe
    assert state_sync.backup_calls == 1
    assert runpod.calls == 1


@pytest.mark.asyncio
async def test_adapter_terminated_confirmation_completes_service_without_reopening_work() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "DELETE":
            return httpx.Response(503)
        return httpx.Response(200, json={"desiredStatus": "TERMINATED"})

    async with httpx.AsyncClient(
        base_url=RUNPOD_API_BASE_URL,
        transport=httpx.MockTransport(handler),
    ) as client:
        service, _, state_sync = _service()
        service._runpod_adapter = RunPodLifecycleAdapter(  # type: ignore[attr-defined]
            client=client,
            env={"RUNPOD_POD_ID": "pod-current", "RUNPOD_API_KEY": "secret"},
        )
        service.arm_on_generation_enqueue()

        readiness = await service.manual_drain_backup_and_terminate()

    assert readiness.is_safe
    assert calls == ["DELETE", "GET"]
    assert state_sync.backup_calls == 1
    assert service.session is not None
    assert service.session.status is AutoTerminateState.TERMINATION_REQUESTING


@pytest.mark.asyncio
async def test_adapter_delivery_ambiguity_enters_hard_freeze_without_duplicate_delete() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "DELETE":
            raise httpx.ReadError("connection lost", request=request)
        return httpx.Response(200, json={"desiredStatus": "RUNNING"})

    async with httpx.AsyncClient(
        base_url=RUNPOD_API_BASE_URL,
        transport=httpx.MockTransport(handler),
    ) as client:
        service, _, _ = _service()
        service._runpod_adapter = RunPodLifecycleAdapter(  # type: ignore[attr-defined]
            client=client,
            env={"RUNPOD_POD_ID": "pod-current", "RUNPOD_API_KEY": "secret"},
        )
        service.arm_on_generation_enqueue()

        with pytest.raises(RunPodTerminateError) as raised:
            await service.manual_drain_backup_and_terminate()
        assert raised.value.ambiguous
        assert service.session is not None
        assert service.session.status is AutoTerminateState.TERMINATION_AMBIGUOUS

        readiness = await service.manual_drain_backup_and_terminate()

    assert not readiness.is_safe
    assert calls == ["DELETE", "GET"]


@pytest.mark.asyncio
async def test_manual_terminate_checks_readiness_before_entering_draining() -> None:
    service, runpod, state_sync = _service((_generation(GenerationStatus.PENDING),))
    service.arm_on_generation_enqueue()

    readiness = await service.manual_drain_backup_and_terminate()

    assert not readiness.is_safe
    assert "generation_not_completed" in readiness.block_reasons
    assert service.session is not None
    assert service.session.status is AutoTerminateState.ARMED
    assert state_sync.backup_calls == 0
    assert runpod.calls == 0


def test_transient_lifecycle_updates_use_compare_and_set_transitions() -> None:
    service, _, _ = _service()
    service.arm_on_generation_enqueue()

    assert not service.set_transient_state(AutoTerminateState.READY)
    assert service.session is not None
    assert service.session.status is AutoTerminateState.ARMED
    assert service.set_transient_state(AutoTerminateState.WAITING)
    service.begin_draining()
    assert not service.set_transient_state(AutoTerminateState.READY)
    assert service.session.status is AutoTerminateState.DRAINING
    with pytest.raises(PodLifecycleTransitionError):
        service.set_transient_state(AutoTerminateState.ARMED)


@pytest.mark.asyncio
async def test_auto_terminate_toggle_cannot_reopen_draining_or_requesting() -> None:
    service, _, _ = _service()
    service.arm_on_generation_enqueue()
    service.begin_draining()
    updated = service.set_auto_terminate_enabled(False)
    assert updated is not None
    assert updated.auto_terminate_enabled
    assert updated.status is AutoTerminateState.DRAINING

    await service.request_terminate()
    updated = service.set_auto_terminate_enabled(False)
    assert updated is not None
    assert updated.auto_terminate_enabled
    assert updated.status is AutoTerminateState.TERMINATION_REQUESTING


@pytest.mark.asyncio
async def test_termination_safety_uses_unbounded_active_generation_and_manifest_checks() -> None:
    service, _, _ = _service()
    service.arm_on_generation_enqueue()
    current = service._generation_repository.generations[0]  # type: ignore[attr-defined]
    queue = service._dispatch_queue_repository  # type: ignore[attr-defined]
    queue.items.append(  # type: ignore[attr-defined]
        SimpleNamespace(
            generation=current,
            job=SimpleNamespace(status=GenerationStatus.RUNNING),
            entry=SimpleNamespace(submission_state=SubmissionState.SUBMITTED),
        )
    )
    queue.list_queue = lambda **_: (_ for _ in ()).throw(AssertionError("bounded queue lookup"))  # type: ignore[attr-defined]
    readiness = await service.check_readiness()
    assert not readiness.is_safe
    assert "generation_work_active" in readiness.block_reasons

    service2, _, _ = _service()
    service2.arm_on_generation_enqueue()
    drive = service2._drive_sync_repository  # type: ignore[attr-defined]
    drive.manifest_jobs.append(SimpleNamespace(status=DriveSyncStatus.SYNCING))  # type: ignore[attr-defined]
    drive.list_manifest_jobs = lambda **_: (_ for _ in ()).throw(  # type: ignore[attr-defined]
        AssertionError("bounded manifest lookup")
    )
    readiness = await service2.check_readiness()
    assert not readiness.is_safe
    assert "manifest_active" in readiness.block_reasons


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "late_status",
    [GenerationStatus.FAILED, GenerationStatus.CANCELLED],
)
async def test_current_session_safety_does_not_truncate_terminal_generations(
    late_status: GenerationStatus,
) -> None:
    generations = tuple(_generation() for _ in range(5000)) + (_generation(late_status),)
    service, _, _ = _service(generations)
    repository = service._generation_repository  # type: ignore[attr-defined]
    repository.list_since = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[attr-defined]
        AssertionError("bounded current-session lookup")
    )
    service.arm_on_generation_enqueue()

    readiness = await service.check_readiness()

    assert not readiness.is_safe
    assert (
        "generation_failed" if late_status is GenerationStatus.FAILED else "generation_cancelled"
    ) in readiness.block_reasons


@pytest.mark.asyncio
async def test_current_session_safety_does_not_truncate_unsynced_completed_generation() -> None:
    generations = tuple(_generation() for _ in range(5001))
    service, _, _ = _service(generations)
    repository = service._generation_repository  # type: ignore[attr-defined]
    repository.list_since = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[attr-defined]
        AssertionError("bounded current-session lookup")
    )
    drive = service._drive_sync_repository  # type: ignore[attr-defined]
    drive.missing_generation_id = generations[-1].id
    service.arm_on_generation_enqueue()

    readiness = await service.check_readiness()

    assert not readiness.is_safe
    assert "drive_not_synced" in readiness.block_reasons


def test_persistent_mutation_admission_is_closed_after_draining() -> None:
    service, _, _ = _service()
    service.arm_on_generation_enqueue()
    committed: list[str] = []
    with service.admit_persistent_mutation():
        committed.append("before-drain")
    service.begin_draining()
    with pytest.raises(PodLifecycleWorkBlockedError), service.admit_persistent_mutation():
        committed.append("after-drain")
    assert committed == ["before-drain"]


def test_generation_favorite_and_note_mutations_share_drain_admission(tmp_path: Path) -> None:
    lifecycle, _, state_sync = _service()
    current = lifecycle._generation_repository.generations[0]  # type: ignore[attr-defined]

    class MutableGenerationRepository:
        def __init__(self) -> None:
            self.generation = current
            self.writes = 0

        def get_by_id(self, generation_id: object) -> Generation | None:
            return self.generation if self.generation.id == generation_id else None

        def set_favorite(self, generation_id: object, favorite: bool) -> Generation:
            assert generation_id == self.generation.id
            self.writes += 1
            self.generation = replace(self.generation, favorite=favorite)
            return self.generation

        def update_note(self, generation_id: object, note: str | None) -> Generation:
            assert generation_id == self.generation.id
            self.writes += 1
            self.generation = replace(self.generation, user_note=note)
            return self.generation

    class EmptyArtifacts:
        def list_by_generation(self, generation_id: object) -> tuple[object, ...]:
            del generation_id
            return ()

    repository = MutableGenerationRepository()
    history = GenerationHistoryService(
        repository,  # type: ignore[arg-type]
        EmptyArtifacts(),  # type: ignore[arg-type]
        Settings(_env_file=None, data_dir=tmp_path),
        state_changed_callback=state_sync.mark_dirty,
        work_gate=lifecycle,
    )
    history.set_favorite(current.id, True)
    assert repository.writes == 1
    assert state_sync.mark_dirty_calls == 1

    lifecycle.begin_draining()
    with pytest.raises(PodLifecycleWorkBlockedError):
        history.update_note(current.id, "blocked")
    assert repository.writes == 1
    assert repository.generation.user_note is None


@pytest.mark.asyncio
@pytest.mark.parametrize("grace_seconds", [0.0, 15.0])
async def test_lifecycle_transient_states_do_not_mark_state_sync_dirty(
    grace_seconds: float,
) -> None:
    state_sync = FakeStateSync()
    service, runpod, _ = _service(
        state_changed_callback=state_sync.mark_dirty,
        state_sync=state_sync,
    )
    service.arm_on_generation_enqueue()
    coordinator = AutoTerminateCoordinator(
        service,
        SimpleNamespace(
            auto_terminate_grace_seconds=grace_seconds,
            auto_terminate_check_interval_seconds=1.0,
        ),
        now_factory=lambda: datetime(2026, 8, 10, 12, tzinfo=UTC),
    )
    await coordinator.run_once()
    await coordinator.run_once()
    assert state_sync.mark_dirty_calls == 0
    assert state_sync.is_clean
    if grace_seconds == 0.0:
        assert runpod.calls == 1
    else:
        assert runpod.calls == 0


@pytest.mark.asyncio
async def test_auto_not_ready_does_not_abort_manual_draining_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, runpod, _ = _service()
    service.arm_on_generation_enqueue()
    original_check_readiness = service.check_readiness

    async def racing_check() -> object:
        readiness = await original_check_readiness()
        service.begin_draining()
        return replace(readiness, is_safe=False)

    monkeypatch.setattr(service, "check_readiness", racing_check)
    coordinator = AutoTerminateCoordinator(
        service,
        SimpleNamespace(
            auto_terminate_grace_seconds=0.0,
            auto_terminate_check_interval_seconds=1.0,
        ),
    )

    await coordinator.run_once()

    assert service.session is not None
    assert service.session.status is AutoTerminateState.DRAINING
    assert runpod.calls == 0


def test_enqueue_admission_wins_race_with_draining_without_losing_generation() -> None:
    lifecycle, _, _ = _service()
    repository = BlockingDispatchRepository()
    queue = GenerationQueueService(
        repository,
        Settings(_env_file=None),
        lifecycle_gate=lifecycle,
        generation_enqueued_callback=lifecycle.arm_on_generation_enqueue,
    )
    errors: list[BaseException] = []

    def enqueue() -> None:
        try:
            queue.enqueue(_settings())
        except BaseException as exc:  # pragma: no cover - assertion below reports races
            errors.append(exc)

    enqueue_thread = threading.Thread(
        target=enqueue,
        daemon=True,
    )
    enqueue_thread.start()
    assert repository.entered.wait(timeout=2.0)

    drain_thread = threading.Thread(target=lifecycle.begin_draining, daemon=True)
    drain_thread.start()
    assert drain_thread.is_alive()
    repository.release.set()
    enqueue_thread.join(timeout=2.0)
    drain_thread.join(timeout=2.0)
    assert not errors
    assert repository.persisted == 1
    assert lifecycle.session is not None
    assert lifecycle.session.status is AutoTerminateState.DRAINING


def test_draining_wins_before_enqueue_and_repository_is_not_called() -> None:
    lifecycle, _, _ = _service()
    repository = BlockingDispatchRepository()
    queue = GenerationQueueService(
        repository,
        Settings(_env_file=None),
        lifecycle_gate=lifecycle,
        generation_enqueued_callback=lifecycle.arm_on_generation_enqueue,
    )
    lifecycle.begin_draining()
    with pytest.raises(GenerationQueueServiceError):
        queue.enqueue(_settings())
    assert repository.persisted == 0


def test_batch_enqueue_is_rejected_before_repository_after_draining() -> None:
    lifecycle, _, _ = _service()
    repository = BlockingDispatchRepository()
    queue = GenerationQueueService(
        repository,
        Settings(_env_file=None),
        lifecycle_gate=lifecycle,
    )
    lifecycle.begin_draining()

    with pytest.raises(GenerationQueueServiceError):
        queue.enqueue_batch(
            _settings(),
            count=2,
            seed_strategy="sequential",
            start_seed=1,
            seed_step=1,
            name="blocked batch",
        )

    assert repository.batch_persisted == 0


def test_upscale_enqueue_is_rejected_before_source_lookup_after_draining() -> None:
    lifecycle, _, _ = _service()
    service = UpscaleEnqueueService(
        object(),
        object(),
        object(),
        Settings(_env_file=None),
        work_gate=lifecycle,
    )
    lifecycle.begin_draining()

    with pytest.raises(UpscaleEnqueueError) as error:
        service.enqueue(uuid4(), object())  # type: ignore[arg-type]

    assert error.value.code == "pod_lifecycle_draining"


@pytest.mark.asyncio
async def test_model_preparation_is_rejected_before_catalog_lookup_after_draining() -> None:
    lifecycle, _, _ = _service()
    service = ModelPreparationService(
        object(),
        object(),
        Settings(_env_file=None, remote_model_enabled=True, rclone_remote="drive"),
        lambda: None,  # type: ignore[arg-type]
        work_gate=lifecycle,
    )
    lifecycle.begin_draining()

    with pytest.raises(ModelPreparationServiceError) as error:
        await service.prepare_selected("checkpoints/model.safetensors", None, (), None)

    assert error.value.code == "pod_lifecycle_draining"


def test_drive_manifest_rebuild_is_rejected_before_repository_after_draining() -> None:
    lifecycle, _, _ = _service()
    service = DriveSyncService(
        object(),
        object(),
        object(),
        Settings(_env_file=None, rclone_remote="drive"),
        adapter=object(),
        work_gate=lifecycle,
    )
    lifecycle.begin_draining()

    with pytest.raises(DriveSyncServiceError) as error:
        service.enqueue_manifest_rebuild("2026-08-10")

    assert error.value.code == "pod_lifecycle_draining"


def test_manual_terminate_initial_not_ready_does_not_enter_draining() -> None:
    pending = _generation(GenerationStatus.PENDING)
    lifecycle, runpod, state_sync = _service((pending,))
    repository = BlockingDispatchRepository(pending)
    queue = GenerationQueueService(
        repository,
        Settings(_env_file=None),
        lifecycle_gate=lifecycle,
        generation_enqueued_callback=lifecycle.arm_on_generation_enqueue,
    )
    enqueue_errors: list[BaseException] = []
    terminate_results: list[object] = []
    terminate_errors: list[BaseException] = []
    terminate_started = threading.Event()

    def enqueue() -> None:
        try:
            queue.enqueue(_settings())
        except BaseException as exc:  # pragma: no cover - assertion below reports races
            enqueue_errors.append(exc)

    def terminate() -> None:
        terminate_started.set()
        try:
            terminate_results.append(asyncio.run(lifecycle.manual_drain_backup_and_terminate()))
        except BaseException as exc:  # pragma: no cover - assertion below reports races
            terminate_errors.append(exc)

    enqueue_thread = threading.Thread(target=enqueue, daemon=True)
    terminate_thread = threading.Thread(target=terminate, daemon=True)
    enqueue_thread.start()
    assert repository.entered.wait(timeout=2.0)
    terminate_thread.start()
    assert terminate_started.wait(timeout=2.0)
    repository.release.set()
    enqueue_thread.join(timeout=2.0)
    terminate_thread.join(timeout=2.0)

    assert not enqueue_errors
    assert not terminate_errors
    assert len(terminate_results) == 1
    assert not getattr(terminate_results[0], "is_safe", True)
    assert repository.persisted == 1
    assert state_sync.backup_calls == 0
    assert runpod.calls == 0
    assert lifecycle.session is not None
    assert lifecycle.session.status is AutoTerminateState.ARMED


@pytest.mark.asyncio
async def test_auto_terminate_waits_for_startup_restore_gate() -> None:
    service, runpod, _ = _service()
    service.arm_on_generation_enqueue()
    startup_ready = False
    coordinator = AutoTerminateCoordinator(
        service,
        SimpleNamespace(
            auto_terminate_grace_seconds=0.0,
            auto_terminate_check_interval_seconds=1.0,
        ),
        startup_restore_ready=lambda: startup_ready,
    )
    assert await coordinator.run_once() is None
    assert runpod.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["dirty", "failed"])
async def test_final_backup_race_or_failure_aborts_termination(mode: str) -> None:
    service, runpod, state_sync = _service()
    state_sync.dirty_on_backup = mode == "dirty"
    state_sync.fail_on_backup = mode == "failed"
    service.arm_on_generation_enqueue()
    coordinator = AutoTerminateCoordinator(
        service,
        SimpleNamespace(
            auto_terminate_grace_seconds=0.0,
            auto_terminate_check_interval_seconds=1.0,
        ),
    )
    await coordinator.run_once()
    await coordinator.run_once()
    assert state_sync.backup_calls == 1
    assert runpod.calls == 0


@pytest.mark.asyncio
async def test_new_generation_during_grace_cancels_countdown_and_draining_blocks_work() -> None:
    service, runpod, _ = _service()
    service.arm_on_generation_enqueue()
    settings = SimpleNamespace(
        auto_terminate_grace_seconds=15.0,
        auto_terminate_check_interval_seconds=1.0,
    )
    coordinator = AutoTerminateCoordinator(service, settings)
    await coordinator.run_once()
    assert service.session is not None
    assert service.session.status is AutoTerminateState.WAITING
    service.arm_on_generation_enqueue()
    await coordinator.run_once()
    assert runpod.calls == 0
    service.begin_draining()
    with pytest.raises(PodLifecycleWorkBlockedError):
        service.ensure_work_allowed()
