from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUIQueueStatus
from runpod_sdxl_image_studio.adapters.runpod.pod_lifecycle import (
    RunPodIdentity,
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


class FakeQueueRepository:
    def __init__(self) -> None:
        self.items: list[object] = []

    def list_queue(self, *, limit: int = 200, **_: object) -> tuple[object, ...]:
        return tuple(self.items[:limit])


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

    def status_counts(self) -> dict[DriveSyncStatus, int]:
        return self.counts

    def get_by_generation(self, generation_id: object) -> object:
        return self.record

    def list_manifest_jobs(self, limit: int = 50) -> tuple[object, ...]:
        return tuple(self.manifest_jobs[:limit])

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
async def test_manual_terminate_uses_same_drain_backup_path() -> None:
    service, runpod, state_sync = _service()
    service.arm_on_generation_enqueue()
    readiness = await service.drain_backup_and_terminate(require_armed=False)
    assert readiness.is_safe
    assert state_sync.backup_calls == 1
    assert runpod.calls == 1


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


def test_manual_terminate_waits_for_enqueue_admission_and_aborts_on_pending_generation() -> None:
    pending = _generation(GenerationStatus.PENDING)
    lifecycle, runpod, _ = _service((pending,))
    repository = BlockingDispatchRepository(pending)
    queue = GenerationQueueService(
        repository,
        Settings(_env_file=None),
        lifecycle_gate=lifecycle,
        generation_enqueued_callback=lifecycle.arm_on_generation_enqueue,
    )
    enqueue_errors: list[BaseException] = []
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
            asyncio.run(lifecycle.drain_backup_and_terminate())
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
    assert terminate_errors
    assert repository.persisted == 1
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
