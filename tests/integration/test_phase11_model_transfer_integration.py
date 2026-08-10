"""Phase 11 worker and model-transfer integration coverage."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image
from sqlalchemy import create_engine

from runpod_sdxl_image_studio.adapters.catalog.upscaler_catalog import UpscalerCatalog
from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepository,
    GenerationCompletionRepository,
    GenerationRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_start_repository import (
    GenerationStartRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.model_transfer_repository import (
    ModelTransferRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.upscale_settings_repository import (
    UpscaleSettingsRepository,
)
from runpod_sdxl_image_studio.adapters.rclone.remote_model_catalog import (
    RemoteModelAdapterError,
)
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationKind
from runpod_sdxl_image_studio.domain.generation_artifact import ArtifactType, GenerationArtifact
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot
from runpod_sdxl_image_studio.domain.model_transfer import (
    ModelTransferErrorCode,
    ModelTransferProgress,
    ModelTransferStatus,
    RemoteModelCatalog,
    RemoteModelEntry,
    RemoteModelKind,
)
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.domain.upscale import (
    UpscaleMethod,
    UpscaleSettings,
    UpscaleSizingMode,
)
from runpod_sdxl_image_studio.jobs.model_transfer_worker import ModelTransferWorker
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.services.model_preparation_service import (
    ModelPreparationService,
)
from runpod_sdxl_image_studio.services.upscale_enqueue_service import UpscaleEnqueueService


def _fixture(tmp_path: Path, *, visible: bool = True, upscale_nodes: bool = False):
    payload = b"phase-11-integration-model"
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        database_url=f"sqlite:///{tmp_path / 'image_studio.sqlite3'}",
        rclone_remote="drive",
        remote_model_enabled=True,
        checkpoint_dir=tmp_path / "checkpoints",
        lora_dir=tmp_path / "loras",
        vae_dir=tmp_path / "vae",
        upscaler_dir=tmp_path / "upscale_models",
    )
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    repository = ModelTransferRepository(create_session_factory(engine))
    entry = RemoteModelEntry(
        RemoteModelKind.UPSCALER,
        "realesrgan/4x.pth",
        "realesrgan/4x.pth",
        len(payload),
        datetime.now(UTC),
        "sha-256",
        hashlib.sha256(payload).hexdigest(),
    )
    capabilities = ComfyUICapabilities(
        checkpoints=(),
        vaes=(),
        samplers=(),
        schedulers=(),
        loras=(),
        upscale_models=(entry.relative_path,) if visible else (),
        available_node_classes=(
            frozenset(
                {
                    "LoadImage",
                    "UpscaleModelLoader",
                    "ImageUpscaleWithModel",
                    "ImageScale",
                    "SaveImage",
                }
            )
            if upscale_nodes
            else frozenset()
        ),
        warnings=(),
    )

    async def refresh() -> CapabilityRefreshResult:
        return CapabilityRefreshResult(True, "ok", capabilities)

    adapter = IntegrationModelAdapter(payload, RemoteModelCatalog((entry,), datetime.now(UTC)))
    service = ModelPreparationService(repository, adapter, settings, refresh)
    return settings, repository, service, adapter, entry, payload, engine


class IntegrationModelAdapter:
    def __init__(self, payload: bytes, catalog: RemoteModelCatalog) -> None:
        self.payload = payload
        self.catalog = catalog
        self.download_calls = 0
        self.fail_download = False
        self.write_wrong_payload = False
        self.wait_for_cancel = False
        self.started = asyncio.Event()
        self.thread_started = threading.Event()

    async def list_catalog(self) -> RemoteModelCatalog:
        return self.catalog

    async def download(self, entry, destination, **kwargs) -> None:
        self.download_calls += 1
        started = kwargs.get("process_started_callback")
        if started is not None:
            result = started(777)
            if asyncio.iscoroutine(result):
                await result
        self.started.set()
        self.thread_started.set()
        if self.wait_for_cancel:
            cancel_check = kwargs["cancel_check"]
            shutdown_check = kwargs.get("shutdown_check")
            while not await cancel_check():
                if shutdown_check is not None:
                    shutdown_result = shutdown_check()
                    if asyncio.iscoroutine(shutdown_result):
                        shutdown_result = await shutdown_result
                    if shutdown_result:
                        raise RemoteModelAdapterError(
                            ModelTransferErrorCode.APP_RESTART_INTERRUPTED.value,
                            "shutdown by integration test",
                        )
                await asyncio.sleep(0)
            raise RemoteModelAdapterError(
                ModelTransferErrorCode.CANCELLED.value,
                "cancelled by integration test",
            )
        if self.fail_download:
            raise RemoteModelAdapterError(
                ModelTransferErrorCode.DOWNLOAD_FAILED.value,
                "download failed in integration test",
            )
        progress = kwargs.get("progress_callback")
        if progress is not None:
            result = progress(ModelTransferProgress(entry.size_bytes, entry.size_bytes, 100.0))
            if asyncio.iscoroutine(result):
                await result
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.payload if not self.write_wrong_payload else b"wrong")
        finished = kwargs.get("process_finished_callback")
        if finished is not None:
            result = finished()
            if asyncio.iscoroutine(result):
                await result


def _worker(repository, service, settings, *, callback_count: list[int] | None = None):
    return ModelTransferWorker(
        repository,
        service,
        settings,
        worker_id="integration-worker",
        state_changed_callback=(
            (lambda: callback_count.append(1)) if callback_count is not None else None
        ),
    )


def test_worker_integration_downloads_verifies_and_completes_model(tmp_path: Path) -> None:
    settings, repository, service, adapter, entry, payload, engine = _fixture(tmp_path)
    callback_count: list[int] = []
    job = repository.enqueue(entry, entry.relative_path)
    worker = _worker(repository, service, settings, callback_count=callback_count)

    assert asyncio.run(worker.run_once()) is True

    completed = repository.get(job.id)
    assert completed is not None
    assert completed.status is ModelTransferStatus.COMPLETED, completed.error_code
    assert completed.progress_percentage == 100.0
    final_path = settings.upscaler_dir / entry.relative_path
    assert final_path.read_bytes() == payload
    assert not list(final_path.parent.glob(".*.download"))
    assert adapter.download_calls == 1
    assert callback_count
    engine.dispose()


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("download", ModelTransferErrorCode.DOWNLOAD_FAILED.value),
        ("hash", ModelTransferErrorCode.SIZE_MISMATCH.value),
        ("visibility", ModelTransferErrorCode.MODEL_NOT_VISIBLE.value),
    ],
)
def test_worker_integration_persists_failure_without_retrying_another_model(
    tmp_path: Path, failure: str, expected_code: str
) -> None:
    settings, repository, service, adapter, entry, _payload, engine = _fixture(
        tmp_path, visible=failure != "visibility"
    )
    if failure == "download":
        adapter.fail_download = True
    elif failure == "hash":
        adapter.write_wrong_payload = True
    job = repository.enqueue(entry, entry.relative_path)

    assert asyncio.run(_worker(repository, service, settings).run_once()) is True

    failed = repository.get(job.id)
    assert failed is not None
    assert failed.status is ModelTransferStatus.FAILED
    assert failed.error_code == expected_code
    assert adapter.download_calls == 1
    engine.dispose()


def test_worker_integration_cancel_stops_active_transfer_and_marks_cancelled(
    tmp_path: Path,
) -> None:
    settings, repository, service, adapter, entry, _payload, engine = _fixture(tmp_path)
    adapter.wait_for_cancel = True
    job = repository.enqueue(entry, entry.relative_path)
    worker = _worker(repository, service, settings)

    async def run_and_cancel() -> None:
        task = asyncio.create_task(worker.run_once())
        await adapter.started.wait()
        repository.request_cancel(job.id)
        assert await task is True

    asyncio.run(run_and_cancel())

    cancelled = repository.get(job.id)
    assert cancelled is not None
    assert cancelled.status is ModelTransferStatus.CANCELLED
    assert cancelled.error_code == ModelTransferErrorCode.CANCELLED.value
    engine.dispose()


def test_worker_stop_interrupts_active_transfer_without_leaving_worker_running(
    tmp_path: Path,
) -> None:
    settings, repository, service, adapter, entry, _payload, engine = _fixture(tmp_path)
    adapter.wait_for_cancel = True
    job = repository.enqueue(entry, entry.relative_path)
    worker = _worker(repository, service, settings)

    worker.start()
    assert adapter.thread_started.wait(timeout=5.0)
    worker.stop()

    interrupted = repository.get(job.id)
    assert interrupted is not None
    assert interrupted.status is ModelTransferStatus.FAILED
    assert interrupted.error_code == ModelTransferErrorCode.APP_RESTART_INTERRUPTED.value
    assert adapter.download_calls == 1
    assert worker._thread is not None
    assert worker._thread.is_alive() is False
    engine.dispose()


def test_worker_integration_retry_reuses_catalog_and_completes_after_failure(
    tmp_path: Path,
) -> None:
    settings, repository, service, adapter, entry, payload, engine = _fixture(tmp_path)
    adapter.fail_download = True
    job = repository.enqueue(entry, entry.relative_path)
    worker = _worker(repository, service, settings)
    assert asyncio.run(worker.run_once()) is True
    failed = repository.get(job.id)
    assert failed is not None
    assert failed.status is ModelTransferStatus.FAILED

    adapter.fail_download = False
    retried = asyncio.run(service.retry(job.id))
    assert retried.status is ModelTransferStatus.PENDING
    assert asyncio.run(worker.run_once()) is True

    completed = repository.get(retried.id)
    assert completed is not None
    assert completed.status is ModelTransferStatus.COMPLETED, completed.error_code
    assert (settings.upscaler_dir / entry.relative_path).read_bytes() == payload
    engine.dispose()


def test_same_session_upscaler_transfer_updates_enqueue_and_generation_preflight(
    tmp_path: Path,
) -> None:
    """A completed transfer is immediately visible to both upscale service boundaries."""

    settings, transfer_repository, transfer_service, adapter, entry, _payload, engine = _fixture(
        tmp_path, upscale_nodes=True
    )
    assert UpscalerCatalog.scan(settings.upscaler_dir).contains(entry.relative_path) is False

    created_at = datetime.now(UTC)
    parent_id, parent_job_id = uuid4(), uuid4()
    parent_settings = GenerationSettings(
        positive_prompt="a cat",
        negative_prompt="",
        seed=42,
        width=64,
        height=64,
        steps=1,
        cfg_scale=1.0,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="base.safetensors",
    )
    parent_snapshot = GenerationSettingsSnapshot.from_settings(parent_settings)
    factory = create_session_factory(engine)
    GenerationStartRepository(factory).create_pending(
        parent_snapshot,
        generation_id=parent_id,
        job_id=parent_job_id,
        kind=GenerationKind.STANDARD,
        parent_generation_id=None,
        created_at=created_at,
    )
    image_bytes = BytesIO()
    Image.new("RGB", (64, 64), "white").save(image_bytes, format="PNG")
    stored = LocalStorageAdapter(settings).store_image(
        image_bytes.getvalue(), parent_id, created_at
    )
    parent_artifact = GenerationArtifact(
        id=uuid4(),
        generation_id=parent_id,
        artifact_type=ArtifactType.IMAGE,
        local_path=LocalStorageAdapter(settings).relative_path(stored.path),
        sha256=stored.sha256,
        size_bytes=stored.size_bytes,
        width=stored.width,
        height=stored.height,
        mime_type=stored.mime_type,
        created_at=created_at,
    )
    GenerationCompletionRepository(factory).complete_generation(
        parent_id, parent_job_id, parent_artifact, created_at
    )

    upscale_settings = UpscaleSettings(
        method=UpscaleMethod.IMAGE,
        sizing_mode=UpscaleSizingMode.FACTOR,
        scale_factor=2,
        upscaler_name=entry.relative_path,
    )
    enqueue_service = UpscaleEnqueueService(
        GenerationRepository(factory),
        GenerationArtifactRepository(factory),
        GenerationDispatchQueueRepository(factory),
        settings,
        catalog_provider=lambda: UpscalerCatalog.scan(settings.upscaler_dir),
    )

    transfer_job = transfer_repository.enqueue(entry, entry.relative_path)
    worker = _worker(transfer_repository, transfer_service, settings)
    assert asyncio.run(worker.run_once()) is True
    completed_transfer = transfer_repository.get(transfer_job.id)
    assert completed_transfer is not None
    assert completed_transfer.status is ModelTransferStatus.COMPLETED
    assert adapter.download_calls == 1

    plan = enqueue_service.plan(parent_id, upscale_settings)
    assert plan.target_width == 128
    assert plan.target_height == 128
    queue_item = enqueue_service.enqueue(parent_id, upscale_settings)
    persisted_upscale = UpscaleSettingsRepository(factory).get_by_generation(
        queue_item.generation.id
    )
    assert persisted_upscale is not None

    async def capability_refresh() -> CapabilityRefreshResult:
        return CapabilityRefreshResult(
            True,
            "ok",
            ComfyUICapabilities(
                checkpoints=("base.safetensors",),
                vaes=(),
                samplers=("euler",),
                schedulers=("normal",),
                loras=(),
                upscale_models=(entry.relative_path,),
                available_node_classes=frozenset(
                    {
                        "LoadImage",
                        "UpscaleModelLoader",
                        "ImageUpscaleWithModel",
                        "ImageScale",
                        "SaveImage",
                    }
                ),
                warnings=(),
            ),
        )

    generation_service = GenerationService(
        object(),
        object(),
        object(),
        LocalStorageAdapter(settings),
        capability_refresh,
        settings,
        upscaler_catalog_provider=lambda: UpscalerCatalog.scan(settings.upscaler_dir),
    )
    asyncio.run(
        generation_service._preflight_upscale(  # noqa: SLF001 - final preflight regression boundary
            persisted_upscale,
            parent_snapshot,
        )
    )
    assert UpscalerCatalog.scan(settings.upscaler_dir).contains(entry.relative_path) is True
    engine.dispose()
