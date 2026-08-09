"""Phase 11 model catalog and transfer boundary tests."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base
from runpod_sdxl_image_studio.adapters.database.repositories.model_transfer_repository import (
    ModelTransferRepository,
)
from runpod_sdxl_image_studio.adapters.rclone.remote_model_catalog import (
    RemoteModelCatalogAdapter,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.model_transfer import (
    ModelTransferErrorCode,
    ModelTransferProgress,
    ModelTransferStatus,
    RemoteModelCatalog,
    RemoteModelEntry,
    RemoteModelKind,
)
from runpod_sdxl_image_studio.domain.system_status import CapabilityRefreshResult
from runpod_sdxl_image_studio.services.model_preparation_service import (
    ModelPreparationService,
    ModelPreparationServiceError,
)
from runpod_sdxl_image_studio.ui.app_builder import build_app


def _settings(tmp_path: Path) -> Settings:
    roots = {
        "checkpoint_dir": tmp_path / "checkpoints",
        "lora_dir": tmp_path / "loras",
        "vae_dir": tmp_path / "vae",
        "upscaler_dir": tmp_path / "upscale_models",
    }
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        database_url="sqlite:///:memory:",
        rclone_remote="drive",
        remote_model_enabled=True,
        **roots,
    )


def _fixture(tmp_path: Path, *, visible: bool = True):
    settings = _settings(tmp_path)
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    repository = ModelTransferRepository(create_session_factory(engine))
    payload = b"phase-11-model"
    entry = RemoteModelEntry(
        RemoteModelKind.CHECKPOINT,
        "nested/model.safetensors",
        "nested/model.safetensors",
        len(payload),
        datetime.now(UTC),
        "sha-256",
        hashlib.sha256(payload).hexdigest(),
    )
    adapter = FakeModelAdapter(payload, RemoteModelCatalog((entry,), datetime.now(UTC)))
    capabilities = ComfyUICapabilities(
        checkpoints=(entry.relative_path,) if visible else (),
        vaes=(),
        samplers=(),
        schedulers=(),
        loras=(),
        upscale_models=(),
        available_node_classes=frozenset(),
        warnings=(),
    )

    async def refresh() -> CapabilityRefreshResult:
        return CapabilityRefreshResult(True, "ok", capabilities)

    service = ModelPreparationService(repository, adapter, settings, refresh)
    return settings, repository, adapter, service, entry, payload


class FakeModelAdapter:
    def __init__(self, payload: bytes, catalog: RemoteModelCatalog) -> None:
        self.payload = payload
        self.catalog = catalog
        self.download_calls = 0
        self.write_wrong_payload = False

    async def list_catalog(self) -> RemoteModelCatalog:
        return self.catalog

    async def download(self, entry, destination, **kwargs) -> None:
        self.download_calls += 1
        started = kwargs.get("process_started_callback")
        if started is not None:
            result = started(1234)
            if asyncio.iscoroutine(result):
                await result
        progress = kwargs.get("progress_callback")
        if progress is not None:
            result = progress(ModelTransferProgress(entry.size_bytes, entry.size_bytes, 100.0))
            if asyncio.iscoroutine(result):
                await result
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"wrong" if self.write_wrong_payload else self.payload)
        finished = kwargs.get("process_finished_callback")
        if finished is not None:
            result = finished()
            if asyncio.iscoroutine(result):
                await result


def test_remote_catalog_filters_unsafe_and_non_model_entries(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    adapter = RemoteModelCatalogAdapter(settings)
    payload = [
        {"Name": "characters/example.safetensors", "Size": 10, "IsDir": False},
        {"Name": "styles/example.safetensors", "Size": 11, "IsDir": False},
        {"Name": "readme.txt", "Size": 2, "IsDir": False},
        {"Name": "preview.png", "Size": 2, "IsDir": False},
        {"Name": "../escape.safetensors", "Size": 2, "IsDir": False},
        {"Name": "folder", "Size": -1, "IsDir": True},
    ]

    entries = adapter._parse_entries(RemoteModelKind.LORA, payload)

    assert [item.relative_path for item in entries] == [
        "characters/example.safetensors",
        "styles/example.safetensors",
    ]


def test_remote_model_settings_are_typed_and_reject_traversal(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    assert settings.remote_model_base_path == "SDXLModels"
    assert settings.remote_checkpoint_subdir == "checkpoints"
    assert settings.remote_model_download_timeout_seconds == 3600
    with pytest.raises(ValueError):
        Settings(_env_file=None, remote_model_base_path="../outside")


def test_duplicate_remote_enqueue_returns_one_active_job(tmp_path: Path) -> None:
    _settings_value, repository, _adapter, _service, entry, _payload = _fixture(tmp_path)

    first = repository.enqueue(entry, entry.relative_path)
    second = repository.enqueue(entry, entry.relative_path)

    assert second.id == first.id
    assert len(repository.list_jobs()) == 1


def test_transfer_verifies_and_atomically_places_nested_model(tmp_path: Path) -> None:
    settings, repository, adapter, service, entry, payload = _fixture(tmp_path)
    job = repository.enqueue(entry, entry.relative_path)
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None

    result = asyncio.run(service.process_job(claimed, "worker-1"))
    final_path = settings.checkpoint_dir / entry.relative_path

    assert result.status is ModelTransferStatus.COMPLETED
    assert final_path.read_bytes() == payload
    assert not list(final_path.parent.glob(".*.download"))
    assert adapter.download_calls == 1
    assert repository.get(job.id).status is ModelTransferStatus.COMPLETED


def test_mismatch_preserves_existing_final_model(tmp_path: Path) -> None:
    settings, repository, adapter, service, entry, _payload = _fixture(tmp_path)
    final_path = settings.checkpoint_dir / entry.relative_path
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"old-valid-file")
    adapter.write_wrong_payload = True
    job = repository.enqueue(entry, entry.relative_path)
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None

    with pytest.raises(ModelPreparationServiceError) as caught:
        asyncio.run(service.process_job(claimed, "worker-1"))

    assert caught.value.code == ModelTransferErrorCode.SIZE_MISMATCH.value
    assert final_path.read_bytes() == b"old-valid-file"
    assert not list(final_path.parent.glob(".*.download"))
    assert repository.get(job.id).status is ModelTransferStatus.DOWNLOADING


def test_local_matching_model_skips_download(tmp_path: Path) -> None:
    settings, repository, adapter, service, entry, payload = _fixture(tmp_path)
    final_path = settings.checkpoint_dir / entry.relative_path
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(payload)

    job = asyncio.run(service.prepare_entry(entry))

    assert job.status is ModelTransferStatus.COMPLETED
    assert adapter.download_calls == 0
    assert final_path.read_bytes() == payload


def test_visibility_failure_does_not_select_another_model(tmp_path: Path) -> None:
    _settings_value, repository, _adapter, service, entry, _payload = _fixture(
        tmp_path, visible=False
    )
    job = repository.enqueue(entry, entry.relative_path)
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None

    with pytest.raises(ModelPreparationServiceError) as caught:
        asyncio.run(service.process_job(claimed, "worker-1"))

    assert caught.value.code == ModelTransferErrorCode.MODEL_NOT_VISIBLE.value
    assert repository.get(job.id).status is ModelTransferStatus.DOWNLOADING


def test_stateless_restore_terminalizes_model_jobs_once(tmp_path: Path) -> None:
    _settings_value, repository, _adapter, _service, entry, _payload = _fixture(tmp_path)
    repository.enqueue(entry, entry.relative_path)
    claimed = repository.claim_next("old-worker", 120)
    assert claimed is not None

    assert repository.reconcile_stateless_restore() == 1
    assert repository.reconcile_stateless_restore() == 0
    restored = repository.get(claimed.id)
    assert restored is not None
    assert restored.status is ModelTransferStatus.FAILED
    assert restored.error_code == ModelTransferErrorCode.STATELESS_RESTORE_INTERRUPTED.value


def test_application_restart_does_not_start_a_second_download(tmp_path: Path) -> None:
    _settings_value, repository, _adapter, _service, entry, _payload = _fixture(tmp_path)
    repository.enqueue(entry, entry.relative_path)
    claimed = repository.claim_next("old-worker", 120)
    assert claimed is not None

    assert repository.reconcile_interrupted() == 1
    interrupted = repository.get(claimed.id)
    assert interrupted is not None
    assert interrupted.status is ModelTransferStatus.FAILED
    assert interrupted.error_code == ModelTransferErrorCode.APP_RESTART_INTERRUPTED.value


def test_reconciliation_can_repair_valid_file_after_db_completion_gap(tmp_path: Path) -> None:
    settings, repository, _adapter, _service, entry, payload = _fixture(tmp_path)
    final_path = settings.checkpoint_dir / entry.relative_path
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(payload)
    job = repository.enqueue(entry, entry.relative_path)
    claimed = repository.claim_next("worker-1", 120)
    assert claimed is not None

    repaired = repository.repair_completed(job.id, hashlib.sha256(payload).hexdigest())

    assert repaired.status is ModelTransferStatus.COMPLETED
    assert repaired.local_sha256 == hashlib.sha256(payload).hexdigest()
    assert final_path.read_bytes() == payload


def test_model_preparation_is_a_separate_mobile_ready_tab() -> None:
    demo = build_app(Settings(_env_file=None, environment="phase11-ui-test"))
    values = {
        component["props"].get("value") or component["props"].get("label")
        for component in demo.config["components"]
        if component["type"] in {"button", "tabitem"}
    }

    assert "モデル準備" in values
    assert "Remote一覧を更新" in values
    assert "選択モデルをPodへ準備" in values
    assert ".model-preparation-actions" in demo.config["css"]
