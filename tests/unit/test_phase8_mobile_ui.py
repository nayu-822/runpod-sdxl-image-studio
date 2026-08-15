from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import gradio as gr
import pytest
from sqlalchemy import create_engine

from runpod_sdxl_image_studio.adapters.database.engine import create_session_factory
from runpod_sdxl_image_studio.adapters.database.models import Base
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepository,
)
from runpod_sdxl_image_studio.config import Settings
from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.generation_history import GenerationHistoryItem
from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.services.generation_history_service import GenerationHistoryError
from runpod_sdxl_image_studio.services.generation_queue_service import (
    GenerationQueueService,
    GenerationQueueServiceError,
)
from runpod_sdxl_image_studio.services.interactive_generation_service import (
    InteractiveGenerationError,
)
from runpod_sdxl_image_studio.ui.components.lora_editor import build_lora_editor
from runpod_sdxl_image_studio.ui.components.mobile_actions import (
    make_mobile_status_poll_handler,
    make_mobile_status_refresh_handler,
)
from runpod_sdxl_image_studio.ui.mobile_styles import MOBILE_UI_CSS
from runpod_sdxl_image_studio.ui.tabs.history_tab import render_history_thumbnails
from runpod_sdxl_image_studio.ui.tabs.preset_tab import make_recent_lora_add_handler
from runpod_sdxl_image_studio.ui.tabs.system_tab import (
    build_generation_tab,
    make_batch_enqueue_handler,
    make_enqueue_handler,
    make_interactive_start_handler,
)


def _queue_item(
    generation_id: UUID,
    *,
    sequence: int,
    status: GenerationStatus,
    progress_value: int | None = None,
    progress_maximum: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        entry=SimpleNamespace(sequence=sequence),
        generation=SimpleNamespace(
            id=generation_id,
            status=status,
            error_summary=None,
            settings_snapshot=SimpleNamespace(seed=123),
        ),
        job=SimpleNamespace(
            progress_value=progress_value,
            progress_maximum=progress_maximum,
            current_node="KSampler",
        ),
    )


class _FakeQueueService:
    def __init__(self, items: tuple[object, ...] = (), error: Exception | None = None) -> None:
        self.items = items
        self.error = error
        self.list_limit: int | None = None
        self.latest_candidate_calls = 0
        self.detail_calls = 0

    def list_jobs(self, *, limit: int = 200) -> tuple[object, ...]:
        self.list_limit = limit
        if self.error is not None:
            raise self.error
        return self.items

    def get_latest_status_candidate(self) -> object | None:
        self.latest_candidate_calls += 1
        if self.error is not None:
            raise self.error
        return max(
            self.items,
            key=lambda item: item.entry.sequence,
            default=None,
        )

    def get_job_detail(self, generation_id: UUID) -> object | None:
        self.detail_calls += 1
        return next(
            (
                item
                for item in self.items
                if getattr(getattr(item, "generation", None), "id", None) == generation_id
            ),
            None,
        )


class _FakeHistoryService:
    def __init__(self, detail: object | None = None) -> None:
        self.detail = detail
        self.detail_calls = 0
        self.image_path = Path("data/generations/2026-08-08/image.png")

    def get_detail(self, _generation_id: UUID) -> object:
        self.detail_calls += 1
        if self.detail is None:
            raise GenerationHistoryError("detail unavailable")
        return self.detail

    def absolute_data_path(self, _relative_path: str | None) -> Path:
        return self.image_path


class _FakeEnqueueService:
    def __init__(self, item: object | None = None, enqueue_item: object | None = None) -> None:
        self.item = item
        self.enqueue_item = enqueue_item if enqueue_item is not None else item
        self.detail_calls = 0
        self.enqueue_calls = 0
        self.parent_generation_ids: list[UUID | None] = []

    def get_job_detail(self, generation_id: UUID) -> object | None:
        self.detail_calls += 1
        if self.item is None:
            return None
        return self.item if self.item.generation.id == generation_id else None

    def enqueue(
        self,
        _settings: object,
        *,
        parent_generation_id: UUID | None = None,
        admission_check: Callable[[], None] | None = None,
    ) -> object:
        if admission_check is not None:
            admission_check()
        self.enqueue_calls += 1
        self.parent_generation_ids.append(parent_generation_id)
        if self.enqueue_item is None:
            raise AssertionError("enqueue should not be called without a result item")
        return SimpleNamespace(
            item=self.enqueue_item,
            queue_position=self.enqueue_item.entry.sequence,
        )


class _FakeBatchService:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_id = uuid4()

    def enqueue_batch(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.calls += 1
        return SimpleNamespace(
            batch=SimpleNamespace(id=self.batch_id),
            items=(object(), object()),
        )


def _enqueue_inputs(
    parent_generation_id: UUID | None,
    *,
    regeneration_valid: bool = True,
    regeneration_requested: bool = True,
) -> tuple[object, ...]:
    return (
        "checkpoint.safetensors",
        "positive prompt",
        "negative prompt",
        "1024 × 1024",
        1024,
        1024,
        "Fixed",
        123,
        28,
        5.5,
        "euler",
        "normal",
        None,
        [],
        str(parent_generation_id) if parent_generation_id is not None else None,
        regeneration_valid,
        regeneration_requested,
    )


def test_mobile_css_covers_responsive_layout_without_overflow_masking() -> None:
    assert ".generation-layout" in MOBILE_UI_CSS
    assert ".tab-nav" in MOBILE_UI_CSS
    assert "grid-template-columns: minmax(0, 1fr)" in MOBILE_UI_CSS
    assert "@media (max-width: 639px)" in MOBILE_UI_CSS
    assert "@media (min-width: 1024px)" in MOBILE_UI_CSS
    assert "env(safe-area-inset-bottom" in MOBILE_UI_CSS
    assert "min-height: 44px" in MOBILE_UI_CSS
    assert "overflow-x: hidden" not in MOBILE_UI_CSS


def test_generation_tab_exposes_mobile_status_and_prompt_controls() -> None:
    with gr.Blocks():
        generation = build_generation_tab(max_loras=2)

    assert generation.positive_prompt.lines == 6
    assert generation.negative_prompt.lines == 4
    assert generation.status_poll_timer.value == 5
    assert generation.generate_button.elem_classes == ["mobile-tap-button"]
    assert generation.final_upscale_message.value == ""
    assert ".interactive-result-gallery" in MOBILE_UI_CSS
    assert generation.lora_editor.rows[0].container.elem_classes == ["lora-card-row"]
    assert generation.lora_editor.rows[0].name.label == "LoRA名"


def test_interactive_final_upscale_without_model_does_not_create_a_run() -> None:
    class _Interactive:
        def __init__(self) -> None:
            self.start_calls = 0

        def start(self, *_args: object, **_kwargs: object) -> object:
            self.start_calls += 1
            raise AssertionError("invalid final-upscale settings must not start a run")

    service = _Interactive()
    handler = make_interactive_start_handler(service, max_loras=2)  # type: ignore[arg-type]
    result = asyncio.run(
        handler(
            "checkpoint.safetensors",
            "positive",
            "negative",
            "1024 × 1024",
            1024,
            1024,
            "Fixed",
            1,
            28,
            5.5,
            "euler",
            "normal",
            None,
            [],
            1,
            1,
            None,
            1,
            False,
            1.5,
            "lanczos",
            20,
            5.5,
            "euler",
            "normal",
            0.4,
            True,
            "2026-08-14",
        )
    )

    assert service.start_calls == 0
    assert result[0] is None
    assert "アップスケーラーを選択" in result[3]


def test_lora_editor_uses_fixed_state_rows_and_mobile_card_actions() -> None:
    with gr.Blocks():
        editor = build_lora_editor(max_loras=3)

    assert len(editor.rows) == 3
    assert editor.add_button.elem_classes == ["mobile-tap-button"]
    assert [row.container.visible for row in editor.rows] == [True, False, False]
    assert [row.up_button.value for row in editor.rows] == ["↑ 上へ"] * 3
    assert [row.down_button.value for row in editor.rows] == ["↓ 下へ"] * 3


def test_mobile_status_handler_uses_bounded_queue_lookup_and_completed_detail() -> None:
    generation_id = uuid4()
    item = _queue_item(generation_id, sequence=4, status=GenerationStatus.COMPLETED)
    queue = _FakeQueueService((item,))
    detail = SimpleNamespace(
        generation_id=str(generation_id),
        image_path="generations/2026-08-08/image.png",
        status_text="completed",
        snapshot=SimpleNamespace(seed=123),
        favorite=True,
    )
    history = _FakeHistoryService(detail)
    handler = make_mobile_status_refresh_handler(queue, history)

    result = handler(None, "old card", "old.png", "old details", "1", False)

    assert len(result) == 7
    assert queue.list_limit is None
    assert result[0] == str(generation_id)
    assert "completed" in result[1]
    assert result[2] == str(history.image_path)
    assert "123" in result[3]
    assert result[4] == "123"
    assert result[5] is True
    assert history.detail_calls == 1


def test_mobile_status_handler_preserves_last_state_on_queue_failure() -> None:
    queue = _FakeQueueService(error=GenerationQueueServiceError("database unavailable"))
    history = _FakeHistoryService()
    handler = make_mobile_status_refresh_handler(queue, history)

    result = handler(None, "### old status", "old.png", "old details", "123", True)

    assert len(result) == 7
    assert isinstance(result[0], dict)
    assert "old status" in result[1]
    assert "最新状態を取得できませんでした" in result[1]
    assert all(isinstance(result[index], dict) for index in (2, 3, 4, 5))
    assert result[6] == "最新状態を取得できませんでした。"
    assert history.detail_calls == 0


def test_mobile_status_handler_reports_running_progress_without_exposing_prompt() -> None:
    generation_id = uuid4()
    item = _queue_item(
        generation_id,
        sequence=8,
        status=GenerationStatus.RUNNING,
        progress_value=2,
        progress_maximum=4,
    )
    queue = _FakeQueueService((item,))
    history = _FakeHistoryService()
    handler = make_mobile_status_refresh_handler(queue, history)

    result = handler(None, None, None, None, None, False)

    assert len(result) == 7
    assert result[0] == str(generation_id)
    assert "50%" in result[1]
    assert "KSampler" in result[1]
    assert "prompt" not in result[1].lower()
    assert history.detail_calls == 0


def test_mobile_status_poll_does_not_overwrite_active_generation_id() -> None:
    generation_id = uuid4()
    item = _queue_item(generation_id, sequence=8, status=GenerationStatus.RUNNING)
    queue = _FakeQueueService((item,))
    history = _FakeHistoryService()
    handler = make_mobile_status_poll_handler(queue, history)

    result = handler(str(generation_id), "old card", None, "", "", False)

    assert len(result) == 6
    assert "running" in result[0]
    assert history.detail_calls == 0


def test_mobile_status_poll_without_active_generation_stays_idle_without_latest_lookup() -> None:
    generation_id = uuid4()
    queue = _FakeQueueService(
        (_queue_item(generation_id, sequence=8, status=GenerationStatus.RUNNING),)
    )
    history = _FakeHistoryService()
    handler = make_mobile_status_poll_handler(queue, history)

    result = handler(None, None, None, None, None, False)

    assert len(result) == 6
    assert queue.latest_candidate_calls == 0
    assert queue.detail_calls == 0
    assert str(generation_id) not in result[0]
    assert "idle" in result[0]
    assert "生成待機中" in result[0]
    assert result[1] is None
    assert result[2] == ""
    assert result[3] == ""
    assert result[4] is False
    assert result[5] == ""
    assert history.detail_calls == 0

    preserved = handler(None, "old card", "old.png", "old details", "123", True)

    assert queue.latest_candidate_calls == 0
    assert preserved[0] == "old card"
    assert all(isinstance(preserved[index], dict) for index in (1, 2, 3, 4))
    assert preserved[5] == ""


def test_mobile_status_reload_uses_latest_candidate_only_for_full_refresh() -> None:
    generation_id = uuid4()
    queue = _FakeQueueService(
        (_queue_item(generation_id, sequence=8, status=GenerationStatus.RUNNING),)
    )
    handler = make_mobile_status_refresh_handler(queue, _FakeHistoryService())

    result = handler(None, None, None, None, None, False)

    assert queue.latest_candidate_calls == 1
    assert queue.detail_calls == 0
    assert result[0] == str(generation_id)
    assert "running" in result[1]


def test_mobile_status_poll_keeps_active_generation_when_newer_generation_exists() -> None:
    active_id = uuid4()
    newer_id = uuid4()
    queue = _FakeQueueService(
        (
            _queue_item(active_id, sequence=8, status=GenerationStatus.RUNNING),
            _queue_item(newer_id, sequence=9, status=GenerationStatus.RUNNING),
        )
    )
    handler = make_mobile_status_poll_handler(queue, _FakeHistoryService())

    result = handler(str(active_id), None, None, None, None, False)

    assert queue.latest_candidate_calls == 0
    assert queue.detail_calls == 1
    assert str(active_id)[:12] in result[0]
    assert str(newer_id)[:12] not in result[0]


def test_batch_enqueue_and_timer_poll_keep_active_generation_state() -> None:
    batch_service = _FakeBatchService()
    batch_handler = make_batch_enqueue_handler(batch_service, max_loras=2)
    batch_result = asyncio.run(
        batch_handler(
            "checkpoint.safetensors",
            "positive",
            "negative",
            1024,
            1024,
            "Fixed",
            123,
            28,
            5.5,
            "euler",
            "normal",
            None,
            [],
            2,
            "sequential",
            123,
            1,
            "Batch",
        )
    )

    assert batch_service.calls == 1
    assert len(batch_result) == 3
    assert f"Batch `{batch_service.batch_id}`" in batch_result[2]

    active_id = uuid4()
    newer_id = uuid4()
    queue = _FakeQueueService(
        (
            _queue_item(active_id, sequence=8, status=GenerationStatus.RUNNING),
            _queue_item(newer_id, sequence=9, status=GenerationStatus.RUNNING),
        )
    )
    poll_handler = make_mobile_status_poll_handler(queue, _FakeHistoryService())

    fresh_result = poll_handler(None, None, None, None, None, False)
    active_result = poll_handler(str(active_id), None, None, None, None, False)

    assert queue.latest_candidate_calls == 0
    assert fresh_result[0] != str(newer_id)
    assert str(active_id)[:12] in active_result[0]
    assert str(newer_id)[:12] not in active_result[0]


def test_latest_status_candidate_uses_sequence_26_for_reload_after_more_than_20_jobs() -> None:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    repository = GenerationDispatchQueueRepository(create_session_factory(engine))
    service = GenerationQueueService(
        repository,
        Settings(_env_file=None, queue_max_pending_jobs=40),
    )
    try:
        items = tuple(
            service.enqueue(
                GenerationSettings(
                    positive_prompt="positive",
                    negative_prompt="negative",
                    checkpoint_name="checkpoint.safetensors",
                    sampler_name="euler",
                    scheduler_name="normal",
                    vae_name=None,
                    loras=(),
                    width=1024,
                    height=1024,
                    seed=index,
                    steps=28,
                    cfg_scale=5.5,
                )
            ).item
            for index in range(26)
        )
        latest = service.get_latest_status_candidate()

        assert items[-1].entry.sequence == 26
        assert latest is not None
        assert latest.generation.id == items[-1].generation.id

        handler = make_mobile_status_refresh_handler(service, _FakeHistoryService())
        reloaded = handler(None, None, None, None, None, False)
        assert reloaded[0] == str(items[-1].generation.id)
    finally:
        engine.dispose()


def test_latest_status_candidate_falls_back_to_newest_terminal_queue_item() -> None:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    repository = GenerationDispatchQueueRepository(create_session_factory(engine))
    service = GenerationQueueService(
        repository,
        Settings(_env_file=None, queue_max_pending_jobs=4),
    )
    try:
        first = service.enqueue(
            GenerationSettings(
                positive_prompt="positive",
                negative_prompt="negative",
                checkpoint_name="checkpoint.safetensors",
                sampler_name="euler",
                scheduler_name="normal",
                width=1024,
                height=1024,
                seed=1,
                steps=28,
                cfg_scale=5.5,
            )
        ).item
        second = service.enqueue(
            GenerationSettings(
                positive_prompt="positive",
                negative_prompt="negative",
                checkpoint_name="checkpoint.safetensors",
                sampler_name="euler",
                scheduler_name="normal",
                width=1024,
                height=1024,
                seed=2,
                steps=28,
                cfg_scale=5.5,
            )
        ).item
        repository.mark_cancelled(first.generation.id)
        repository.mark_cancelled(second.generation.id)

        latest = service.get_latest_status_candidate()

        assert latest is not None
        assert latest.generation.id == second.generation.id
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "status",
    [
        GenerationStatus.PENDING,
        GenerationStatus.QUEUED,
        GenerationStatus.RUNNING,
        GenerationStatus.FAILED,
        GenerationStatus.CANCELLED,
    ],
)
def test_result_regeneration_enqueues_only_completed_generation(status: GenerationStatus) -> None:
    parent_id = uuid4()
    item = _queue_item(parent_id, sequence=1, status=status)
    service = _FakeEnqueueService(item)
    handler = make_enqueue_handler(service, max_loras=2)

    result = asyncio.run(
        handler(
            *_enqueue_inputs(
                parent_id,
                regeneration_valid=True,
                regeneration_requested=True,
            )
        )
    )

    assert service.detail_calls == 1
    assert service.enqueue_calls == 0
    assert result[0].interactive is True
    assert result[4] is False
    assert isinstance(result[5], dict)
    assert "完了済みGenerationだけ" in result[3]


@pytest.mark.parametrize(
    ("parent_id", "regeneration_valid", "message"),
    [
        (None, False, "復元に失敗"),
        (uuid4(), False, "利用できない設定"),
    ],
)
def test_result_regeneration_restore_failure_does_not_enqueue(
    parent_id: UUID | None,
    regeneration_valid: bool,
    message: str,
) -> None:
    service = _FakeEnqueueService()
    handler = make_enqueue_handler(service, max_loras=2)

    result = asyncio.run(
        handler(
            *_enqueue_inputs(
                parent_id,
                regeneration_valid=regeneration_valid,
                regeneration_requested=True,
            )
        )
    )

    assert service.detail_calls == 0
    assert service.enqueue_calls == 0
    assert result[0].interactive is True
    assert result[4] is False
    assert isinstance(result[5], dict)
    assert message in result[3]


def test_result_regeneration_returns_new_active_generation_id_after_completed_parent() -> None:
    parent_id = uuid4()
    new_id = uuid4()
    parent = _queue_item(parent_id, sequence=1, status=GenerationStatus.COMPLETED)
    queued = _queue_item(new_id, sequence=2, status=GenerationStatus.PENDING)
    service = _FakeEnqueueService(parent, enqueue_item=queued)
    handler = make_enqueue_handler(service, max_loras=2)

    result = asyncio.run(handler(*_enqueue_inputs(parent_id)))

    assert service.detail_calls == 1
    assert service.enqueue_calls == 1
    assert service.parent_generation_ids == [parent_id]
    assert result[5] == str(new_id)


@pytest.mark.parametrize(
    "active_status",
    ["active", "cancelling", "completed", "failed", "cancelled", None],
)
def test_legacy_regeneration_is_admitted_only_without_active_interactive_run(
    active_status: str | None,
) -> None:
    parent_id = uuid4()
    new_id = uuid4()
    parent = _queue_item(parent_id, sequence=1, status=GenerationStatus.COMPLETED)
    queued = _queue_item(new_id, sequence=2, status=GenerationStatus.PENDING)
    service = _FakeEnqueueService(parent, enqueue_item=queued)

    class _InteractiveAdmission:
        def ensure_no_active_run(self) -> None:
            if active_status in {"active", "cancelling"}:
                raise InteractiveGenerationError(
                    f"interactive run is {active_status}; regeneration is blocked"
                )

    handler = make_enqueue_handler(
        service,
        max_loras=2,
        interactive_service=_InteractiveAdmission(),  # type: ignore[arg-type]
    )

    result = asyncio.run(
        handler(
            *_enqueue_inputs(
                parent_id,
                regeneration_valid=True,
                regeneration_requested=True,
            )
        )
    )

    if active_status not in {"active", "cancelling"}:
        assert service.enqueue_calls == 1
        assert result[5] == str(new_id)
    else:
        assert service.enqueue_calls == 0
        assert active_status in result[3]
        assert result[5] is None or isinstance(result[5], dict)


class _SharedAdmissionGate:
    def __init__(self, *, hold_first_admission: bool = False) -> None:
        self.lock = threading.RLock()
        self.hold_first_admission = hold_first_admission
        self.first_admission_entered = threading.Event()
        self.release_first_admission = threading.Event()
        self._first_admission_seen = False

    def ensure_work_allowed(self) -> None:
        return

    @contextmanager
    def admit_work(self) -> Iterator[None]:
        with self.lock:
            if self.hold_first_admission and not self._first_admission_seen:
                self._first_admission_seen = True
                self.first_admission_entered.set()
                if not self.release_first_admission.wait(timeout=5):
                    raise AssertionError("the first admission did not get released")
            yield


class _AdmissionDispatchRepository:
    def __init__(self) -> None:
        self.enqueue_calls = 0
        self.item = _queue_item(uuid4(), sequence=1, status=GenerationStatus.PENDING)

    def enqueue_single(self, _snapshot: object, **_kwargs: object) -> object:
        self.enqueue_calls += 1
        return self.item


class _RaceInteractiveAdmission:
    def __init__(self, gate: _SharedAdmissionGate) -> None:
        self._gate = gate
        self.active = False
        self.start_called = threading.Event()
        self.active_confirmed = threading.Event()

    def ensure_no_active_run(self) -> None:
        if self.active:
            raise InteractiveGenerationError(
                "generation is unavailable while an interactive run is active or cancelling"
            )

    def start(self) -> None:
        self.start_called.set()
        with self._gate.admit_work():
            self.active = True
            self.active_confirmed.set()


class _BlockingPreflight:
    def __init__(self, *, block: bool) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = block

    async def check(self, *_args: object, **_kwargs: object) -> object:
        self.entered.set()
        if self.block and not self.release.wait(timeout=5):
            raise AssertionError("preflight did not get released")
        return SimpleNamespace(is_ready=True, warnings=())


def test_legacy_enqueue_rechecks_inside_shared_admission_after_interactive_start() -> None:
    gate = _SharedAdmissionGate()
    dispatch = _AdmissionDispatchRepository()
    queue = GenerationQueueService(  # type: ignore[arg-type]
        dispatch,
        Settings(_env_file=None),
        lifecycle_gate=gate,
    )
    interactive = _RaceInteractiveAdmission(gate)
    preflight = _BlockingPreflight(block=True)
    handler = make_enqueue_handler(
        queue,
        max_loras=2,
        preflight_service=preflight,  # type: ignore[arg-type]
        interactive_service=interactive,  # type: ignore[arg-type]
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        legacy_future = executor.submit(
            lambda: asyncio.run(
                handler(
                    *_enqueue_inputs(
                        None,
                        regeneration_valid=False,
                        regeneration_requested=False,
                    )
                )
            )
        )
        assert preflight.entered.wait(timeout=5)
        interactive_future = executor.submit(interactive.start)
        assert interactive.active_confirmed.wait(timeout=5)
        preflight.release.set()
        result = legacy_future.result(timeout=5)
        interactive_future.result(timeout=5)

    assert dispatch.enqueue_calls == 0
    assert result[5] is None or isinstance(result[5], dict)
    assert "active" in result[3]


def test_legacy_enqueue_admission_wins_before_interactive_start() -> None:
    gate = _SharedAdmissionGate(hold_first_admission=True)
    dispatch = _AdmissionDispatchRepository()
    queue = GenerationQueueService(  # type: ignore[arg-type]
        dispatch,
        Settings(_env_file=None),
        lifecycle_gate=gate,
    )
    interactive = _RaceInteractiveAdmission(gate)
    preflight = _BlockingPreflight(block=False)
    handler = make_enqueue_handler(
        queue,
        max_loras=2,
        preflight_service=preflight,  # type: ignore[arg-type]
        interactive_service=interactive,  # type: ignore[arg-type]
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        legacy_future = executor.submit(
            lambda: asyncio.run(
                handler(
                    *_enqueue_inputs(
                        None,
                        regeneration_valid=False,
                        regeneration_requested=False,
                    )
                )
            )
        )
        assert gate.first_admission_entered.wait(timeout=5)
        interactive_future = executor.submit(interactive.start)
        assert interactive.start_called.wait(timeout=5)
        assert not interactive.active_confirmed.wait(timeout=0.1)
        gate.release_first_admission.set()
        result = legacy_future.result(timeout=5)
        interactive_future.result(timeout=5)

    assert dispatch.enqueue_calls == 1
    assert result[1] == "Queued"
    assert interactive.active is True


def test_recent_lora_shortcut_fills_the_first_empty_middle_row() -> None:
    handler = make_recent_lora_add_handler(4)
    state = [
        {
            "row_id": "one",
            "lora_name": "one.safetensors",
            "model_strength": 0.8,
            "clip_strength": 0.7,
        },
        {
            "row_id": "empty",
            "lora_name": None,
            "model_strength": 1.0,
            "clip_strength": 1.0,
        },
        {
            "row_id": "three",
            "lora_name": "three.safetensors",
            "model_strength": 0.6,
            "clip_strength": 0.5,
        },
    ]

    result = handler(
        "two.safetensors",
        state,
        ("one.safetensors", "two.safetensors", "three.safetensors"),
    )

    updated = result[0]
    assert [row["lora_name"] for row in updated] == [
        "one.safetensors",
        "two.safetensors",
        "three.safetensors",
    ]
    assert updated[0]["model_strength"] == 0.8
    assert updated[2]["clip_strength"] == 0.5


def test_history_gallery_never_falls_back_to_primary_image() -> None:
    class _ThumbnailService:
        def absolute_data_path(self, relative_path: str | None) -> Path | None:
            if relative_path == "thumb.webp":
                return Path("data/thumb.webp")
            return None

    item_with_thumbnail = GenerationHistoryItem(
        generation_id="one",
        created_at_text="2026-08-08",
        status_text="completed",
        checkpoint_label="checkpoint",
        lora_labels=(),
        seed_text="1",
        resolution_text="1024 × 1024",
        thumbnail_path="thumb.webp",
        favorite=False,
        kind_text="normal",
        error_summary=None,
    )
    item_without_thumbnail = GenerationHistoryItem(
        generation_id="two",
        created_at_text="2026-08-08",
        status_text="completed",
        checkpoint_label="checkpoint",
        lora_labels=(),
        seed_text="2",
        resolution_text="1024 × 1024",
        thumbnail_path=None,
        favorite=False,
        kind_text="normal",
        error_summary=None,
    )

    values = render_history_thumbnails(
        _ThumbnailService(), (item_with_thumbnail, item_without_thumbnail)
    )

    assert values[0][0].endswith("thumb.webp")
    assert values[1][0].endswith("thumbnail_placeholder.svg")
    assert "サムネイル未生成" in values[1][1]
