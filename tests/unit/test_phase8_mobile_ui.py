from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import gradio as gr

from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.generation_history import GenerationHistoryItem
from runpod_sdxl_image_studio.services.generation_history_service import GenerationHistoryError
from runpod_sdxl_image_studio.services.generation_queue_service import GenerationQueueServiceError
from runpod_sdxl_image_studio.ui.components.lora_editor import build_lora_editor
from runpod_sdxl_image_studio.ui.components.mobile_actions import (
    make_mobile_status_refresh_handler,
)
from runpod_sdxl_image_studio.ui.mobile_styles import MOBILE_UI_CSS
from runpod_sdxl_image_studio.ui.tabs.history_tab import render_history_thumbnails
from runpod_sdxl_image_studio.ui.tabs.system_tab import build_generation_tab


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

    def list_jobs(self, *, limit: int = 200) -> tuple[object, ...]:
        self.list_limit = limit
        if self.error is not None:
            raise self.error
        return self.items

    def get_job_detail(self, generation_id: UUID) -> object | None:
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
    assert generation.lora_editor.rows[0].container.elem_classes == ["lora-card-row"]
    assert generation.lora_editor.rows[0].name.label == "LoRA名"


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
    assert queue.list_limit == 20
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
