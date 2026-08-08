"""既存Serviceを使ったモバイル向け状態・結果表示ハンドラー。"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

import gradio as gr

from runpod_sdxl_image_studio.domain.generation import GenerationStatus
from runpod_sdxl_image_studio.domain.generation_history import GenerationDetailView
from runpod_sdxl_image_studio.domain.generation_queue import GenerationQueueItem
from runpod_sdxl_image_studio.services.generation_history_service import (
    GenerationHistoryError,
    GenerationHistoryService,
)
from runpod_sdxl_image_studio.services.generation_queue_service import (
    GenerationQueueService,
    GenerationQueueServiceError,
)
from runpod_sdxl_image_studio.ui.view_models import (
    GenerationStatusCardView,
    generation_status_card_markdown,
)

MobileStatusOutputs = tuple[object, object, object, object, object, object, object]


def make_mobile_status_refresh_handler(
    queue_service: GenerationQueueService,
    history_service: GenerationHistoryService,
) -> Callable[..., MobileStatusOutputs]:
    """Create a read-only status poll using the selected Generation when available."""

    def handler(
        active_generation_id: str | None = None,
        current_card: str | None = None,
        current_image: object = None,
        current_details: str | None = None,
        current_seed: str | None = None,
        current_favorite: bool = False,
    ) -> MobileStatusOutputs:
        try:
            item = _resolve_item(queue_service, active_generation_id)
            if item is None:
                if active_generation_id:
                    detail = history_service.get_detail(UUID(active_generation_id))
                    return _completed_detail_outputs(history_service, detail)
                return (
                    None,
                    generation_status_card_markdown(
                        GenerationStatusCardView(None, "idle", None, None, None, "生成待機中")
                    ),
                    None,
                    "",
                    "",
                    False,
                    "",
                )

            card = generation_status_card_markdown(_status_view(item))
            if item.generation.status is not GenerationStatus.COMPLETED:
                return (
                    str(item.generation.id),
                    card,
                    gr.skip() if current_image is not None else None,
                    gr.skip() if current_details else "",
                    gr.skip() if current_seed else "",
                    gr.skip() if current_favorite else False,
                    "",
                )
            detail = history_service.get_detail(item.generation.id)
            return _completed_detail_outputs(history_service, detail)
        except (GenerationQueueServiceError, GenerationHistoryError, ValueError):
            return (
                gr.skip(),
                _preserve_status_card(current_card),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "最新状態を取得できませんでした。",
            )

    return handler


def _resolve_item(
    service: GenerationQueueService,
    active_generation_id: str | None,
) -> GenerationQueueItem | None:
    if active_generation_id and active_generation_id.strip():
        return service.get_job_detail(UUID(active_generation_id.strip()))
    items = service.list_jobs(limit=20)
    return max(items, key=lambda item: item.entry.sequence, default=None)


def _status_view(item: GenerationQueueItem) -> GenerationStatusCardView:
    job = item.job
    percentage = None
    if job.progress_value is not None and job.progress_maximum:
        percentage = min(100.0, max(0.0, job.progress_value / job.progress_maximum * 100))
    status = item.generation.status
    message = {
        GenerationStatus.PENDING: "生成準備中です。",
        GenerationStatus.QUEUED: "キューで待機中です。",
        GenerationStatus.RUNNING: "生成中です。",
        GenerationStatus.COMPLETED: "生成が完了しました。",
        GenerationStatus.FAILED: item.generation.error_summary or "生成に失敗しました。",
        GenerationStatus.CANCELLED: "生成をキャンセルしました。",
    }[status]
    return GenerationStatusCardView(
        generation_id=str(item.generation.id),
        status=status.value,
        queue_position=item.entry.sequence,
        progress_percentage=percentage,
        current_step=job.current_node,
        message=message,
    )


def _completed_detail_outputs(
    service: GenerationHistoryService,
    detail: GenerationDetailView,
) -> MobileStatusOutputs:
    generation_id = str(detail.generation_id)
    image_path = service.absolute_data_path(detail.image_path)
    status = GenerationStatusCardView(
        generation_id=generation_id,
        status=detail.status_text,
        queue_position=None,
        progress_percentage=100.0,
        current_step=None,
        message="生成が完了しました。",
    )
    result_details = (
        f"Generation ID: `{generation_id}`\n"
        f"実使用seed: `{detail.snapshot.seed}`\n"
        f"状態: `{detail.status_text}`"
    )
    return (
        generation_id,
        generation_status_card_markdown(status),
        str(image_path) if image_path is not None else None,
        result_details,
        str(detail.snapshot.seed),
        detail.favorite,
        "",
    )


def _preserve_status_card(current_card: str | None) -> str:
    if current_card:
        return current_card + "\n\n⚠ 最新状態を取得できませんでした。"
    return generation_status_card_markdown(
        GenerationStatusCardView(
            None, "unknown", None, None, None, "最新状態を取得できませんでした。"
        )
    )


__all__ = ["MobileStatusOutputs", "make_mobile_status_refresh_handler"]
