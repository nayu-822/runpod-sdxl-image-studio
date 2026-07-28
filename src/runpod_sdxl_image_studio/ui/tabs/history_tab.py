"""Mobile-friendly generation history tab and UI boundary handlers."""

from __future__ import annotations

import html
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from uuid import UUID

import gradio as gr

from runpod_sdxl_image_studio.domain.generation import GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_history import (
    GenerationDetailView,
    GenerationHistoryFilter,
    GenerationHistoryItem,
)
from runpod_sdxl_image_studio.services.generation_history_service import (
    GenerationHistoryError,
    GenerationHistoryService,
)
from runpod_sdxl_image_studio.services.generation_recovery_service import (
    GenerationRecoveryService,
)
from runpod_sdxl_image_studio.ui.components.lora_editor import (
    render_state_updates,
)


@dataclass(frozen=True)
class HistoryTabComponents:
    refresh_button: gr.Button
    page_state: gr.State
    date_filter: gr.Textbox
    status_filter: gr.Dropdown
    kind_filter: gr.Dropdown
    favorite_filter: gr.Dropdown
    cards: gr.Markdown
    thumbnail_gallery: gr.Gallery
    page: gr.Markdown
    previous_button: gr.Button
    next_button: gr.Button
    selected: gr.Dropdown
    detail: gr.Markdown
    image: gr.Image
    favorite: gr.Checkbox
    note: gr.Textbox
    save_note_button: gr.Button
    restore_button: gr.Button
    regenerate_button: gr.Button
    message: gr.Markdown


def begin_regeneration() -> tuple[gr.Button, bool]:
    """Disable regeneration immediately and mark the chained request."""

    return gr.Button(value="再生成中...", interactive=False), True


def enable_regeneration_button() -> gr.Button:
    """Restore regeneration after all success and failure paths."""

    return gr.Button(value="同条件で再生成", interactive=True)


def build_history_tab() -> HistoryTabComponents:
    gr.Markdown("## 生成履歴")
    with gr.Row():
        date_filter = gr.Textbox(label="日付 (YYYY-MM-DD)", placeholder="2026-07-26")
        status_filter = gr.Dropdown(
            [("すべて", "")] + [(status.value, status.value) for status in GenerationStatus],
            value="",
            label="status",
        )
        kind_filter = gr.Dropdown(
            [("すべて", "")] + [(kind.value, kind.value) for kind in GenerationKind],
            value="",
            label="kind",
        )
        favorite_filter = gr.Dropdown(
            [("すべて", ""), ("お気に入りのみ", "favorite")],
            value="",
            label="favorite",
        )
    refresh_button = gr.Button("履歴を更新", min_width=140)
    page_state = gr.State(1)
    cards = gr.Markdown("履歴を読み込んでください。")
    thumbnail_gallery = gr.Gallery(
        label="履歴サムネイル",
        columns=2,
        rows=2,
        object_fit="contain",
        height="auto",
    )
    with gr.Row():
        previous_button = gr.Button("前へ", interactive=False)
        page = gr.Markdown("1ページ目")
        next_button = gr.Button("次へ", interactive=False)
    selected = gr.Dropdown([], label="Generation", allow_custom_value=False)
    detail = gr.Markdown("")
    image = gr.Image(label="履歴画像", type="filepath")
    favorite = gr.Checkbox(label="お気に入り")
    note = gr.Textbox(label="メモ", lines=3, max_lines=8, max_length=2000)
    with gr.Row():
        save_note_button = gr.Button("メモを保存")
        restore_button = gr.Button("設定を生成画面へ復元")
        regenerate_button = gr.Button("同条件で再生成")
    message = gr.Markdown("")
    return HistoryTabComponents(
        refresh_button,
        page_state,
        date_filter,
        status_filter,
        kind_filter,
        favorite_filter,
        cards,
        thumbnail_gallery,
        page,
        previous_button,
        next_button,
        selected,
        detail,
        image,
        favorite,
        note,
        save_note_button,
        restore_button,
        regenerate_button,
        message,
    )


def make_history_refresh_handler(
    service: GenerationHistoryService,
    recovery_service: GenerationRecoveryService | None = None,
) -> Callable[..., Awaitable[tuple[object, ...]]]:
    async def handler(
        page: int = 1,
        date_text: str = "",
        status: str | None = None,
        kind: str | None = None,
        favorite: str | None = None,
    ) -> tuple[object, ...]:
        if recovery_service is not None:
            await recovery_service.recover()
        try:
            normalized_page = max(1, int(page))
            selected_date = date.fromisoformat(date_text) if date_text.strip() else None
            result = service.list_history(
                GenerationHistoryFilter(
                    date=selected_date,
                    status=GenerationStatus(status) if status else None,
                    kind=GenerationKind(kind) if kind else None,
                    favorite=True if favorite == "favorite" else None,
                    offset=(normalized_page - 1) * service.page_size,
                )
            )
            items = tuple(service.to_item(generation) for generation in result.generations)
            choices = [(item.generation_id, item.generation_id) for item in items]
            return (
                normalized_page,
                render_history_thumbnails(service, items),
                render_history_items(items),
                gr.Dropdown(choices=choices, value=choices[0][1] if choices else None),
                f"{result.page}ページ目 / 全{result.total_count}件",
                gr.Button(interactive=result.page > 1),
                gr.Button(interactive=result.has_next),
                "履歴を更新しました。",
            )
        except GenerationHistoryError as exc:
            return (
                gr.skip(),
                gr.skip(),
                "履歴を取得できませんでした。",
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                str(exc),
            )
        except ValueError:
            return (
                gr.skip(),
                gr.skip(),
                "履歴フィルターの日付またはstatusを確認してください。",
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "",
            )

    return handler


def previous_history_page(page: int) -> int:
    return max(1, int(page) - 1)


def next_history_page(page: int) -> int:
    return max(1, int(page) + 1)


def make_history_detail_handler(
    service: GenerationHistoryService,
) -> Callable[[str | None], tuple[object, ...]]:
    def handler(selected: str | None) -> tuple[object, ...]:
        if not selected:
            return "", None, gr.skip(), gr.skip(), "履歴を選択してください。"
        try:
            detail = service.get_detail(UUID(selected))
            image_path = service.absolute_data_path(detail.image_path)
            return (
                render_generation_detail(detail),
                str(image_path) if image_path else None,
                detail.favorite,
                detail.user_note or "",
                "",
            )
        except (GenerationHistoryError, ValueError) as exc:
            return "", None, gr.skip(), gr.skip(), str(exc)

    return handler


def make_favorite_handler(
    service: GenerationHistoryService,
) -> Callable[[str | None, bool], tuple[object, ...]]:
    def handler(selected: str | None, favorite: bool) -> tuple[object, ...]:
        if not selected:
            return gr.skip(), "Generationを選択してください。"
        try:
            service.set_favorite(UUID(selected), favorite)
            return favorite, "お気に入りを保存しました。"
        except (GenerationHistoryError, ValueError) as exc:
            return gr.skip(), str(exc)

    return handler


def make_note_handler(service: GenerationHistoryService) -> Callable[[str | None, str], str]:
    def handler(selected: str | None, note: str) -> str:
        if not selected:
            return "Generationを選択してください。"
        try:
            service.update_note(UUID(selected), note)
            return "メモを保存しました。"
        except (GenerationHistoryError, ValueError) as exc:
            return str(exc)

    return handler


def make_restore_handler(
    service: GenerationHistoryService,
    max_loras: int,
) -> Callable[..., tuple[object, ...]]:
    def handler(
        selected: str | None,
        checkpoint_choices: object = None,
        vae_choices: object = None,
        lora_choices: object = None,
    ) -> tuple[object, ...]:
        if not selected:
            return (
                ("履歴を選択してください。",)
                + (gr.skip(),) * (12 + component_output_count(max_loras))
                + (None, False)
            )
        try:
            restored = service.restore_settings(
                UUID(selected),
                checkpoints=_string_choices(checkpoint_choices),
                vaes=_string_choices(vae_choices),
                loras=_lora_values(lora_choices),
                max_loras=max_loras,
            )
            settings = restored.settings
            lora_state = [
                {
                    "row_id": f"restored-{index}",
                    "lora_name": lora.name,
                    "model_strength": lora.model_strength,
                    "clip_strength": lora.clip_strength,
                }
                for index, lora in enumerate(settings.loras)
            ] or [
                {
                    "row_id": "restored-0",
                    "lora_name": None,
                    "model_strength": 1.0,
                    "clip_strength": 1.0,
                }
            ]
            lora_updates = render_state_updates(
                lora_state,
                lora_choices,
                max_loras,
                clear_unavailable=False,
            )
            warning = " / ".join(restored.warnings)
            unverified_warning = (
                "現在のComfyUI一覧を取得していないため、モデルの存在確認は行っていません。"
            )
            blocking = any(item != unverified_warning for item in restored.warnings)
            return (
                "設定を復元しました。" + (f" 警告: {warning}" if warning else ""),
                settings.positive_prompt,
                settings.negative_prompt,
                gr.Dropdown(value=settings.checkpoint_name),
                gr.Dropdown(value=settings.vae_name),
                settings.width,
                settings.height,
                "Previous seed",
                settings.seed,
                settings.steps,
                settings.cfg_scale,
                gr.Dropdown(value=settings.sampler_name),
                gr.Dropdown(value=settings.scheduler_name),
                *lora_updates,
                str(restored.parent_generation_id),
                not blocking,
            )
        except (GenerationHistoryError, ValueError):
            return (
                ("設定を復元できませんでした。",)
                + (gr.skip(),) * (12 + component_output_count(max_loras))
                + (None, False)
            )

    return handler


def component_output_count(max_loras: int) -> int:
    """Return state, row, and add-button outputs for the restore handler."""

    return 2 + 7 * max(1, max_loras)


def _string_choices(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _lora_values(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return ()
    values: list[str] = []
    for item in value:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], str):
            values.append(item[1])
    return tuple(values)


def render_history_items(items: tuple[GenerationHistoryItem, ...]) -> str:
    if not items:
        return "履歴はありません。"
    lines: list[str] = []
    for item in items:
        loras = ", ".join(html.escape(value) for value in item.lora_labels) or "なし"
        error = f" / {html.escape(item.error_summary)}" if item.error_summary else ""
        thumbnail = "サムネイルあり" if item.thumbnail_path else "サムネイルなし"
        lines.append(
            f"- **{html.escape(item.created_at_text)}** `{html.escape(item.status_text)}` "
            f"`{html.escape(item.kind_text)}` / {html.escape(item.checkpoint_label)} / "
            f"LoRA: {loras} / seed: `{html.escape(item.seed_text)}` / "
            f"{html.escape(item.resolution_text)} / {thumbnail}{error}"
        )
    return "\n".join(lines)


def render_history_thumbnails(
    service: GenerationHistoryService,
    items: tuple[GenerationHistoryItem, ...],
) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for item in items:
        if item.thumbnail_path is None:
            continue
        path = service.absolute_data_path(item.thumbnail_path)
        if path is not None:
            values.append(
                (
                    str(path),
                    f"{item.created_at_text} / {item.status_text} / seed {item.seed_text}",
                )
            )
    return values


def render_generation_detail(generation: GenerationDetailView) -> str:
    snapshot = generation.snapshot
    loras = (
        "\n".join(
            f"- {html.escape(lora.name)} "
            f"(model={lora.model_strength:g}, clip={lora.clip_strength:g})"
            for lora in snapshot.loras
        )
        or "- なし"
    )
    values = [
        f"### Generation `{html.escape(generation.generation_id)}`",
        f"status: `{html.escape(generation.status_text)}` / "
        f"kind: `{html.escape(generation.kind_text)}`",
        f"created: {html.escape(generation.created_at_text)}",
        f"checkpoint: `{html.escape(snapshot.checkpoint_name)}` / "
        f"VAE: `{html.escape(snapshot.vae_name or '内蔵')}`",
        f"seed: `{snapshot.seed}` / "
        f"size: `{snapshot.width} × {snapshot.height}` / steps: `{snapshot.steps}`",
        f"CFG: `{snapshot.cfg_scale:g}` / "
        f"sampler: `{html.escape(snapshot.sampler_name)}` / "
        f"scheduler: `{html.escape(snapshot.scheduler_name)}`",
        f"Positive prompt: {html.escape(snapshot.positive_prompt)}",
        f"Negative prompt: {html.escape(snapshot.negative_prompt)}",
        f"LoRA:\n{loras}",
    ]
    if generation.error_summary:
        values.append(f"error: {html.escape(generation.error_summary)}")
    return "\n\n".join(values)
