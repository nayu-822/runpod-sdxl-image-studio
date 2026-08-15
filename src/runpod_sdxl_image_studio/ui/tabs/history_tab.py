"""Mobile-friendly generation history tab and UI boundary handlers."""

from __future__ import annotations

import html
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TypeVar
from uuid import UUID
from zoneinfo import ZoneInfo

import gradio as gr

from runpod_sdxl_image_studio.domain.generation import GenerationKind, GenerationStatus
from runpod_sdxl_image_studio.domain.generation_diff import GenerationDiff, PromptTokenChange
from runpod_sdxl_image_studio.domain.generation_history import (
    GenerationDetailView,
    GenerationHistoryItem,
    GenerationHistoryQuery,
    GenerationHistorySort,
    LoraSearchMode,
)
from runpod_sdxl_image_studio.services.generation_diff_service import (
    GenerationDiffError,
    GenerationDiffService,
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

_HistoryEnum = TypeVar("_HistoryEnum", bound=StrEnum)
_THUMBNAIL_PLACEHOLDER = (
    Path(__file__).resolve().parents[1] / "assets" / "thumbnail_placeholder.svg"
)


@dataclass(frozen=True)
class HistoryTabComponents:
    refresh_button: gr.Button
    clear_button: gr.Button
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
    search_text: gr.Textbox
    status_search: gr.Dropdown
    kind_search: gr.Dropdown
    parent_search: gr.Textbox
    date_from_search: gr.Textbox
    date_to_search: gr.Textbox
    checkpoint_search: gr.Textbox
    vae_search: gr.Textbox
    lora_search: gr.Textbox
    lora_search_mode: gr.Dropdown
    seed_search: gr.Number
    width_search: gr.Number
    height_search: gr.Number
    error_code_search: gr.Textbox
    sort_search: gr.Dropdown
    query_summary: gr.Markdown
    seed_copy: gr.Textbox
    diff_button: gr.Button
    diff_view: gr.Markdown


def begin_regeneration() -> tuple[gr.Button, bool]:
    """Disable regeneration immediately and mark the chained request."""

    return gr.Button(value="再生成中...", interactive=False), True


def enable_regeneration_button() -> gr.Button:
    """Restore regeneration after all success and failure paths."""

    return gr.Button(value="同条件で再生成", interactive=True)


def build_history_tab() -> HistoryTabComponents:
    gr.Markdown("## 生成履歴")
    with gr.Accordion("高度な履歴検索", open=False, elem_classes=["advanced-history-filter"]):
        search_text = gr.Textbox(label="検索テキスト", lines=2, max_lines=4, max_length=500)
        with gr.Row(elem_classes=["history-filter"]):
            date_from_search = gr.Textbox(label="開始日（YYYY-MM-DD）")
            date_to_search = gr.Textbox(label="終了日（YYYY-MM-DD）")
        with gr.Row(elem_classes=["history-filter"]):
            status_search = gr.Dropdown(
                [(status.value, status.value) for status in GenerationStatus],
                value=[],
                multiselect=True,
                label="status（複数可）",
            )
            kind_search = gr.Dropdown(
                [(kind.value, kind.value) for kind in GenerationKind],
                value=[],
                multiselect=True,
                label="kind（複数可）",
            )
        with gr.Row(elem_classes=["history-filter"]):
            checkpoint_search = gr.Textbox(label="checkpoint（単一指定）")
            vae_search = gr.Textbox(label="VAE（単一指定）")
        with gr.Row(elem_classes=["history-filter"]):
            lora_search = gr.Textbox(label="LoRA（カンマ区切り）")
            lora_search_mode = gr.Dropdown(
                [
                    ("いずれかを含む", LoraSearchMode.ANY.value),
                    ("すべてを含む", LoraSearchMode.ALL.value),
                ],
                value=LoraSearchMode.ANY.value,
                label="LoRA検索方式",
            )
        with gr.Row(elem_classes=["history-filter"]):
            seed_search = gr.Number(label="seed", precision=0)
            width_search = gr.Number(label="幅", precision=0)
            height_search = gr.Number(label="高さ", precision=0)
        parent_search = gr.Textbox(label="親Generation ID")
        error_code_search = gr.Textbox(label="error code（カンマ区切り）")
        sort_search = gr.Dropdown(
            [
                ("新しい順", GenerationHistorySort.NEWEST.value),
                ("古い順", GenerationHistorySort.OLDEST.value),
                ("seed昇順", GenerationHistorySort.SEED_ASC.value),
                ("seed降順", GenerationHistorySort.SEED_DESC.value),
                ("解像度が大きい順", GenerationHistorySort.RESOLUTION_DESC.value),
                ("最近完了した順", GenerationHistorySort.RECENTLY_COMPLETED.value),
            ],
            value=GenerationHistorySort.NEWEST.value,
            label="並び順",
        )
    with gr.Row(elem_classes=["history-filter"]):
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
    with gr.Row(elem_classes=["history-actions"]):
        refresh_button = gr.Button("履歴を検索", min_width=140)
        clear_button = gr.Button("検索条件をクリア")
    page_state = gr.State(1)
    cards = gr.Markdown("履歴を読み込んでください。")
    thumbnail_gallery = gr.Gallery(
        label="履歴サムネイル",
        columns=4,
        rows=2,
        object_fit="contain",
        height="auto",
        elem_classes=["history-gallery"],
    )
    with gr.Row(elem_classes=["history-actions"]):
        previous_button = gr.Button("前へ", interactive=False)
        page = gr.Markdown("1ページ目")
        next_button = gr.Button("次へ", interactive=False)
    selected = gr.Dropdown([], label="選択中Generation", allow_custom_value=False)
    detail = gr.Markdown("")
    image = gr.Image(label="履歴画像", type="filepath")
    seed_copy = gr.Textbox(
        label="実使用seed（選択してコピー）",
        interactive=False,
        show_copy_button=True,
        max_length=20,
    )
    favorite = gr.Checkbox(label="お気に入り")
    note = gr.Textbox(label="メモ", lines=3, max_lines=8, max_length=2000)
    with gr.Row(elem_classes=["history-actions"]):
        save_note_button = gr.Button("メモを保存")
        restore_button = gr.Button("設定を生成画面へ復元")
        regenerate_button = gr.Button("同条件で再生成")
    message = gr.Markdown("")
    with gr.Row(elem_classes=["history-actions"]):
        diff_button = gr.Button("親Generationとの差分")
        diff_view = gr.Markdown("")
    query_summary = gr.Markdown("検索条件: 全件")
    return HistoryTabComponents(
        refresh_button,
        clear_button,
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
        search_text,
        status_search,
        kind_search,
        parent_search,
        date_from_search,
        date_to_search,
        checkpoint_search,
        vae_search,
        lora_search,
        lora_search_mode,
        seed_search,
        width_search,
        height_search,
        error_code_search,
        sort_search,
        query_summary,
        seed_copy,
        diff_button,
        diff_view,
    )


def build_advanced_history_query(
    text: str | None = None,
    date_from_text: str | None = None,
    date_to_text: str | None = None,
    checkpoint: str | None = None,
    vae: str | None = None,
    loras: str | None = None,
    lora_mode: str | None = None,
    seed: float | int | None = None,
    width: float | int | None = None,
    height: float | int | None = None,
    error_code: str | None = None,
    sort: str | None = None,
    statuses: object = None,
    kinds: object = None,
    parent_generation_id: str | None = None,
) -> GenerationHistoryQuery:
    """UI入力を型付き検索条件へ変換する。"""

    def integer(value: float | int | None) -> int | None:
        return int(value) if value is not None else None

    status_values = _enum_values(statuses, GenerationStatus)
    kind_values = _enum_values(kinds, GenerationKind)
    parent = (
        UUID(parent_generation_id.strip())
        if parent_generation_id and parent_generation_id.strip()
        else None
    )
    return GenerationHistoryQuery(
        text=text,
        date_from=_tokyo_date_start(date_from_text),
        date_to=_tokyo_date_end(date_to_text),
        checkpoint_names=(checkpoint,) if checkpoint else (),
        vae_names=(vae,) if vae else (),
        lora_names=tuple(value.strip() for value in (loras or "").split(",") if value.strip()),
        lora_search_mode=LoraSearchMode(lora_mode or LoraSearchMode.ANY.value),
        seed=integer(seed),
        width=integer(width),
        height=integer(height),
        error_codes=tuple(
            value.strip() for value in (error_code or "").split(",") if value.strip()
        ),
        statuses=status_values,
        kinds=kind_values,
        parent_generation_id=parent,
        sort=GenerationHistorySort(sort or GenerationHistorySort.NEWEST.value),
    )


def _enum_values(value: object, enum_type: type[_HistoryEnum]) -> tuple[_HistoryEnum, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(enum_type(item) for item in value if isinstance(item, str) and item)


def _tokyo_date_start(value: str | None) -> datetime | None:
    if not value or not value.strip():
        return None
    local = datetime.combine(date.fromisoformat(value.strip()), time.min, ZoneInfo("Asia/Tokyo"))
    return local.astimezone(UTC)


def _tokyo_date_end(value: str | None) -> datetime | None:
    if not value or not value.strip():
        return None
    local = datetime.combine(date.fromisoformat(value.strip()), time.min, ZoneInfo("Asia/Tokyo"))
    return (local + timedelta(days=1)).astimezone(UTC)


def make_history_refresh_handler(
    service: GenerationHistoryService,
    recovery_service: GenerationRecoveryService | None = None,
    *,
    reset_page: bool = False,
) -> Callable[..., Awaitable[tuple[object, ...]]]:
    async def handler(
        page: int = 1,
        date_text: str = "",
        status: str | None = None,
        kind: str | None = None,
        favorite: str | None = None,
        search_text: str | None = None,
        statuses: object = None,
        kinds: object = None,
        parent_generation_id: str | None = None,
        date_from_text: str | None = None,
        date_to_text: str | None = None,
        checkpoint: str | None = None,
        vae: str | None = None,
        loras: str | None = None,
        lora_mode: str | None = None,
        seed: float | int | None = None,
        width: float | int | None = None,
        height: float | int | None = None,
        error_code: str | None = None,
        sort: str | None = None,
    ) -> tuple[object, ...]:
        if recovery_service is not None:
            await recovery_service.recover()
        try:
            normalized_page = 1 if reset_page else max(1, int(page))
            selected_date = date.fromisoformat(date_text) if date_text.strip() else None
            advanced_query = build_advanced_history_query(
                search_text,
                date_from_text,
                date_to_text,
                checkpoint,
                vae,
                loras,
                lora_mode,
                seed,
                width,
                height,
                error_code,
                sort,
                statuses,
                kinds,
                parent_generation_id,
            )
            if not advanced_query.statuses and status:
                advanced_query = replace(advanced_query, statuses=(GenerationStatus(status),))
            if not advanced_query.kinds and kind:
                advanced_query = replace(advanced_query, kinds=(GenerationKind(kind),))
            result = service.list_history(
                replace(
                    advanced_query,
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
                render_query_summary(advanced_query),
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
                gr.skip(),
                "",
            )

    return handler


def make_generation_diff_handler(
    history_service: GenerationHistoryService,
    diff_service: GenerationDiffService,
) -> Callable[[str | None], str]:
    """Render a safe parent-generation diff for the selected item."""

    def handler(selected: str | None) -> str:
        if not selected:
            return "Generationを選択してください。"
        try:
            target = history_service.get_generation(UUID(selected))
            if target.parent_generation_id is None:
                return "親Generationがありません。"
            source = history_service.get_generation(target.parent_generation_id)
            diff = diff_service.compare(source, target)
            return render_generation_diff(diff)
        except (GenerationHistoryError, GenerationDiffError, ValueError) as exc:
            return str(exc)

    return handler


def previous_history_page(page: int) -> int:
    return max(1, int(page) - 1)


def next_history_page(page: int) -> int:
    return max(1, int(page) + 1)


def clear_history_filters() -> tuple[object, ...]:
    """高度検索・基本検索・選択状態をまとめて初期化する。"""

    return (
        1,
        "",
        "",
        "",
        "",
        "",
        [],
        [],
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        [],
        None,
        None,
        "",
        GenerationHistorySort.NEWEST.value,
        "検索条件: 全件",
        gr.Dropdown(value=None),
        "",
        "",
    )


def make_history_detail_handler(
    service: GenerationHistoryService,
) -> Callable[[str | None], tuple[object, ...]]:
    def handler(selected: str | None) -> tuple[object, ...]:
        if not selected:
            return "", None, "", gr.skip(), gr.skip(), "履歴を選択してください。"
        try:
            detail = service.get_detail(UUID(selected))
            image_path = service.absolute_data_path(detail.image_path)
            return (
                render_generation_detail(detail),
                str(image_path) if image_path else None,
                seed_copy_value(detail.snapshot.seed),
                detail.favorite,
                detail.user_note or "",
                "",
            )
        except (GenerationHistoryError, ValueError) as exc:
            return "", None, "", gr.skip(), gr.skip(), str(exc)

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
                + (gr.skip(),) * (23 + component_output_count(max_loras))
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
                gr.Dropdown(
                    value=settings.final_upscale_model,
                    visible=settings.final_upscale,
                ),
                settings.clip_skip,
                settings.hires_fix,
                settings.hires_scale,
                settings.hires_resize_method,
                settings.hires_steps,
                settings.hires_cfg_scale,
                settings.hires_sampler_name,
                settings.hires_scheduler_name,
                settings.hires_denoise,
                settings.final_upscale,
                *lora_updates,
                str(restored.parent_generation_id),
                not blocking,
            )
        except (GenerationHistoryError, ValueError):
            return (
                ("設定を復元できませんでした。",)
                + (gr.skip(),) * (23 + component_output_count(max_loras))
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


def seed_copy_value(seed: int) -> str:
    """履歴snapshotの実使用seedを整数文字列として返す。"""

    return str(int(seed))


def render_query_summary(query: GenerationHistoryQuery) -> str:
    """現在の検索条件をUIへ安全に要約する。"""

    values = [f"sort={query.sort.value}"]
    if query.text:
        values.append(f"text={query.text}")
    if query.statuses:
        values.append("status=" + ",".join(value.value for value in query.statuses))
    if query.kinds:
        values.append("kind=" + ",".join(value.value for value in query.kinds))
    if query.parent_generation_id:
        values.append(f"parent={query.parent_generation_id}")
    if query.lora_names:
        values.append(f"LoRA({query.lora_search_mode.value})=" + ",".join(query.lora_names))
    return "検索条件: " + html.escape(" / ".join(values))


def render_history_thumbnails(
    service: GenerationHistoryService,
    items: tuple[GenerationHistoryItem, ...],
) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for item in items:
        path = service.absolute_data_path(item.thumbnail_path)
        if path is None:
            path = _THUMBNAIL_PLACEHOLDER
            label = f"{item.created_at_text} / サムネイル未生成 / seed {item.seed_text}"
        else:
            label = f"{item.created_at_text} / {item.status_text} / seed {item.seed_text}"
        values.append((str(path), label))
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
    values.extend(html.escape(warning) for warning in generation.restore_warnings)
    return "\n\n".join(values)


def render_generation_diff(diff: GenerationDiff) -> str:
    """Render typed diff values as escaped Markdown text."""

    def tokens(items: tuple[PromptTokenChange, ...]) -> str:
        return (
            ", ".join(
                f"{html.escape(item.value)} ({html.escape(item.change_type.value)})"
                for item in items
            )
            or "変更なし"
        )

    lines = [
        f"### Prompt差分 `{html.escape(str(diff.source_generation_id))}` → "
        f"`{html.escape(str(diff.target_generation_id))}`",
        f"Positive: {tokens(diff.positive_prompt_changes)}",
        f"Negative: {tokens(diff.negative_prompt_changes)}",
    ]
    for setting_change in diff.setting_changes:
        lines.append(
            f"- {html.escape(setting_change.field_name)}: "
            f"`{html.escape(str(setting_change.before))}` → "
            f"`{html.escape(str(setting_change.after))}`"
        )
    for lora_change in diff.lora_changes:
        lines.append(
            f"- LoRA {html.escape(lora_change.name)}: {html.escape(lora_change.change_type.value)}"
        )
    return "\n\n".join(lines)
