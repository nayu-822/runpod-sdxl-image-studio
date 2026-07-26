"""Gradio management tab backed by the LoRA catalog service."""

from __future__ import annotations

import html
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from uuid import UUID

import gradio as gr
from pydantic import ValidationError

from runpod_sdxl_image_studio.domain.lora_metadata import LoraMetadata, LoraMetadataUpdate
from runpod_sdxl_image_studio.domain.lora_search import LoraSearchQuery, LoraSort
from runpod_sdxl_image_studio.services.comfyui_service import ComfyUIService
from runpod_sdxl_image_studio.services.lora_catalog_service import (
    LoraCatalogError,
    LoraCatalogService,
)
from runpod_sdxl_image_studio.ui.components.lora_editor import render_state_updates


@dataclass(frozen=True)
class LoraManagementTabComponents:
    search: gr.Textbox
    category_filter: gr.Dropdown
    favorites_only: gr.Checkbox
    include_missing: gr.Checkbox
    sort: gr.Dropdown
    sync_button: gr.Button
    result_list: gr.Markdown
    selected: gr.Dropdown
    display_name: gr.Textbox
    category: gr.Textbox
    favorite: gr.Checkbox
    trigger_words: gr.Textbox
    recommended_model: gr.Number
    recommended_clip: gr.Number
    compatible_models: gr.Textbox
    notes: gr.Textbox
    thumbnail_upload: gr.File
    thumbnail_preview: gr.Image
    save_button: gr.Button
    delete_thumbnail_button: gr.Button
    message: gr.Markdown


def build_lora_management_tab(catalog: LoraCatalogService) -> LoraManagementTabComponents:
    del catalog
    gr.Markdown("## LoRA管理")
    with gr.Row():
        search = gr.Textbox(label="検索", placeholder="file name / display name / trigger / notes")
        category_filter = gr.Dropdown([], label="カテゴリ", allow_custom_value=False)
        sort = gr.Dropdown(
            choices=[
                ("お気に入り・最近使用", LoraSort.FAVORITES_RECENT.value),
                ("最近使用", LoraSort.RECENT.value),
                ("利用回数", LoraSort.USAGE.value),
                ("名前", LoraSort.NAME.value),
            ],
            value=LoraSort.FAVORITES_RECENT.value,
            label="並び順",
        )
    with gr.Row():
        favorites_only = gr.Checkbox(label="お気に入りのみ")
        include_missing = gr.Checkbox(label="missingを含める")
        sync_button = gr.Button("ComfyUI一覧と同期", variant="secondary")
    result_list = gr.Markdown("LoRA metadataはまだ同期されていません。")
    selected = gr.Dropdown([], label="編集対象LoRA", interactive=False)
    with gr.Accordion("metadata編集", open=False):
        display_name = gr.Textbox(label="表示名", max_lines=1)
        category = gr.Textbox(label="カテゴリ", max_lines=1)
        favorite = gr.Checkbox(label="お気に入り")
        trigger_words = gr.Textbox(
            label="トリガーワード（カンマ区切り）",
            lines=2,
            max_lines=5,
        )
        with gr.Row():
            recommended_model = gr.Number(label="推奨model strength", minimum=-2, maximum=2)
            recommended_clip = gr.Number(label="推奨clip strength", minimum=-2, maximum=2)
        compatible_models = gr.Textbox(label="対応モデル（カンマ区切り）", lines=2)
        notes = gr.Textbox(label="メモ", lines=4, max_lines=8)
        thumbnail_upload = gr.File(
            label="サムネイル（PNG/JPEG/WebP）",
            file_types=[".png", ".jpg", ".jpeg", ".webp"],
            type="binary",
        )
        thumbnail_preview = gr.Image(label="サムネイル", type="filepath")
        with gr.Row():
            save_button = gr.Button("metadataを保存", variant="primary")
            delete_thumbnail_button = gr.Button("サムネイルを削除")
    message = gr.Markdown("")
    return LoraManagementTabComponents(
        search,
        category_filter,
        favorites_only,
        include_missing,
        sort,
        sync_button,
        result_list,
        selected,
        display_name,
        category,
        favorite,
        trigger_words,
        recommended_model,
        recommended_clip,
        compatible_models,
        notes,
        thumbnail_upload,
        thumbnail_preview,
        save_button,
        delete_thumbnail_button,
        message,
    )


def make_search_handler(
    catalog: LoraCatalogService,
) -> Callable[..., tuple[object, ...]]:
    def handler(
        text: str,
        category: str | None,
        favorites_only: bool,
        include_missing: bool,
        sort: str,
        selected_id: str | None = None,
    ) -> tuple[object, ...]:
        try:
            results = catalog.search(
                LoraSearchQuery(
                    text=text or "",
                    category=category or None,
                    favorites_only=favorites_only,
                    include_missing=include_missing,
                    sort=sort,
                )
            )
            return build_catalog_list_updates(
                results,
                selected_id,
                catalog.categories(),
                category,
            )
        except (LoraCatalogError, ValidationError, ValueError):
            return "LoRA一覧を取得できませんでした。", gr.skip(), gr.skip()

    return handler


def make_sync_handler(
    comfyui: ComfyUIService,
    catalog: LoraCatalogService,
) -> Callable[..., Awaitable[tuple[object, ...]]]:
    async def handler() -> tuple[object, ...]:
        result = await comfyui.refresh_capabilities()
        if not result.is_success or result.capabilities is None:
            return result.message, gr.skip(), gr.skip(), gr.skip()
        try:
            catalog.sync_with_capabilities(result.capabilities.loras)
            results = catalog.search(LoraSearchQuery())
            return (result.message,) + build_catalog_list_updates(
                results,
                None,
                catalog.categories(),
            )
        except LoraCatalogError as exc:
            return str(exc), gr.skip(), gr.skip(), gr.skip()

    return handler


def make_select_handler(
    catalog: LoraCatalogService,
) -> Callable[[str | None], tuple[object, ...]]:
    def handler(selected: str | None) -> tuple[object, ...]:
        if not selected:
            return (gr.skip(),) * 9
        try:
            metadata_id = UUID(selected)
        except ValueError:
            return (gr.skip(),) * 9
        try:
            metadata = catalog.get_metadata(metadata_id)
        except LoraCatalogError:
            return (gr.skip(),) * 9
        if metadata is None:
            return (gr.skip(),) * 9
        preview = catalog.thumbnail_path(metadata.id)
        return (
            metadata.display_name or "",
            metadata.category or "",
            metadata.is_favorite,
            ", ".join(metadata.trigger_words),
            metadata.recommended_model_strength,
            metadata.recommended_clip_strength,
            ", ".join(metadata.compatible_models),
            metadata.notes or "",
            str(preview) if preview is not None else None,
        )

    return handler


def make_save_handler(
    catalog: LoraCatalogService,
    max_loras: int = 8,
) -> Callable[..., tuple[object, ...]]:
    def handler(
        selected: str | None,
        display_name: str,
        category: str,
        favorite: bool,
        trigger_words: str,
        recommended_model: float | None,
        recommended_clip: float | None,
        compatible_models: str,
        notes: str,
        search_text: str = "",
        search_category: str | None = None,
        favorites_only: bool = False,
        include_missing: bool = False,
        sort: str = LoraSort.FAVORITES_RECENT.value,
        generation_category: str | None = None,
        generation_state: object = None,
    ) -> tuple[object, ...]:
        if not selected:
            return (
                "LoRAを選択してください。",
                *metadata_save_preserve_updates(max_loras),
            )
        try:
            metadata = catalog.update_metadata(
                UUID(selected),
                LoraMetadataUpdate(
                    display_name=display_name,
                    category=category,
                    is_favorite=favorite,
                    trigger_words=trigger_words,
                    recommended_model_strength=recommended_model,
                    recommended_clip_strength=recommended_clip,
                    compatible_models=compatible_models,
                    notes=notes,
                ),
            )
            return _metadata_change_updates(
                catalog,
                f"保存しました: {metadata.file_name}",
                selected,
                search_text,
                search_category,
                favorites_only,
                include_missing,
                sort,
                generation_category,
                generation_state,
                max_loras,
            )
        except (LoraCatalogError, ValidationError, ValueError):
            return (
                "入力値を確認してください。",
                *metadata_save_preserve_updates(max_loras),
            )

    return handler


def make_favorite_handler(
    catalog: LoraCatalogService,
    max_loras: int = 8,
) -> Callable[..., tuple[object, ...]]:
    def handler(
        selected: str | None,
        favorite: bool,
        search_text: str = "",
        search_category: str | None = None,
        favorites_only: bool = False,
        include_missing: bool = False,
        sort: str = LoraSort.FAVORITES_RECENT.value,
        generation_category: str | None = None,
        generation_state: object = None,
    ) -> tuple[object, ...]:
        output_count = 8 + 7 * max_loras
        if not selected:
            return (gr.skip(), "LoRAを選択してください。") + (gr.skip(),) * (output_count - 2)
        try:
            metadata_id = UUID(selected)
        except ValueError:
            return (
                gr.Checkbox(value=not favorite),
                "選択対象のUUIDが不正です。",
            ) + (gr.skip(),) * (output_count - 2)
        try:
            catalog.set_favorite(metadata_id, favorite)
            updates = _metadata_change_updates(
                catalog,
                "お気に入りを保存しました。",
                selected,
                search_text,
                search_category,
                favorites_only,
                include_missing,
                sort,
                generation_category,
                generation_state,
                max_loras,
            )
            return (gr.Checkbox(value=favorite),) + updates
        except (LoraCatalogError, ValueError):
            try:
                current = catalog.get_metadata(metadata_id)
            except Exception:  # noqa: BLE001 - do not expose database details in the UI
                current = None
            return (
                gr.Checkbox(value=current.is_favorite if current else not favorite),
                "保存に失敗しました。",
                *(gr.skip() for _ in range(output_count - 2)),
            )

    return handler


def make_thumbnail_save_handler(
    catalog: LoraCatalogService,
) -> Callable[[str | None, bytes | None], tuple[object, ...]]:
    def handler(selected: str | None, payload: bytes | None) -> tuple[object, ...]:
        if not selected or payload is None:
            return "LoRAと画像を選択してください。", gr.skip()
        try:
            metadata = catalog.save_thumbnail(UUID(selected), payload)
            path = catalog.thumbnail_path(metadata.id)
            return "サムネイルを保存しました。", str(path) if path else None
        except (LoraCatalogError, ValueError):
            return "サムネイルを保存できませんでした。", gr.skip()

    return handler


def make_thumbnail_delete_handler(
    catalog: LoraCatalogService,
) -> Callable[[str | None], tuple[object, ...]]:
    def handler(selected: str | None) -> tuple[object, ...]:
        if not selected:
            return "LoRAを選択してください。", gr.skip()
        try:
            catalog.delete_thumbnail(UUID(selected))
            return "サムネイルを削除しました。", None
        except (LoraCatalogError, ValueError):
            return "サムネイルを削除できませんでした。", gr.skip()

    return handler


def build_catalog_list_updates(
    results: Sequence[LoraMetadata],
    selected_id: str | None,
    categories: Sequence[str] = (),
    category: str | None = None,
) -> tuple[object, ...]:
    """Build the shared result, selection, and category updates."""

    options = tuple(
        (
            f"{item.display_name} — {item.file_name}" if item.display_name else item.file_name,
            str(item.id),
        )
        for item in results
    )
    selected_value = selected_id if selected_id in {value for _, value in options} else None
    category_values = tuple(categories)
    return (
        _render_results(results),
        gr.Dropdown(
            choices=list(options),
            value=selected_value,
            interactive=bool(options),
        ),
        gr.Dropdown(
            choices=list(category_values),
            value=category if category in category_values else None,
            interactive=bool(category_values),
        ),
    )


def metadata_save_preserve_updates(max_loras: int) -> tuple[object, ...]:
    """Preserve every save-event output except its user-facing message."""

    return (
        gr.skip(),  # result_list
        gr.skip(),  # selected
        gr.skip(),  # category_filter
        gr.skip(),  # generation choices
        gr.skip(),  # generation state
        *(gr.skip() for _ in range(7 * max_loras)),
        gr.skip(),  # add button
    )


def _metadata_change_updates(
    catalog: LoraCatalogService,
    message: str,
    selected_id: str,
    search_text: str,
    search_category: str | None,
    favorites_only: bool,
    include_missing: bool,
    sort: str,
    generation_category: str | None,
    generation_state: object,
    max_loras: int,
) -> tuple[object, ...]:
    results = catalog.search(
        LoraSearchQuery(
            text=search_text or "",
            category=search_category or None,
            favorites_only=favorites_only,
            include_missing=include_missing,
            sort=sort,
        )
    )
    list_updates = build_catalog_list_updates(
        results,
        selected_id,
        catalog.categories(),
        search_category,
    )
    generation_updates = _generation_lora_updates(
        catalog,
        generation_category,
        generation_state,
        max_loras,
    )
    return (message, *list_updates, *generation_updates)


def _generation_lora_updates(
    catalog: LoraCatalogService,
    category: str | None,
    state: object,
    max_loras: int,
) -> tuple[object, ...]:
    choices = catalog.selector_options(category or None)
    return (list(choices),) + render_state_updates(
        state,
        choices,
        max_loras,
        clear_unavailable=True,
    )


def _render_results(metadata: Sequence[LoraMetadata]) -> str:
    if not metadata:
        return "該当するLoRAはありません。"
    lines = []
    for item in metadata:
        display = html.escape(item.display_name or item.file_name)
        file_name = html.escape(item.file_name)
        category = html.escape(item.category or "未分類")
        missing = " / missing" if item.is_missing else ""
        lines.append(
            f"- **{display}** — `{file_name}` / {category} / "
            f"favorite={item.is_favorite} / usage={item.usage_count}{missing}"
        )
    return "\n".join(lines)
