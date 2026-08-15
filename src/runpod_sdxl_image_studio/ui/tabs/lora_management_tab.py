"""Gradio management tab backed by the LoRA catalog service."""

from __future__ import annotations

import html
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
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
from runpod_sdxl_image_studio.ui.components.lora_editor import (
    LoraEditorComponents,
    component_output_count_for_rows,
    normalize_lora_state,
    preserve_component_updates,
    render_state_updates,
    update_lora_row,
)

_THUMBNAIL_PLACEHOLDER = (
    Path(__file__).resolve().parents[1] / "assets" / "thumbnail_placeholder.svg"
)


@dataclass(frozen=True)
class LoraManagementTabComponents:
    search: gr.Textbox
    category_filter: gr.Dropdown
    favorites_only: gr.Checkbox
    include_missing: gr.Checkbox
    sort: gr.Dropdown
    sync_button: gr.Button
    result_list: gr.Markdown
    result_gallery: gr.Gallery
    gallery_items: gr.State
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
    add_to_generation_button: gr.Button
    message: gr.Markdown


def build_lora_management_tab(catalog: LoraCatalogService) -> LoraManagementTabComponents:
    del catalog
    gr.Markdown("## LoRA")
    with gr.Row(elem_classes=["lora-catalog-toolbar"]):
        search = gr.Textbox(
            label="検索",
            placeholder="名前、カテゴリ、トリガーワードで検索",
            scale=3,
        )
        category_filter = gr.Dropdown([], label="カテゴリ", allow_custom_value=False, scale=1)
        sort = gr.Dropdown(
            choices=[
                ("すべて", LoraSort.NAME.value),
                ("お気に入り", LoraSort.FAVORITES_RECENT.value),
                ("最近使った", LoraSort.RECENT.value),
            ],
            value=LoraSort.NAME.value,
            label="表示",
            scale=1,
        )
    with gr.Row(elem_classes=["lora-catalog-toolbar"]):
        favorites_only = gr.Checkbox(label="お気に入りのみ", visible=False)
        include_missing = gr.Checkbox(label="利用不可も表示", value=True, visible=False)
        sync_button = gr.Button("ComfyUI一覧を更新", variant="secondary")
    result_gallery = gr.Gallery(
        label="LoRA",
        columns=[1, 2, 3, 4, 5],
        rows=2,
        object_fit="cover",
        allow_preview=False,
        show_label=False,
        elem_classes=["lora-catalog-gallery"],
    )
    gallery_items = gr.State([])
    result_list = gr.Markdown("LoRA一覧はまだ同期されていません。", visible=False)
    selected = gr.Dropdown([], label="選択中LoRA", interactive=False, visible=False)
    with gr.Accordion("選択中LoRAの詳細・管理", open=False, elem_classes=["lora-detail"]):
        add_to_generation_button = gr.Button(
            "生成に追加", variant="primary", elem_classes=["mobile-tap-button"]
        )
        display_name = gr.Textbox(label="表示名", max_lines=1)
        category = gr.Textbox(label="カテゴリ", max_lines=1)
        favorite = gr.Checkbox(label="お気に入り")
        trigger_words = gr.Textbox(
            label="トリガーワード（カンマ区切り）",
            lines=2,
            max_lines=5,
        )
        with gr.Row(elem_classes=["metadata-actions"]):
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
        with gr.Row(elem_classes=["metadata-actions"]):
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
        result_gallery,
        gallery_items,
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
        add_to_generation_button,
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
                    favorites_only=(favorites_only or sort == LoraSort.FAVORITES_RECENT.value),
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


def make_gallery_handler(
    catalog: LoraCatalogService,
) -> Callable[..., tuple[object, object]]:
    """Return the image-first gallery and its opaque server-side selection map."""

    def handler(
        text: str,
        category: str | None,
        favorites_only: bool,
        include_missing: bool,
        sort: str,
    ) -> tuple[object, object]:
        try:
            results = catalog.search(
                LoraSearchQuery(
                    text=text or "",
                    category=category or None,
                    favorites_only=(favorites_only or sort == LoraSort.FAVORITES_RECENT.value),
                    include_missing=include_missing,
                    sort=sort,
                )
            )
        except (LoraCatalogError, ValidationError, ValueError):
            return gr.Gallery(value=[]), []
        return _gallery_updates(results, catalog), [str(item.id) for item in results]

    return handler


def make_sync_gallery_handler(
    catalog: LoraCatalogService,
) -> Callable[[], tuple[object, object]]:
    """Refresh the gallery after a capability sync without exposing metadata IDs."""

    def handler() -> tuple[object, object]:
        try:
            results = catalog.search(LoraSearchQuery(include_missing=True))
            return _gallery_updates(results, catalog), [str(item.id) for item in results]
        except LoraCatalogError:
            return gr.Gallery(value=[]), []

    return handler


def make_gallery_select_handler() -> Callable[[object, object], object]:
    """Resolve a Gallery index through the current server-side result map."""

    def handler(items: object, event: object = None) -> str | None:
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
            return None
        index = getattr(event, "index", event)
        if isinstance(index, tuple):
            index = index[0] if index else None
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(items):
            return None
        selected = items[index]
        return selected if isinstance(selected, str) else None

    return handler


def make_add_to_generation_handler(
    catalog: LoraCatalogService,
    max_loras: int,
) -> Callable[[str | None, object, object], tuple[object, ...]]:
    """Add one available catalog item to the existing ordered Generation editor."""

    def handler(
        selected: str | None,
        state: object,
        choices: object,
    ) -> tuple[object, ...]:
        def preserve(message: str) -> tuple[object, ...]:
            return (
                message,
                *render_state_updates(state, choices, max_loras),
                gr.Button(interactive=False),
            )

        if not selected:
            return preserve("LoRAを選択してください。")
        try:
            metadata = catalog.get_metadata(UUID(selected))
        except (LoraCatalogError, ValueError):
            return preserve("LoRAを取得できませんでした。")
        if metadata is None:
            return preserve("LoRAを取得できませんでした。")
        if metadata.is_missing:
            return preserve("現在利用できないLoRAは生成に追加できません。")
        try:
            rows = normalize_lora_state(state, max_loras)
            existing = {
                row["lora_name"]
                for row in rows
                if isinstance(row.get("lora_name"), str) and row["lora_name"]
            }
            if metadata.file_name in existing:
                return preserve("このLoRAはすでに生成へ追加されています。")
            if len(existing) >= max_loras:
                return preserve("LoRAの最大数に達しています。")
            target = next(
                (index for index, row in enumerate(rows) if not row.get("lora_name")),
                None,
            )
            if target is None:
                rows = rows + [{"lora_name": None}]
                target = len(rows) - 1
            updated = update_lora_row(
                rows,
                target,
                metadata.file_name,
                metadata.recommended_model_strength
                if metadata.recommended_model_strength is not None
                else 1.0,
                metadata.recommended_clip_strength
                if metadata.recommended_clip_strength is not None
                else 1.0,
                max_loras,
                False,
            )
            return (
                "生成に追加しました。",
                *render_state_updates(updated, choices, max_loras),
                gr.Button(interactive=len(updated) < max_loras),
            )
        except (TypeError, ValueError):
            return preserve("LoRAを生成へ追加できませんでした。")

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
    *,
    lora_editor: LoraEditorComponents | None = None,
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
                *metadata_save_preserve_updates(max_loras, lora_editor=lora_editor),
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
                *metadata_save_preserve_updates(max_loras, lora_editor=lora_editor),
            )

    return handler


def make_favorite_handler(
    catalog: LoraCatalogService,
    max_loras: int = 8,
    *,
    lora_editor: LoraEditorComponents | None = None,
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
        if not selected:
            return (
                gr.skip(),
                "LoRAを選択してください。",
                *metadata_favorite_preserve_updates(max_loras, lora_editor=lora_editor),
            )
        try:
            metadata_id = UUID(selected)
        except ValueError:
            return (
                gr.Checkbox(value=not favorite),
                "選択対象のUUIDが不正です。",
                *metadata_favorite_preserve_updates(max_loras, lora_editor=lora_editor),
            )
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
                *metadata_favorite_preserve_updates(max_loras, lora_editor=lora_editor),
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


def _gallery_updates(results: Sequence[LoraMetadata], catalog: LoraCatalogService) -> gr.Gallery:
    cards: list[tuple[str, str]] = []
    for item in results:
        thumbnail = catalog.thumbnail_path(item.id)
        path = thumbnail if thumbnail is not None and thumbnail.exists() else _THUMBNAIL_PLACEHOLDER
        display = html.escape(item.display_name or item.file_name)
        category = html.escape(item.category or "未分類")
        availability = "利用不可" if item.is_missing else "利用可能"
        favorite = "★" if item.is_favorite else "♡"
        cards.append((str(path), f"{favorite} {display}\n{category} · {availability}"))
    return gr.Gallery(value=cards)


def metadata_save_preserve_updates(
    max_loras: int | None = None,
    *,
    lora_editor: LoraEditorComponents | None = None,
) -> tuple[object, ...]:
    """Preserve every save-event output except its user-facing message."""

    row_updates = (
        preserve_component_updates(lora_editor)
        if lora_editor is not None
        else tuple(gr.skip() for _ in range(component_output_count_for_rows(max_loras or 0)))
    )
    return (
        gr.skip(),  # result_list
        gr.skip(),  # selected
        gr.skip(),  # category_filter
        gr.skip(),  # generation choices
        gr.skip(),  # generation state
        *row_updates,
        gr.skip(),  # add button
    )


def metadata_favorite_preserve_updates(
    max_loras: int | None = None,
    *,
    lora_editor: LoraEditorComponents | None = None,
) -> tuple[object, ...]:
    """Preserve the shared outputs after favorite's checkbox and message."""

    return metadata_save_preserve_updates(max_loras, lora_editor=lora_editor)


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
