"""Gradio selection-only UI for Phase 5 upscale enqueueing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import gradio as gr

from runpod_sdxl_image_studio.adapters.catalog.upscaler_catalog import UpscalerCatalog
from runpod_sdxl_image_studio.domain.upscale import (
    UpscaleMethod,
    UpscaleSettings,
    UpscaleSizingMode,
)
from runpod_sdxl_image_studio.services.generation_preflight_service import (
    GenerationPreflightService,
)
from runpod_sdxl_image_studio.services.upscale_enqueue_service import (
    UpscaleEnqueueError,
    UpscaleEnqueueService,
)
from runpod_sdxl_image_studio.ui.view_models import preflight_markdown

logger = logging.getLogger(__name__)

_UPSCALE_INPUT_ERROR = "入力内容または親画像を確認してください。"
_PARENT_SELECTION_ERROR = "親画像を選択できません。完了済みの画像を指定してください。"
_RESULT_ERROR = "アップスケール結果を表示できません。"
_INTERNAL_ERROR = "アップスケール処理中に内部エラーが発生しました。"


@dataclass(frozen=True)
class UpscaleTabComponents:
    parent_generation_id: Any
    source_import_id: Any
    latest_button: Any
    source_preview: Any
    method: Any
    sizing_mode: Any
    scale_factor: Any
    target_width: Any
    target_height: Any
    upscaler_name: Any
    catalog_message: Any
    denoise: Any
    plan: Any
    enqueue_button: Any
    status: Any
    result: Any
    comparison: Any


def build_upscale_tab(
    upscaler_catalog: UpscalerCatalog | tuple[str, ...] | None = (),
) -> UpscaleTabComponents:
    if isinstance(upscaler_catalog, UpscalerCatalog):
        upscaler_choices = upscaler_catalog.models or ()
        catalog_text = (
            "upscalerカタログを取得できません。設定ディレクトリを確認してください。"
            if upscaler_catalog.models is None
            else "upscalerカタログは取得済みですが、利用可能なモデルがありません。"
            if not upscaler_catalog.models
            else f"upscalerカタログ: {len(upscaler_catalog.models)}件"
        )
    else:
        upscaler_choices = upscaler_catalog or ()
        catalog_text = (
            "upscalerカタログは未取得です。"
            if upscaler_catalog is None
            else f"upscalerカタログ: {len(upscaler_choices)}件"
        )
    with gr.Row(elem_classes=["comparison-layout"]):
        parent_id = gr.Textbox(label="親Generation ID", placeholder="completed generation UUID")
        source_import_id = gr.Textbox(
            label="外部Import ID",
            placeholder="metadata import UUID（外部画像用）",
            visible=False,
        )
        source_preview = gr.Image(label="親画像", interactive=False, type="filepath")
    method = gr.Radio(
        [
            ("画像アップスケール", UpscaleMethod.IMAGE.value),
            ("Latentアップスケール", UpscaleMethod.LATENT.value),
        ],
        value=UpscaleMethod.IMAGE.value,
        label="方式",
    )
    sizing = gr.Radio(
        [("倍率", UpscaleSizingMode.FACTOR.value), ("寸法", UpscaleSizingMode.DIMENSIONS.value)],
        value=UpscaleSizingMode.FACTOR.value,
        label="出力サイズ",
    )
    with gr.Row(elem_classes=["size-dimensions"]):
        factor = gr.Number(label="倍率", value=2.0, minimum=1.01, maximum=16.0)
        width = gr.Number(label="幅", value=1024, minimum=64, precision=0)
        height = gr.Number(label="高さ", value=1024, minimum=64, precision=0)
    catalog_message = gr.Markdown(catalog_text)
    upscaler = gr.Dropdown(list(upscaler_choices), label="Upscaler", allow_custom_value=False)
    denoise = gr.Slider(0, 1, value=0.35, step=0.01, label="Denoise（Latentのみ）")
    plan = gr.Markdown("出力サイズと負荷見積もりは親画像確認後に表示されます。")
    enqueue_button = gr.Button(
        "アップスケールをキューへ追加",
        variant="primary",
        elem_classes=["mobile-tap-button"],
    )
    status = gr.Markdown()
    result = gr.Image(label="結果", interactive=False)
    comparison = gr.Gallery(
        label="親画像と結果の比較",
        columns=2,
        rows=1,
        elem_classes=["comparison-gallery"],
    )
    return UpscaleTabComponents(
        parent_id,
        source_import_id,
        gr.Button("最新の完了画像を選択"),
        source_preview,
        method,
        sizing,
        factor,
        width,
        height,
        upscaler,
        catalog_message,
        denoise,
        plan,
        enqueue_button,
        status,
        result,
        comparison,
    )


def make_latest_parent_handler(
    service: UpscaleEnqueueService,
) -> Callable[[], tuple[str, str]]:
    def handler() -> tuple[str, str]:
        generation_id = service.latest_completed_generation_id()
        if generation_id is None:
            return "", "完了済みの一次画像がありません。"
        return str(generation_id), f"親画像を選択しました: `{generation_id}`"

    return handler


def make_latest_parent_selection_handler(
    service: UpscaleEnqueueService,
) -> Callable[[], tuple[str, object, str]]:
    def handler() -> tuple[str, object, str]:
        generation_id = service.latest_completed_generation_id()
        if generation_id is None:
            return "", None, "完了済みの画像がありません。"
        return make_parent_selection_handler(service)(str(generation_id))

    return handler


def make_parent_selection_handler(
    service: UpscaleEnqueueService,
) -> Callable[[str | None], tuple[str, object, str]]:
    def handler(selected: str | None) -> tuple[str, object, str]:
        if not selected or not selected.strip():
            return "", None, "親Generationを選択してください。"
        try:
            selection = service.select_parent(UUID(selected.strip()))
            return (
                str(selection.generation_id),
                str(selection.preview_path),
                f"親画像を選択しました: `{selection.generation_id}`",
            )
        except (ValueError, UpscaleEnqueueError):
            return "", None, _PARENT_SELECTION_ERROR
        except Exception:  # noqa: BLE001 - UI must not expose internal exception details
            logger.exception("Upscale parent selection failed")
            return "", None, _INTERNAL_ERROR

    return handler


def make_upscale_result_handler(
    service: UpscaleEnqueueService,
) -> Callable[[str | None], tuple[object, object, str]]:
    def handler(selected: str | None) -> tuple[object, object, str]:
        if not selected or not selected.strip():
            return None, [], ""
        try:
            result = service.comparison_for_generation(UUID(selected.strip()))
            return (
                str(result.result_path),
                list(result.gallery),
                f"アップスケール結果を表示しました: `{result.result_generation_id}`",
            )
        except (ValueError, UpscaleEnqueueError):
            return None, [], _RESULT_ERROR
        except Exception:  # noqa: BLE001 - UI must not expose internal exception details
            logger.exception("Upscale result lookup failed")
            return None, [], _INTERNAL_ERROR

    return handler


def make_upscale_visibility_handler() -> Callable[[str], tuple[object, object]]:
    def handler(method: str) -> tuple[object, object]:
        is_image = method == UpscaleMethod.IMAGE.value
        return gr.update(visible=is_image), gr.update(visible=not is_image)

    return handler


def begin_upscale_enqueue() -> gr.Button:
    return gr.Button(value="アップスケール処理中…", interactive=False)


def make_upscale_enqueue_details_handler(
    service: UpscaleEnqueueService,
    preflight_service: GenerationPreflightService | None = None,
) -> Callable[..., Any]:
    def handler(
        parent_generation_id: str,
        method: str,
        sizing_mode: str,
        scale_factor: float | None,
        target_width: float | None,
        target_height: float | None,
        upscaler_name: str | None,
        denoise: float | None,
        source_import_id: str | None = None,
    ) -> tuple[Any, str]:
        try:
            settings = _settings_from_inputs(
                method,
                sizing_mode,
                scale_factor,
                target_width,
                target_height,
                upscaler_name,
                denoise,
            )
            import_id = (
                UUID(source_import_id.strip())
                if source_import_id and source_import_id.strip()
                else None
            )
            parent_id = (
                UUID(parent_generation_id.strip())
                if parent_generation_id and parent_generation_id.strip()
                else None
            )
            item = (
                service.enqueue_import(import_id, settings)
                if import_id is not None
                else service.enqueue(parent_id, settings)  # type: ignore[arg-type]
            )
            return gr.Button(interactive=True), (
                f"Generation ID: `{item.generation.id}`\n\n"
                f"Queue順序: `{item.entry.sequence}`\n\n"
                f"親Generation ID: `{parent_id}`\n\n"
                f"方式: `{settings.method.value}` / 出力予定サイズ: "
                f"`{item.generation.settings_snapshot.width} x "
                f"{item.generation.settings_snapshot.height}`"
            )
        except (ValueError, UpscaleEnqueueError):
            return gr.Button(interactive=True), _UPSCALE_INPUT_ERROR
        except Exception:  # noqa: BLE001 - restore the button without exposing internals
            logger.exception("Upscale enqueue details failed")
            return gr.Button(interactive=True), _INTERNAL_ERROR

    if preflight_service is None:
        return handler

    async def preflight_handler(
        parent_generation_id: str,
        method: str,
        sizing_mode: str,
        scale_factor: float | None,
        target_width: float | None,
        target_height: float | None,
        upscaler_name: str | None,
        denoise: float | None,
        source_import_id: str | None = None,
    ) -> tuple[Any, str]:
        try:
            settings = _settings_from_inputs(
                method,
                sizing_mode,
                scale_factor,
                target_width,
                target_height,
                upscaler_name,
                denoise,
            )
            import_id = (
                UUID(source_import_id.strip())
                if source_import_id and source_import_id.strip()
                else None
            )
            parent_id = (
                UUID(parent_generation_id.strip())
                if parent_generation_id and parent_generation_id.strip()
                else None
            )
            source_settings = None
            if settings.method is UpscaleMethod.LATENT:
                if import_id is not None:
                    source_settings = service.get_import_generation_settings(import_id)
                elif parent_id is not None:
                    source_settings = service.get_parent_generation_settings(parent_id)
            preflight = await preflight_service.check_upscale(
                settings.method.value,
                upscaler_name=settings.upscaler_name,
                source_settings=source_settings,
            )
            if not preflight.is_ready:
                return gr.Button(interactive=True), preflight_markdown(preflight)
            item = (
                service.enqueue_import(import_id, settings)
                if import_id is not None
                else service.enqueue(parent_id, settings)  # type: ignore[arg-type]
            )
            warning = f"\n\n{preflight_markdown(preflight)}" if preflight.warnings else ""
            return gr.Button(interactive=True), (
                f"Generation ID: `{item.generation.id}`\n\n"
                f"Queue position: `{item.entry.sequence}`\n\n"
                f"Parent Generation ID: `{parent_id}`\n\n"
                f"Method: `{settings.method.value}` / output size: "
                f"`{item.generation.settings_snapshot.width} x "
                f"{item.generation.settings_snapshot.height}`"
                f"{warning}"
            )
        except (ValueError, UpscaleEnqueueError):
            return gr.Button(interactive=True), _UPSCALE_INPUT_ERROR
        except Exception:  # noqa: BLE001 - restore the button without exposing internals
            logger.exception("Upscale enqueue preflight failed")
            return gr.Button(interactive=True), _INTERNAL_ERROR

    return preflight_handler


def make_upscale_enqueue_handler(
    service: UpscaleEnqueueService,
) -> Callable[..., tuple[Any, str]]:
    def handler(
        parent_generation_id: str,
        method: str,
        sizing_mode: str,
        scale_factor: float | None,
        target_width: float | None,
        target_height: float | None,
        upscaler_name: str | None,
        denoise: float | None,
        source_import_id: str | None = None,
    ) -> tuple[Any, str]:
        try:
            settings = _settings_from_inputs(
                method,
                sizing_mode,
                scale_factor,
                target_width,
                target_height,
                upscaler_name,
                denoise,
            )
            import_id = (
                UUID(source_import_id.strip())
                if source_import_id and source_import_id.strip()
                else None
            )
            parent_id = (
                UUID(parent_generation_id.strip())
                if parent_generation_id and parent_generation_id.strip()
                else None
            )
            item = (
                service.enqueue_import(import_id, settings)
                if import_id is not None
                else service.enqueue(parent_id, settings)  # type: ignore[arg-type]
            )
            return gr.Button(
                interactive=True
            ), f"キューへ追加しました（順序 {item.entry.sequence}）。"
        except (ValueError, UpscaleEnqueueError):
            return gr.Button(interactive=True), _UPSCALE_INPUT_ERROR
        except Exception:  # noqa: BLE001 - restore the button without exposing internals
            logger.exception("Upscale enqueue failed")
            return gr.Button(interactive=True), _INTERNAL_ERROR

    return handler


def make_upscale_plan_handler(
    service: UpscaleEnqueueService,
) -> Callable[..., str]:
    def handler(
        parent_generation_id: str,
        method: str,
        sizing_mode: str,
        scale_factor: float | None,
        target_width: float | None,
        target_height: float | None,
        upscaler_name: str | None,
        denoise: float | None,
        source_import_id: str | None = None,
    ) -> str:
        try:
            settings = _settings_from_inputs(
                method,
                sizing_mode,
                scale_factor,
                target_width,
                target_height,
                upscaler_name,
                denoise,
            )
            import_id = (
                UUID(source_import_id.strip())
                if source_import_id and source_import_id.strip()
                else None
            )
            parent_id = (
                UUID(parent_generation_id.strip())
                if parent_generation_id and parent_generation_id.strip()
                else None
            )
            plan = (
                service.plan_import(import_id, settings)
                if import_id is not None
                else service.plan(parent_id, settings)  # type: ignore[arg-type]
            )
            return (
                f"予定サイズ: **{plan.target_width} × {plan.target_height}**  "
                f"（親 {plan.source_width} × {plan.source_height}、"
                f"負荷: `{plan.load_level.value}`）"
            )
        except (ValueError, UpscaleEnqueueError):
            return _UPSCALE_INPUT_ERROR
        except Exception:  # noqa: BLE001 - UI must not expose internal exception details
            logger.exception("Upscale plan failed")
            return _INTERNAL_ERROR

    return handler


def _settings_from_inputs(
    method: str,
    sizing_mode: str,
    scale_factor: float | None,
    target_width: float | None,
    target_height: float | None,
    upscaler_name: str | None,
    denoise: float | None,
) -> UpscaleSettings:
    return UpscaleSettings(
        method=UpscaleMethod(method),
        sizing_mode=UpscaleSizingMode(sizing_mode),
        scale_factor=scale_factor if sizing_mode == UpscaleSizingMode.FACTOR.value else None,
        target_width=int(target_width) if target_width is not None else None,
        target_height=int(target_height) if target_height is not None else None,
        upscaler_name=upscaler_name or None,
        denoise=denoise if method == UpscaleMethod.LATENT.value else None,
    )


__all__ = [
    "UpscaleTabComponents",
    "build_upscale_tab",
    "begin_upscale_enqueue",
    "make_latest_parent_handler",
    "make_latest_parent_selection_handler",
    "make_parent_selection_handler",
    "make_upscale_enqueue_handler",
    "make_upscale_enqueue_details_handler",
    "make_upscale_plan_handler",
    "make_upscale_result_handler",
    "make_upscale_visibility_handler",
]
