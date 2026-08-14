"""Gradio UI for preview-first external image metadata import."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

import gradio as gr

from runpod_sdxl_image_studio.domain.metadata_import import (
    MetadataImportError,
    MetadataImportPreview,
    MetadataModelMapping,
    MetadataSourceKind,
)
from runpod_sdxl_image_studio.services.metadata_import_service import MetadataImportService
from runpod_sdxl_image_studio.ui.components.lora_editor import render_state_updates
from runpod_sdxl_image_studio.ui.tabs.history_tab import component_output_count

logger = logging.getLogger(__name__)

_SAFE_IMPORT_ERROR = "画像またはmetadataを確認してください。"
_SAFE_INTERNAL_ERROR = "metadataの解析中に内部エラーが発生しました。"


class MetadataImportTabComponents:
    image: Any
    sidecar: Any
    parse_button: Any
    import_id: Any
    preview_image: Any
    image_hash: Any
    image_dimensions: Any
    metadata_source: Any
    source_selection: Any
    confirm_sidecar_hash: Any
    select_source_button: Any
    status: Any
    warnings: Any
    unresolved: Any
    raw_metadata: Any
    settings_preview: Any
    mapping_json: Any
    apply_mapping: Any
    apply_generation: Any
    apply_upscale: Any

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


def build_metadata_import_tab(max_loras: int = 8) -> MetadataImportTabComponents:
    gr.Markdown("## 外部画像metadataインポート")
    gr.Markdown(
        "画像と任意のsidecar JSONを解析し、内容を確認してから生成フォームまたは"
        "アップスケールへ適用します。アップロードだけでは実行しません。"
    )
    with gr.Row(elem_classes=["metadata-actions"]):
        image = gr.File(label="画像（PNG / WebP）", file_types=[".png", ".webp"], type="filepath")
        sidecar = gr.File(label="sidecar JSON（任意）", file_types=[".json"], type="filepath")
    parse_button = gr.Button("metadataを解析", variant="primary")
    import_id = gr.State(None)
    preview_image = gr.Image(label="canonical画像", type="filepath", interactive=False)
    image_hash = gr.Textbox(label="stored image SHA-256", interactive=False)
    image_dimensions = gr.Textbox(label="画像寸法", interactive=False)
    metadata_source = gr.Markdown()
    source_selection = gr.Radio(
        label="metadata source",
        choices=[
            ("ComfyUI PNG prompt", MetadataSourceKind.COMFYUI_PROMPT.value),
            ("app sidecar", MetadataSourceKind.APP_SIDECAR.value),
        ],
        value=None,
        interactive=False,
    )
    confirm_sidecar_hash = gr.Checkbox(
        label="sidecarの画像hash不一致を確認して使用する",
        value=False,
        interactive=False,
    )
    select_source_button = gr.Button("選択したmetadata sourceを確定", interactive=False)
    status = gr.Markdown()
    warnings = gr.Markdown()
    unresolved = gr.Markdown()
    with gr.Accordion("raw metadata", open=False):
        raw_metadata = gr.Code(language="json", interactive=False)
    settings_preview = gr.Code(label="生成条件preview", language="json", interactive=False)
    mapping_json = gr.Code(
        label="明示的なmodel mapping(JSON)",
        language="json",
        value="[]",
        lines=5,
    )
    apply_mapping = gr.Button("model mappingを適用")
    with gr.Row(elem_classes=["metadata-actions"]):
        apply_generation = gr.Button(
            "生成フォームへ適用", interactive=False, elem_classes=["mobile-tap-button"]
        )
        apply_upscale = gr.Button(
            "アップスケールへ適用", interactive=False, elem_classes=["mobile-tap-button"]
        )
    return MetadataImportTabComponents(
        image=image,
        sidecar=sidecar,
        parse_button=parse_button,
        import_id=import_id,
        preview_image=preview_image,
        image_hash=image_hash,
        image_dimensions=image_dimensions,
        metadata_source=metadata_source,
        source_selection=source_selection,
        confirm_sidecar_hash=confirm_sidecar_hash,
        select_source_button=select_source_button,
        status=status,
        warnings=warnings,
        unresolved=unresolved,
        raw_metadata=raw_metadata,
        settings_preview=settings_preview,
        mapping_json=mapping_json,
        apply_mapping=apply_mapping,
        apply_generation=apply_generation,
        apply_upscale=apply_upscale,
    )


def make_metadata_import_handler(
    service: MetadataImportService,
) -> Callable[[str | None, str | None], tuple[object, ...]]:
    def handler(image_path: str | None, sidecar_path: str | None) -> tuple[object, ...]:
        try:
            if not image_path:
                raise MetadataImportError("metadata_import_invalid_image", "image is required")
            image_file = Path(image_path)
            image_bytes = image_file.read_bytes()
            sidecar_bytes = Path(sidecar_path).read_bytes() if sidecar_path else None
            preview = service.import_image(
                image_bytes,
                image_file.name,
                sidecar_bytes=sidecar_bytes,
            )
            return _preview_outputs(service, preview)
        except (MetadataImportError, OSError, ValueError):
            return _import_error_outputs()
        except Exception:  # noqa: BLE001 - UI boundary hides internal details
            logger.exception("Metadata import handler failed")
            return _import_internal_error_outputs()

    return handler


def make_metadata_source_selection_handler(
    service: MetadataImportService,
) -> Callable[[str | None, str | None, bool], tuple[object, ...]]:
    def handler(
        import_id: str | None,
        source_kind: str | None,
        confirm_sidecar_hash: bool,
    ) -> tuple[object, ...]:
        try:
            if not import_id or not source_kind:
                raise MetadataImportError(
                    "metadata_import_source_invalid", "metadata source is not selected"
                )
            preview = service.select_metadata_source(
                UUID(import_id),
                source_kind,
                confirm_sidecar_hash_mismatch=bool(confirm_sidecar_hash),
            )
            return _preview_outputs(service, preview)
        except (MetadataImportError, ValueError) as exc:
            if import_id:
                try:
                    preview = service.get_preview(UUID(import_id))
                    warning = getattr(exc, "code", "metadata_import_source_invalid")
                    preview = preview.model_copy(
                        update={"warnings": tuple(dict.fromkeys((*preview.warnings, warning)))}
                    )
                    return _preview_outputs(service, preview)
                except Exception:  # noqa: BLE001 - retain safe error fallback
                    pass
            return _import_error_outputs()
        except Exception:  # noqa: BLE001 - UI boundary hides internal details
            logger.exception("Metadata source selection failed")
            if import_id:
                try:
                    return _preview_outputs(service, service.get_preview(UUID(import_id)))
                except Exception:  # noqa: BLE001 - retain safe error fallback
                    pass
            return _import_internal_error_outputs()

    return handler


def make_metadata_generation_apply_handler(
    service: MetadataImportService,
    max_loras: int,
) -> Callable[..., tuple[object, ...]]:
    def handler(
        import_id: str | None,
        checkpoint_choices: object = None,
        vae_choices: object = None,
        lora_choices: object = None,
    ) -> tuple[object, ...]:
        try:
            if not import_id:
                raise MetadataImportError("metadata_import_unresolved", "import is not selected")
            settings = service.build_generation_settings(UUID(import_id))
            lora_state = [
                {
                    "row_id": f"metadata-{index}",
                    "lora_name": lora.name,
                    "model_strength": lora.model_strength,
                    "clip_strength": lora.clip_strength,
                }
                for index, lora in enumerate(settings.loras)
            ] or [
                {
                    "row_id": "metadata-0",
                    "lora_name": None,
                    "model_strength": 1.0,
                    "clip_strength": 1.0,
                }
            ]
            return (
                "metadataを生成フォームへ適用しました。生成ボタンを押すまで実行しません。",
                settings.positive_prompt,
                settings.negative_prompt,
                gr.Dropdown(value=settings.checkpoint_name),
                gr.Dropdown(value=settings.vae_name),
                settings.width,
                settings.height,
                "Fixed",
                settings.seed,
                settings.steps,
                settings.cfg_scale,
                gr.Dropdown(value=settings.sampler_name),
                gr.Dropdown(value=settings.scheduler_name),
                gr.Dropdown(value=settings.final_upscale_model),
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
                lora_state,
                *render_state_updates(lora_state, lora_choices, max_loras),
                None,
                True,
            )
        except (MetadataImportError, ValueError):
            return _restore_error_outputs(max_loras)
        except Exception:  # noqa: BLE001 - UI boundary hides internal details
            logger.exception("Metadata generation apply failed")
            return _restore_error_outputs(max_loras)

    return handler


def make_metadata_mapping_handler(
    service: MetadataImportService,
) -> Callable[[str | None, str | None], tuple[object, ...]]:
    def handler(import_id: str | None, mapping_json: str | None) -> tuple[object, ...]:
        try:
            if not import_id:
                raise MetadataImportError(
                    "metadata_import_mapping_invalid", "import is not selected"
                )
            payload = json.loads(mapping_json or "[]")
            if not isinstance(payload, list):
                raise MetadataImportError(
                    "metadata_import_mapping_invalid", "mapping must be a list"
                )
            mappings = tuple(MetadataModelMapping.model_validate(item) for item in payload)
            preview = service.apply_model_mapping(UUID(import_id), mappings)
            return _preview_outputs(service, preview)
        except (MetadataImportError, ValueError, TypeError, json.JSONDecodeError):
            if import_id:
                try:
                    preview = service.get_preview(UUID(import_id))
                    warning = "metadata_import_mapping_invalid"
                    preview = preview.model_copy(
                        update={"warnings": tuple(dict.fromkeys((*preview.warnings, warning)))}
                    )
                    return _preview_outputs(service, preview)
                except Exception:  # noqa: BLE001 - retain safe error fallback
                    pass
            return _import_error_outputs()
        except Exception:  # noqa: BLE001 - UI boundary hides internal details
            logger.exception("Metadata mapping handler failed")
            if import_id:
                try:
                    return _preview_outputs(service, service.get_preview(UUID(import_id)))
                except Exception:  # noqa: BLE001 - retain safe error fallback
                    pass
            return _import_internal_error_outputs()

    return handler


def make_metadata_upscale_apply_handler(
    service: MetadataImportService,
) -> Callable[[str | None], tuple[object, ...]]:
    def handler(import_id: str | None) -> tuple[object, ...]:
        try:
            if not import_id:
                raise MetadataImportError("metadata_import_unresolved", "import is not selected")
            path = _gradio_image_path(service, UUID(import_id))
            return str(import_id), "", path, "外部画像をアップスケール対象に指定しました。"
        except (MetadataImportError, ValueError):
            return "", "", None, _SAFE_IMPORT_ERROR
        except Exception:  # noqa: BLE001 - UI boundary hides internal details
            logger.exception("Metadata upscale apply failed")
            return "", "", None, _SAFE_INTERNAL_ERROR

    return handler


def _preview_outputs(
    service: MetadataImportService, preview: MetadataImportPreview
) -> tuple[object, ...]:
    image_path = _gradio_image_path(service, preview.id)
    raw_text = "\n\n".join(
        source.raw_text for source in preview.raw_sources if source.raw_text is not None
    )
    settings = preview.candidate.model_dump(mode="json") if preview.candidate else {}
    source_choices = [candidate.source_kind.value for candidate in preview.candidates]
    selected_source = (
        preview.selected_metadata_source.value
        if preview.selected_metadata_source is not None
        else None
    )
    sidecar_hash_confirmation = "metadata_import_sidecar_hash_mismatch" in preview.warnings
    return (
        str(preview.id),
        str(image_path),
        preview.imported_image.stored_image_sha256,
        f"{preview.imported_image.image_width} × {preview.imported_image.image_height}",
        f"metadata source: `{preview.metadata_source.value}`",
        gr.Radio(
            choices=source_choices,
            value=selected_source,
            interactive=bool(source_choices),
        ),
        gr.Checkbox(
            value=preview.sidecar_hash_confirmed,
            interactive=sidecar_hash_confirmation,
        ),
        gr.Button(interactive=bool(source_choices)),
        f"status: `{preview.status.value}`",
        "\n".join(f"- `{warning}`" for warning in preview.warnings) or "警告はありません。",
        "\n".join(f"- `{field}`" for field in preview.unresolved_fields)
        or "未解決項目はありません。",
        raw_text,
        json.dumps(settings, ensure_ascii=False, indent=2),
        gr.Button(interactive=preview.status.value == "ready"),
        gr.Button(interactive=True),
        gr.Button(interactive=True),
    )


def _import_error_outputs() -> tuple[object, ...]:
    return (
        None,
        None,
        "",
        "",
        "",
        gr.Radio(choices=[], value=None, interactive=False),
        gr.Checkbox(value=False, interactive=False),
        gr.Button(interactive=False),
        _SAFE_IMPORT_ERROR,
        "",
        "",
        "",
        "",
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.Button(interactive=True),
    )


def _import_internal_error_outputs() -> tuple[object, ...]:
    return (
        None,
        None,
        "",
        "",
        "",
        gr.Radio(choices=[], value=None, interactive=False),
        gr.Checkbox(value=False, interactive=False),
        gr.Button(interactive=False),
        _SAFE_INTERNAL_ERROR,
        "",
        "",
        "",
        "",
        gr.Button(interactive=False),
        gr.Button(interactive=False),
        gr.Button(interactive=True),
    )


def _restore_error_outputs(max_loras: int) -> tuple[object, ...]:
    return (
        "metadataから生成条件を復元できませんでした。",
        *([gr.skip()] * (23 + component_output_count(max_loras))),
        None,
        False,
    )


def _gradio_image_path(service: MetadataImportService, import_id: UUID) -> str:
    """Resolve an absolute path only at the isolated Gradio image boundary."""

    return str(service.get_upscale_source_path(import_id))


__all__ = [
    "MetadataImportTabComponents",
    "build_metadata_import_tab",
    "make_metadata_generation_apply_handler",
    "make_metadata_import_handler",
    "make_metadata_mapping_handler",
    "make_metadata_source_selection_handler",
    "make_metadata_upscale_apply_handler",
]
