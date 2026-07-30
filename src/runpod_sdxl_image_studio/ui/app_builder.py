"""Composition root for the Phase 1A Gradio application."""

from __future__ import annotations

import gradio as gr

from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.websocket_client import ComfyUIWebSocketClient
from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.adapters.database.engine import (
    create_image_studio_engine,
    create_session_factory,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_progress_repository import (
    GenerationProgressRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepository,
    GenerationCompletionRepository,
    GenerationFailureRepository,
    GenerationJobRepository,
    GenerationQueueRepository,
    GenerationRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_start_repository import (
    GenerationStartRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.lora_metadata_repository import (
    LoraMetadataRepository,
)
from runpod_sdxl_image_studio.adapters.storage.generation_metadata_storage import (
    GenerationMetadataStorage,
)
from runpod_sdxl_image_studio.adapters.storage.history_thumbnail_storage import (
    HistoryThumbnailStorage,
)
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.adapters.storage.lora_thumbnail_storage import LoraThumbnailStorage
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.domain.lora_search import append_trigger_words
from runpod_sdxl_image_studio.services.comfyui_service import ComfyUIService
from runpod_sdxl_image_studio.services.generation_history_service import (
    GenerationHistoryService,
)
from runpod_sdxl_image_studio.services.generation_recovery_service import (
    GenerationRecoveryService,
)
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.services.lora_catalog_service import (
    LoraCatalogError,
    LoraCatalogService,
)
from runpod_sdxl_image_studio.ui.components.lora_editor import (
    add_lora_row,
    component_outputs,
    lora_settings_from_state,
    move_lora_row,
    normalize_lora_state,
    remove_lora_row,
    render_state_updates,
    update_lora_row,
)
from runpod_sdxl_image_studio.ui.tabs.history_tab import (
    begin_regeneration,
    build_history_tab,
    enable_regeneration_button,
    make_history_detail_handler,
    make_history_refresh_handler,
    make_restore_handler,
    next_history_page,
    previous_history_page,
)
from runpod_sdxl_image_studio.ui.tabs.history_tab import (
    make_favorite_handler as make_history_favorite_handler,
)
from runpod_sdxl_image_studio.ui.tabs.history_tab import (
    make_note_handler as make_history_note_handler,
)
from runpod_sdxl_image_studio.ui.tabs.lora_management_tab import (
    build_lora_management_tab,
    make_favorite_handler,
    make_save_handler,
    make_search_handler,
    make_select_handler,
    make_sync_handler,
    make_thumbnail_delete_handler,
    make_thumbnail_save_handler,
)
from runpod_sdxl_image_studio.ui.tabs.system_tab import (
    build_generation_tab,
    build_system_tab,
    capability_refresh_outputs,
    disable_generate_button,
    make_check_connection_handler,
    make_generate_handler,
    make_refresh_handler,
    size_preset_values,
)
from runpod_sdxl_image_studio.ui.view_models import initial_status_markdown
from runpod_sdxl_image_studio.workflows.loader import load_txt2img_template

APP_TITLE = "RunPod SDXL Image Studio"
APP_CSS = """
.gradio-container { max-width: 960px !important; width: 100% !important; }
@media (max-width: 640px) {
  .gradio-container { padding: 0.75rem !important; }
  button { min-height: 2.75rem; }
}
"""


def build_app(
    settings: Settings | None = None,
    service: ComfyUIService | None = None,
) -> gr.Blocks:
    """Build the UI without starting a server or contacting ComfyUI."""

    app_settings = settings or get_settings()
    client = ComfyUIClient(app_settings)
    comfyui_service = service or ComfyUIService(client)
    database_engine = create_image_studio_engine(app_settings)
    session_factory = create_session_factory(database_engine)
    generation_repository = GenerationRepository(session_factory)
    artifact_repository = GenerationArtifactRepository(session_factory)
    completion_repository = GenerationCompletionRepository(session_factory)
    failure_repository = GenerationFailureRepository(session_factory)
    job_repository = GenerationJobRepository(session_factory)
    queue_repository = GenerationQueueRepository(session_factory)
    start_repository = GenerationStartRepository(session_factory)
    progress_repository = GenerationProgressRepository(session_factory)
    catalog_service = LoraCatalogService(
        LoraMetadataRepository(create_session_factory(database_engine)),
        LoraThumbnailStorage(
            app_settings.data_dir / "lora_thumbnails",
            app_settings.max_lora_thumbnail_bytes,
            app_settings.lora_thumbnail_max_edge,
        ),
    )
    loaded_workflow = load_txt2img_template(
        app_settings.workflow_dir.parent if app_settings.workflow_dir.exists() else None
    )
    generation_service = GenerationService(
        client,
        WorkflowAdapter(loaded_workflow.as_mapping()),
        ComfyUIWebSocketClient(app_settings),
        LocalStorageAdapter(app_settings),
        comfyui_service.refresh_capabilities,
        app_settings,
        lora_catalog_service=catalog_service,
        generation_repository=generation_repository,
        artifact_repository=artifact_repository,
        completion_repository=completion_repository,
        failure_repository=failure_repository,
        job_repository=job_repository,
        queue_repository=queue_repository,
        start_repository=start_repository,
        progress_repository=progress_repository,
        thumbnail_storage=HistoryThumbnailStorage(app_settings),
        metadata_storage=GenerationMetadataStorage(app_settings.data_dir),
    )
    history_service = GenerationHistoryService(
        generation_repository,
        artifact_repository,
        app_settings,
    )
    recovery_service = GenerationRecoveryService(
        client,
        generation_repository,
        job_repository,
        artifact_repository,
        app_settings,
        completed_prompt_handler=generation_service.recover_prompt,
        failure_repository=failure_repository,
    )
    with gr.Blocks(title=APP_TITLE, css=APP_CSS) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        with gr.Tab("生成"):
            generation = build_generation_tab(app_settings.max_loras)
        with gr.Tab("システム"):
            system = build_system_tab(
                app_settings.comfyui_base_url,
                initial_status_markdown(),
            )
        with gr.Tab("LoRA管理"):
            lora_management = build_lora_management_tab(catalog_service)
        with gr.Tab("履歴"):
            history = build_history_tab()

        capability_inputs = [
            generation.checkpoint,
            generation.vae,
            generation.sampler,
            generation.scheduler,
            generation.upscaler,
            generation.lora_editor.state,
            generation.lora_editor.choices,
            generation.lora_category_filter,
        ]
        capability_outputs = capability_refresh_outputs(generation)
        system.connection_button.click(
            fn=make_check_connection_handler(
                comfyui_service,
                app_settings.timezone,
                generation,
                catalog_service,
            ),
            inputs=capability_inputs,
            outputs=[system.status_markdown, system.capability_message, *capability_outputs],
        )
        system.refresh_button.click(
            fn=make_refresh_handler(comfyui_service, generation, catalog_service),
            inputs=capability_inputs,
            outputs=[system.capability_message, *capability_outputs],
        )
        generation.size_preset.change(
            fn=lambda preset: size_preset_values(preset),
            inputs=[generation.size_preset],
            outputs=[generation.width, generation.height],
        )

        def handle_lora_name_change(
            state: object,
            name: str | None,
            model_strength: object,
            clip_strength: object,
            row_index: int,
        ) -> tuple[object, ...]:
            rows = normalize_lora_state(state, app_settings.max_loras)
            previous = rows[row_index].get("lora_name") if row_index < len(rows) else None
            model = model_strength
            clip = clip_strength
            if name != previous:
                metadata = catalog_service.get_by_file_name(name) if name else None
                model = (
                    metadata.recommended_model_strength
                    if metadata and metadata.recommended_model_strength is not None
                    else 1.0
                )
                clip = (
                    metadata.recommended_clip_strength
                    if metadata and metadata.recommended_clip_strength is not None
                    else 1.0
                )
            updated = update_lora_row(state, row_index, name, model, clip, app_settings.max_loras)
            return updated, model, clip

        generation.lora_editor.add_button.click(
            fn=lambda state, choices: render_state_updates(
                add_lora_row(state, app_settings.max_loras),
                choices,
                app_settings.max_loras,
            ),
            inputs=[generation.lora_editor.state, generation.lora_editor.choices],
            outputs=[
                generation.lora_editor.state,
                *component_outputs(generation.lora_editor),
                generation.lora_editor.add_button,
            ],
        )
        for index, row in enumerate(generation.lora_editor.rows):
            row.name.change(
                fn=lambda state, name, model, clip, row_index=index: handle_lora_name_change(
                    state, name, model, clip, row_index
                ),
                inputs=[
                    generation.lora_editor.state,
                    row.name,
                    row.model_strength,
                    row.clip_strength,
                ],
                outputs=[generation.lora_editor.state, row.model_strength, row.clip_strength],
            )
            row.model_strength.change(
                fn=lambda state, name, model, clip, row_index=index: update_lora_row(
                    state, row_index, name, model, clip, app_settings.max_loras
                ),
                inputs=[
                    generation.lora_editor.state,
                    row.name,
                    row.model_strength,
                    row.clip_strength,
                ],
                outputs=[generation.lora_editor.state],
            )
            row.clip_strength.change(
                fn=lambda state, name, model, clip, row_index=index: update_lora_row(
                    state, row_index, name, model, clip, app_settings.max_loras
                ),
                inputs=[
                    generation.lora_editor.state,
                    row.name,
                    row.model_strength,
                    row.clip_strength,
                ],
                outputs=[generation.lora_editor.state],
            )
            row.remove_button.click(
                fn=lambda state, choices, row_index=index: render_state_updates(
                    remove_lora_row(state, row_index, app_settings.max_loras),
                    choices,
                    app_settings.max_loras,
                ),
                inputs=[generation.lora_editor.state, generation.lora_editor.choices],
                outputs=[
                    generation.lora_editor.state,
                    *component_outputs(generation.lora_editor),
                    generation.lora_editor.add_button,
                ],
            )
            row.up_button.click(
                fn=lambda state, choices, row_index=index: render_state_updates(
                    move_lora_row(state, row_index, -1, app_settings.max_loras),
                    choices,
                    app_settings.max_loras,
                ),
                inputs=[generation.lora_editor.state, generation.lora_editor.choices],
                outputs=[
                    generation.lora_editor.state,
                    *component_outputs(generation.lora_editor),
                    generation.lora_editor.add_button,
                ],
            )
            row.down_button.click(
                fn=lambda state, choices, row_index=index: render_state_updates(
                    move_lora_row(state, row_index, 1, app_settings.max_loras),
                    choices,
                    app_settings.max_loras,
                ),
                inputs=[generation.lora_editor.state, generation.lora_editor.choices],
                outputs=[
                    generation.lora_editor.state,
                    *component_outputs(generation.lora_editor),
                    generation.lora_editor.add_button,
                ],
            )
        generation.lora_editor.trigger_button.click(
            fn=lambda prompt, state: _append_selected_triggers(
                prompt, state, catalog_service, app_settings.max_loras
            ),
            inputs=[generation.positive_prompt, generation.lora_editor.state],
            outputs=[generation.positive_prompt, generation.lora_editor.trigger_message],
        )

        generation.lora_category_filter.change(
            fn=lambda category, state: _filter_lora_category(
                category, state, catalog_service, app_settings.max_loras
            ),
            inputs=[generation.lora_category_filter, generation.lora_editor.state],
            outputs=[
                generation.lora_editor.choices,
                generation.lora_editor.state,
                *component_outputs(generation.lora_editor),
                generation.lora_editor.add_button,
            ],
        )

        search_inputs = [
            lora_management.search,
            lora_management.category_filter,
            lora_management.favorites_only,
            lora_management.include_missing,
            lora_management.sort,
            lora_management.selected,
        ]
        search_outputs = [
            lora_management.result_list,
            lora_management.selected,
            lora_management.category_filter,
        ]
        search_handler = make_search_handler(catalog_service)
        for source in (
            lora_management.search,
            lora_management.category_filter,
            lora_management.favorites_only,
            lora_management.include_missing,
            lora_management.sort,
        ):
            source.change(fn=search_handler, inputs=search_inputs, outputs=search_outputs)
        lora_management.search.submit(
            fn=search_handler, inputs=search_inputs, outputs=search_outputs
        )
        lora_management.sync_button.click(
            fn=make_sync_handler(comfyui_service, catalog_service),
            outputs=[
                lora_management.message,
                lora_management.result_list,
                lora_management.selected,
                lora_management.category_filter,
            ],
        )
        lora_management.selected.change(
            fn=make_select_handler(catalog_service),
            inputs=[lora_management.selected],
            outputs=[
                lora_management.display_name,
                lora_management.category,
                lora_management.favorite,
                lora_management.trigger_words,
                lora_management.recommended_model,
                lora_management.recommended_clip,
                lora_management.compatible_models,
                lora_management.notes,
                lora_management.thumbnail_preview,
            ],
        )
        lora_management.save_button.click(
            fn=make_save_handler(
                catalog_service,
                app_settings.max_loras,
                lora_editor=generation.lora_editor,
            ),
            inputs=[
                lora_management.selected,
                lora_management.display_name,
                lora_management.category,
                lora_management.favorite,
                lora_management.trigger_words,
                lora_management.recommended_model,
                lora_management.recommended_clip,
                lora_management.compatible_models,
                lora_management.notes,
                lora_management.search,
                lora_management.category_filter,
                lora_management.favorites_only,
                lora_management.include_missing,
                lora_management.sort,
                generation.lora_category_filter,
                generation.lora_editor.state,
            ],
            outputs=[
                lora_management.message,
                lora_management.result_list,
                lora_management.selected,
                lora_management.category_filter,
                generation.lora_editor.choices,
                generation.lora_editor.state,
                *component_outputs(generation.lora_editor),
                generation.lora_editor.add_button,
            ],
        )
        lora_management.favorite.change(
            fn=make_favorite_handler(
                catalog_service,
                app_settings.max_loras,
                lora_editor=generation.lora_editor,
            ),
            inputs=[
                lora_management.selected,
                lora_management.favorite,
                lora_management.search,
                lora_management.category_filter,
                lora_management.favorites_only,
                lora_management.include_missing,
                lora_management.sort,
                generation.lora_category_filter,
                generation.lora_editor.state,
            ],
            outputs=[
                lora_management.favorite,
                lora_management.message,
                lora_management.result_list,
                lora_management.selected,
                lora_management.category_filter,
                generation.lora_editor.choices,
                generation.lora_editor.state,
                *component_outputs(generation.lora_editor),
                generation.lora_editor.add_button,
            ],
        )
        lora_management.thumbnail_upload.change(
            fn=make_thumbnail_save_handler(catalog_service),
            inputs=[lora_management.selected, lora_management.thumbnail_upload],
            outputs=[lora_management.message, lora_management.thumbnail_preview],
        )
        lora_management.delete_thumbnail_button.click(
            fn=make_thumbnail_delete_handler(catalog_service),
            inputs=[lora_management.selected],
            outputs=[lora_management.message, lora_management.thumbnail_preview],
        )
        history.refresh_button.click(
            fn=make_history_refresh_handler(history_service, recovery_service),
            inputs=[
                history.page_state,
                history.date_filter,
                history.status_filter,
                history.kind_filter,
                history.favorite_filter,
            ],
            outputs=[
                history.page_state,
                history.thumbnail_gallery,
                history.cards,
                history.selected,
                history.page,
                history.previous_button,
                history.next_button,
                history.message,
            ],
        )
        history.previous_button.click(
            fn=previous_history_page,
            inputs=[history.page_state],
            outputs=[history.page_state],
        ).then(
            fn=make_history_refresh_handler(history_service, recovery_service),
            inputs=[
                history.page_state,
                history.date_filter,
                history.status_filter,
                history.kind_filter,
                history.favorite_filter,
            ],
            outputs=[
                history.page_state,
                history.thumbnail_gallery,
                history.cards,
                history.selected,
                history.page,
                history.previous_button,
                history.next_button,
                history.message,
            ],
        )
        history.next_button.click(
            fn=next_history_page,
            inputs=[history.page_state],
            outputs=[history.page_state],
        ).then(
            fn=make_history_refresh_handler(history_service, recovery_service),
            inputs=[
                history.page_state,
                history.date_filter,
                history.status_filter,
                history.kind_filter,
                history.favorite_filter,
            ],
            outputs=[
                history.page_state,
                history.thumbnail_gallery,
                history.cards,
                history.selected,
                history.page,
                history.previous_button,
                history.next_button,
                history.message,
            ],
        )
        history.selected.change(
            fn=make_history_detail_handler(history_service),
            inputs=[history.selected],
            outputs=[
                history.detail,
                history.image,
                history.favorite,
                history.note,
                history.message,
            ],
        )
        history.favorite.change(
            fn=make_history_favorite_handler(history_service),
            inputs=[history.selected, history.favorite],
            outputs=[history.favorite, history.message],
        )
        history.save_note_button.click(
            fn=make_history_note_handler(history_service),
            inputs=[history.selected, history.note],
            outputs=[history.message],
        )
        restore_outputs = [
            history.message,
            generation.positive_prompt,
            generation.negative_prompt,
            generation.checkpoint,
            generation.vae,
            generation.width,
            generation.height,
            generation.seed_mode,
            generation.seed,
            generation.steps,
            generation.cfg_scale,
            generation.sampler,
            generation.scheduler,
            generation.lora_editor.state,
            *component_outputs(generation.lora_editor),
            generation.lora_editor.add_button,
            generation.restored_from_generation,
            generation.regeneration_valid,
        ]
        history.restore_button.click(
            fn=make_restore_handler(history_service, app_settings.max_loras),
            inputs=[
                history.selected,
                generation.checkpoint_choices,
                generation.vae_choices,
                generation.lora_editor.choices,
            ],
            outputs=restore_outputs,
        )
        generation_inputs = [
            generation.checkpoint,
            generation.positive_prompt,
            generation.negative_prompt,
            generation.size_preset,
            generation.width,
            generation.height,
            generation.seed_mode,
            generation.seed,
            generation.steps,
            generation.cfg_scale,
            generation.sampler,
            generation.scheduler,
            generation.vae,
            generation.lora_editor.state,
            generation.restored_from_generation,
            generation.regeneration_valid,
            generation.regeneration_requested,
        ]
        regenerate_event = history.regenerate_button.click(
            fn=begin_regeneration,
            outputs=[history.regenerate_button, generation.regeneration_requested],
            queue=False,
        )
        regenerate_event = regenerate_event.then(
            fn=make_restore_handler(history_service, app_settings.max_loras),
            inputs=[
                history.selected,
                generation.checkpoint_choices,
                generation.vae_choices,
                generation.lora_editor.choices,
            ],
            outputs=restore_outputs,
        )
        regeneration_generation_event = regenerate_event.then(
            fn=make_generate_handler(generation_service, app_settings.max_loras),
            inputs=generation_inputs,
            outputs=[
                generation.generate_button,
                generation.progress,
                generation.result_image,
                generation.result_details,
                generation.regeneration_requested,
            ],
            concurrency_limit=1,
        )
        regeneration_generation_event.then(
            fn=enable_regeneration_button,
            outputs=[history.regenerate_button],
        )
        generate_event = generation.generate_button.click(
            fn=disable_generate_button,
            outputs=[generation.generate_button],
            queue=False,
        )
        generate_event.then(
            fn=make_generate_handler(generation_service, app_settings.max_loras),
            inputs=generation_inputs,
            outputs=[
                generation.generate_button,
                generation.progress,
                generation.result_image,
                generation.result_details,
                generation.regeneration_requested,
            ],
            concurrency_limit=1,
        )
    return demo


def _filter_lora_category(
    category: str | None,
    state: object,
    catalog: LoraCatalogService,
    max_loras: int,
) -> tuple[object, ...]:
    """Apply a catalog category without changing the stored LoRA strengths."""

    choices = catalog.selector_options(category or None)
    return (list(choices),) + render_state_updates(
        state,
        choices,
        max_loras,
        clear_unavailable=True,
    )


def _append_selected_triggers(
    prompt: str,
    state: object,
    catalog: LoraCatalogService,
    max_loras: int,
) -> tuple[str, str]:
    try:
        settings = lora_settings_from_state(state, max_loras)
        words: list[str] = []
        for metadata in catalog.metadata_for_files(lora.name for lora in settings):
            if metadata is not None:
                words.extend(metadata.trigger_words)
        updated = append_trigger_words(prompt or "", tuple(words))
        return (
            updated,
            "トリガーワードをPrompt末尾へ追加しました。"
            if words
            else "追加するトリガーワードはありません。",
        )
    except (ValueError, LoraCatalogError):
        return prompt, "トリガーワードを追加できませんでした。"
