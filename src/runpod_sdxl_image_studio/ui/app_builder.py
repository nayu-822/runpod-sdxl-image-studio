"""Composition root for the Phase 1A Gradio application."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import gradio as gr

from runpod_sdxl_image_studio.adapters.catalog.upscaler_catalog import UpscalerCatalog
from runpod_sdxl_image_studio.adapters.comfyui.cancellation import ComfyUICancellationAdapter
from runpod_sdxl_image_studio.adapters.comfyui.client import ComfyUIClient
from runpod_sdxl_image_studio.adapters.comfyui.models import ComfyUICapabilities
from runpod_sdxl_image_studio.adapters.comfyui.upscale_workflow_adapter import (
    UpscaleWorkflowAdapter,
)
from runpod_sdxl_image_studio.adapters.comfyui.websocket_client import ComfyUIWebSocketClient
from runpod_sdxl_image_studio.adapters.comfyui.workflow_adapter import WorkflowAdapter
from runpod_sdxl_image_studio.adapters.database.engine import (
    create_image_studio_engine,
    create_session_factory,
)
from runpod_sdxl_image_studio.adapters.database.repositories.drive_sync_repository import (
    DriveSyncRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_custom_size_repository import (  # noqa: E501
    GenerationCustomSizeRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_dispatch_queue_repository import (  # noqa: E501
    GenerationDispatchQueueRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_form_state_repository import (  # noqa: E501
    GenerationFormStateRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_progress_repository import (
    GenerationProgressRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_repository import (
    GenerationArtifactRepository,
    GenerationCancellationRepository,
    GenerationCompletionRepository,
    GenerationFailureRepository,
    GenerationJobRepository,
    GenerationQueueRepository,
    GenerationRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.generation_start_repository import (
    GenerationStartRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.interactive_run_repository import (
    InteractiveRunRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.lora_metadata_repository import (
    LoraMetadataRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.metadata_import_repository import (
    MetadataImportRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.model_transfer_repository import (
    ModelTransferRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.pod_lifecycle_repository import (
    PodLifecycleRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.preset_repository import (
    PresetRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.system_error_repository import (
    SystemErrorEventRepository,
)
from runpod_sdxl_image_studio.adapters.database.repositories.upscale_settings_repository import (
    UpscaleSettingsRepository,
)
from runpod_sdxl_image_studio.adapters.drive.google_drive_adapter import GoogleDriveAdapter
from runpod_sdxl_image_studio.adapters.rclone.remote_model_catalog import (
    RemoteModelCatalogAdapter,
)
from runpod_sdxl_image_studio.adapters.rclone.state_backup_storage import StateBackupStorage
from runpod_sdxl_image_studio.adapters.runpod.pod_lifecycle import RunPodLifecycleAdapter
from runpod_sdxl_image_studio.adapters.storage.disk_usage import LocalDiskUsageAdapter
from runpod_sdxl_image_studio.adapters.storage.generation_metadata_storage import (
    GenerationMetadataStorage,
)
from runpod_sdxl_image_studio.adapters.storage.history_thumbnail_storage import (
    HistoryThumbnailStorage,
)
from runpod_sdxl_image_studio.adapters.storage.imported_image_storage import ImportedImageStorage
from runpod_sdxl_image_studio.adapters.storage.local_storage import LocalStorageAdapter
from runpod_sdxl_image_studio.adapters.storage.lora_thumbnail_storage import LoraThumbnailStorage
from runpod_sdxl_image_studio.config import Settings, get_settings
from runpod_sdxl_image_studio.domain.generation_queue import (
    GenerationQueueItem,
    ReconciliationOutcome,
)
from runpod_sdxl_image_studio.domain.lora_search import append_trigger_words
from runpod_sdxl_image_studio.jobs.auto_terminate_worker import (
    AutoTerminateCoordinator,
    AutoTerminateRuntime,
)
from runpod_sdxl_image_studio.jobs.drive_sync_worker import DriveSyncRuntime, DriveSyncWorker
from runpod_sdxl_image_studio.jobs.generation_queue_worker import (
    GenerationQueueRuntime,
    GenerationQueueWorker,
)
from runpod_sdxl_image_studio.jobs.model_transfer_worker import (
    ModelTransferRuntime,
    ModelTransferWorker,
)
from runpod_sdxl_image_studio.jobs.startup_model_restore import StartupModelRestoreRuntime
from runpod_sdxl_image_studio.services.comfyui_service import ComfyUIService
from runpod_sdxl_image_studio.services.drive_sync_service import DriveSyncService
from runpod_sdxl_image_studio.services.generation_custom_size_service import (
    GenerationCustomSizeError,
    GenerationCustomSizeService,
)
from runpod_sdxl_image_studio.services.generation_diff_service import GenerationDiffService
from runpod_sdxl_image_studio.services.generation_execution_service import (
    GenerationExecutionService,
)
from runpod_sdxl_image_studio.services.generation_form_state_service import (
    GenerationFormStateService,
)
from runpod_sdxl_image_studio.services.generation_history_service import (
    GenerationHistoryError,
    GenerationHistoryService,
)
from runpod_sdxl_image_studio.services.generation_persistence import (
    GenerationPersistenceRepositories,
)
from runpod_sdxl_image_studio.services.generation_preflight_service import (
    GenerationPreflightService,
)
from runpod_sdxl_image_studio.services.generation_queue_service import GenerationQueueService
from runpod_sdxl_image_studio.services.generation_recovery_service import (
    GenerationRecoveryService,
)
from runpod_sdxl_image_studio.services.generation_service import GenerationService
from runpod_sdxl_image_studio.services.interactive_generation_service import (
    InteractiveGenerationService,
)
from runpod_sdxl_image_studio.services.lora_catalog_service import (
    LoraCatalogError,
    LoraCatalogService,
)
from runpod_sdxl_image_studio.services.metadata_import_service import MetadataImportService
from runpod_sdxl_image_studio.services.model_preparation_service import ModelPreparationService
from runpod_sdxl_image_studio.services.pod_lifecycle_service import PodLifecycleService
from runpod_sdxl_image_studio.services.preset_service import PresetService
from runpod_sdxl_image_studio.services.recent_settings_service import RecentSettingsService
from runpod_sdxl_image_studio.services.state_sync_service import StateSyncService
from runpod_sdxl_image_studio.services.stateless_reconciliation_service import (
    StatelessReconciliationService,
)
from runpod_sdxl_image_studio.services.system_health_service import SystemHealthService
from runpod_sdxl_image_studio.services.upscale_enqueue_service import UpscaleEnqueueService
from runpod_sdxl_image_studio.services.upscale_service import UpscaleService
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
from runpod_sdxl_image_studio.ui.components.mobile_actions import (
    make_mobile_status_poll_handler,
    make_mobile_status_refresh_handler,
)
from runpod_sdxl_image_studio.ui.mobile_styles import mobile_ui_css
from runpod_sdxl_image_studio.ui.tabs.drive_sync_tab import (
    build_drive_sync_tab,
    make_drive_connection_handler,
    make_drive_discovery_handler,
    make_drive_failed_manifest_handler,
    make_drive_manifest_handler,
    make_drive_resync_handler,
    make_drive_retry_failed_handler,
    make_drive_retry_selected_handler,
    make_drive_sync_refresh_handler,
)
from runpod_sdxl_image_studio.ui.tabs.history_tab import (
    begin_regeneration,
    build_history_tab,
    clear_history_filters,
    enable_regeneration_button,
    make_generation_diff_handler,
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
from runpod_sdxl_image_studio.ui.tabs.metadata_import_tab import (
    build_metadata_import_tab,
    make_metadata_generation_apply_handler,
    make_metadata_import_handler,
    make_metadata_mapping_handler,
    make_metadata_source_selection_handler,
    make_metadata_upscale_apply_handler,
)
from runpod_sdxl_image_studio.ui.tabs.model_preparation_tab import (
    build_model_preparation_tab,
    make_model_cancel_handler,
    make_model_catalog_refresh_handler,
    make_model_jobs_refresh_handler,
    make_model_prepare_handler,
    make_model_retry_handler,
)
from runpod_sdxl_image_studio.ui.tabs.preset_tab import (
    build_preset_tab,
    make_preset_apply_handler,
    make_preset_clear_handler,
    make_preset_delete_handler,
    make_preset_duplicate_handler,
    make_preset_favorite_handler,
    make_preset_save_handler,
    make_preset_search_handler,
    make_preset_select_handler,
    make_preset_update_handler,
    make_recent_checkpoint_handler,
    make_recent_lora_add_handler,
    make_recent_settings_handler,
    make_recent_vae_handler,
    preset_apply_output_count,
)
from runpod_sdxl_image_studio.ui.tabs.queue_tab import (
    build_queue_tab,
    make_queue_ambiguous_fail_handler,
    make_queue_ambiguous_link_handler,
    make_queue_cancel_handler,
    make_queue_detail_handler,
    make_queue_refresh_handler,
    make_queue_retry_batch_handler,
    make_queue_retry_handler,
)
from runpod_sdxl_image_studio.ui.tabs.system_tab import (
    build_generation_tab,
    build_system_tab,
    capability_refresh_outputs,
    disable_batch_enqueue_button,
    disable_generate_button,
    interactive_action_updates,
    make_batch_enqueue_handler,
    make_check_connection_handler,
    make_custom_size_delete_handler,
    make_custom_size_refresh_handler,
    make_enqueue_handler,
    make_interactive_cancel_handler,
    make_interactive_gallery_restore_handler,
    make_interactive_gallery_select_handler,
    make_interactive_refresh_handler,
    make_interactive_start_handler,
    make_pod_lifecycle_readiness_handler,
    make_pod_lifecycle_terminate_handler,
    make_pod_lifecycle_toggle_handler,
    make_refresh_handler,
    make_size_preset_handler,
    make_startup_restore_handler,
    make_state_backup_handler,
    make_system_health_handler,
    startup_restore_outputs,
)
from runpod_sdxl_image_studio.ui.tabs.upscale_tab import (
    begin_upscale_enqueue,
    build_upscale_tab,
    make_latest_parent_selection_handler,
    make_local_upscaler_refresh_handler,
    make_parent_selection_handler,
    make_upscale_enqueue_details_handler,
    make_upscale_plan_handler,
    make_upscale_result_handler,
    make_upscale_visibility_handler,
)
from runpod_sdxl_image_studio.ui.view_models import initial_status_markdown, state_sync_markdown
from runpod_sdxl_image_studio.workflows.loader import load_txt2img_template, load_workflow_template

APP_TITLE = "RunPod SDXL Image Studio"
APP_CSS = mobile_ui_css()


@dataclass(frozen=True)
class ApplicationRuntime:
    """The app and its one explicitly owned background worker runtime."""

    demo: gr.Blocks
    queue_runtime: GenerationQueueRuntime
    drive_sync_runtime: DriveSyncRuntime | None = None
    model_transfer_runtime: ModelTransferRuntime | None = None
    auto_terminate_runtime: AutoTerminateRuntime | None = None
    startup_model_restore_runtime: StartupModelRestoreRuntime | None = None
    state_sync_service: StateSyncService | None = None
    stateless_reconciliation_service: StatelessReconciliationService | None = None
    run_stateless_reconciliation: bool = False

    def start(self) -> None:
        """Start the process-level queue worker."""

        if self.run_stateless_reconciliation and self.stateless_reconciliation_service is not None:
            result = self.stateless_reconciliation_service.reconcile()
            if not result.is_success:
                raise RuntimeError("stateless reconciliation failed; workers were not started")
        # Form/model restoration starts asynchronously before queue workers so
        # its durable transfer jobs are visible to the normal worker loop.
        if self.startup_model_restore_runtime is not None:
            self.startup_model_restore_runtime.start()
        self.queue_runtime.start()
        if self.drive_sync_runtime is not None:
            self.drive_sync_runtime.start()
        if self.model_transfer_runtime is not None:
            self.model_transfer_runtime.start()
        if self.auto_terminate_runtime is not None:
            self.auto_terminate_runtime.start()

    def stop(self) -> None:
        """Stop the process-level queue worker."""

        if self.auto_terminate_runtime is not None:
            self.auto_terminate_runtime.stop()
        if self.startup_model_restore_runtime is not None:
            self.startup_model_restore_runtime.stop()
        if self.drive_sync_runtime is not None:
            self.drive_sync_runtime.stop()
        if self.model_transfer_runtime is not None:
            self.model_transfer_runtime.stop()
        self.queue_runtime.stop()
        if self.state_sync_service is not None:
            self.state_sync_service.close()


def build_app(
    settings: Settings | None = None,
    service: ComfyUIService | None = None,
    *,
    initial_remote_sha256: str | None = None,
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
    dispatch_queue_repository = GenerationDispatchQueueRepository(session_factory)
    drive_sync_repository = DriveSyncRepository(session_factory)
    model_transfer_repository = ModelTransferRepository(session_factory)
    system_error_repository = SystemErrorEventRepository(session_factory)
    start_repository = GenerationStartRepository(session_factory)
    interactive_run_repository = InteractiveRunRepository(session_factory)
    custom_size_repository = GenerationCustomSizeRepository(session_factory)
    progress_repository = GenerationProgressRepository(session_factory)
    drive_adapter = GoogleDriveAdapter(app_settings)
    state_sync_service = StateSyncService(
        app_settings,
        storage=StateBackupStorage(app_settings, drive_adapter),
        initial_remote_sha256=initial_remote_sha256,
    )
    upscale_settings_repository = UpscaleSettingsRepository(session_factory)

    def legacy_upscaler_name(generation_id: UUID) -> str | None:
        snapshot = upscale_settings_repository.get_by_generation(generation_id)
        return snapshot.upscaler_name if snapshot is not None else None

    form_state_repository = GenerationFormStateRepository(session_factory)
    form_state_service = GenerationFormStateService(
        form_state_repository,
        generation_repository.get_latest,
        upscaler_provider=legacy_upscaler_name,
        state_changed_callback=state_sync_service.mark_dirty,
    )
    lifecycle_service = PodLifecycleService(
        PodLifecycleRepository(session_factory),
        generation_repository,
        dispatch_queue_repository,
        model_transfer_repository,
        drive_sync_repository,
        state_sync_service,
        RunPodLifecycleAdapter(),
        settings=app_settings,
        comfyui_queue_provider=client.get_queue_status,
    )
    lifecycle_service.initialize_session()
    form_state_service.set_work_gate(lifecycle_service)
    custom_size_service = GenerationCustomSizeService(
        custom_size_repository,
        app_settings,
        state_changed_callback=state_sync_service.mark_dirty,
        work_gate=lifecycle_service,
    )
    catalog_service = LoraCatalogService(
        LoraMetadataRepository(create_session_factory(database_engine)),
        LoraThumbnailStorage(
            app_settings.data_dir / "lora_thumbnails",
            app_settings.max_lora_thumbnail_bytes,
            app_settings.lora_thumbnail_max_edge,
        ),
        state_changed_callback=state_sync_service.mark_dirty,
        work_gate=lifecycle_service,
    )
    remote_model_adapter = RemoteModelCatalogAdapter(app_settings)
    model_preparation_service = ModelPreparationService(
        model_transfer_repository,
        remote_model_adapter,
        app_settings,
        comfyui_service.refresh_capabilities,
        lora_catalog_service=catalog_service,
        state_changed_callback=state_sync_service.mark_dirty,
        work_gate=lifecycle_service,
    )
    loaded_workflow = load_txt2img_template(
        app_settings.workflow_dir.parent if app_settings.workflow_dir.exists() else None
    )
    workflow_root = app_settings.workflow_dir.parent if app_settings.workflow_dir.exists() else None
    loaded_image_upscale = load_workflow_template("sdxl_image_upscale", workflow_root)
    loaded_latent_upscale = load_workflow_template("sdxl_latent_upscale", workflow_root)
    metadata_import_repository = MetadataImportRepository(session_factory)
    imported_image_storage = ImportedImageStorage(app_settings)
    metadata_import_service = MetadataImportService(
        metadata_import_repository,
        imported_image_storage,
        app_settings,
        work_gate=lifecycle_service,
        state_changed_callback=state_sync_service.mark_dirty,
    )
    upscale_catalog = UpscalerCatalog.scan(app_settings.upscaler_dir)
    upscale_enqueue_service = UpscaleEnqueueService(
        generation_repository,
        artifact_repository,
        dispatch_queue_repository,
        app_settings,
        catalog=upscale_catalog,
        catalog_provider=lambda: UpscalerCatalog.scan(app_settings.upscaler_dir),
        metadata_import_repository=metadata_import_repository,
        imported_image_storage=imported_image_storage,
        upscale_settings_repository=upscale_settings_repository,
        work_gate=lifecycle_service,
        generation_enqueued_callback=lifecycle_service.arm_on_generation_enqueue,
        state_changed_callback=state_sync_service.mark_dirty,
    )

    def set_phase6_capabilities(capabilities: ComfyUICapabilities | None) -> None:
        metadata_import_service.set_capabilities(capabilities)
        upscale_enqueue_service.set_capabilities(capabilities)

    generation_service = GenerationService(
        client,
        WorkflowAdapter(loaded_workflow.as_mapping()),
        ComfyUIWebSocketClient(app_settings),
        LocalStorageAdapter(app_settings),
        comfyui_service.refresh_capabilities,
        app_settings,
        lora_catalog_service=catalog_service,
        persistence=GenerationPersistenceRepositories(
            generation=generation_repository,
            job=job_repository,
            artifact=artifact_repository,
            start=start_repository,
            queue=queue_repository,
            progress=progress_repository,
            completion=completion_repository,
            failure=failure_repository,
        ),
        thumbnail_storage=HistoryThumbnailStorage(app_settings),
        metadata_storage=GenerationMetadataStorage(app_settings.data_dir),
        upscale_settings_repository=upscale_settings_repository,
        upscale_workflow_adapter=UpscaleWorkflowAdapter(
            loaded_image_upscale.as_mapping(), loaded_latent_upscale.as_mapping()
        ),
        upscaler_catalog=upscale_catalog,
        upscaler_catalog_provider=lambda: UpscalerCatalog.scan(app_settings.upscaler_dir),
        metadata_import_repository=metadata_import_repository,
        imported_image_storage=imported_image_storage,
    )
    drive_sync_service = DriveSyncService(
        drive_sync_repository,
        generation_repository,
        artifact_repository,
        app_settings,
        drive_adapter,
        metadata_repair_handler=generation_service.repair_optional_artifacts,
        work_gate=lifecycle_service,
    )
    generation_service.set_drive_sync_enqueue_handler(drive_sync_service.enqueue_generation)
    cancellation_adapter = ComfyUICancellationAdapter(client, app_settings)
    queue_service = GenerationQueueService(
        dispatch_queue_repository,
        app_settings,
        cancellation_adapter,
        upscale_settings_repository=upscale_settings_repository,
        lifecycle_gate=lifecycle_service,
        generation_enqueued_callback=lifecycle_service.arm_on_generation_enqueue,
        state_changed_callback=state_sync_service.mark_dirty,
    )
    interactive_generation_service = InteractiveGenerationService(
        interactive_run_repository,
        queue_service,
        app_settings,
        artifact_repository=artifact_repository,
        state_changed_callback=state_sync_service.mark_dirty,
    )
    disk_usage_adapter = LocalDiskUsageAdapter()
    preflight_service = GenerationPreflightService(
        comfyui_service,
        app_settings,
        disk_usage_adapter=disk_usage_adapter,
        workflow_template=loaded_workflow.as_mapping(),
        workflow_templates={
            "sdxl_txt2img": loaded_workflow.as_mapping(),
            "sdxl_image_upscale": loaded_image_upscale.as_mapping(),
            "sdxl_latent_upscale": loaded_latent_upscale.as_mapping(),
        },
        drive_status_provider=drive_sync_service.check_connection,
        error_recorder=system_error_repository,
        work_gate=lifecycle_service,
        state_changed_callback=state_sync_service.mark_dirty,
    )
    system_health_service = SystemHealthService(
        comfyui_service,
        queue_service,
        drive_sync_service,
        app_settings,
        disk_usage_adapter=disk_usage_adapter,
        error_history_repository=system_error_repository,
        state_changed_callback=state_sync_service.mark_dirty,
        work_gate=lifecycle_service,
    )
    execution_service = GenerationExecutionService(
        generation_service,
        dispatch_queue_repository,
        UpscaleService(generation_service),
    )
    history_service = GenerationHistoryService(
        generation_repository,
        artifact_repository,
        app_settings,
        state_changed_callback=state_sync_service.mark_dirty,
        work_gate=lifecycle_service,
    )
    recovery_service = GenerationRecoveryService(
        client,
        generation_repository,
        job_repository,
        artifact_repository,
        app_settings,
        completed_prompt_handler=generation_service.recover_prompt,
        completed_optional_artifact_handler=generation_service.repair_optional_artifacts,
        failure_repository=failure_repository,
        cancellation_repository=GenerationCancellationRepository(session_factory),
    )
    stateless_reconciliation_service = StatelessReconciliationService(
        dispatch_queue_repository,
        drive_sync_repository,
        model_transfer_repository=model_transfer_repository,
        state_changed_callback=state_sync_service.mark_dirty,
    )

    async def reconcile_queue_item(item: GenerationQueueItem) -> ReconciliationOutcome:
        queue_item = item
        if (
            queue_item.job.prompt_id is not None
            and queue_item.generation.comfy_prompt_id is not None
            and queue_item.job.prompt_id != queue_item.generation.comfy_prompt_id
        ):
            return ReconciliationOutcome.UNAVAILABLE
        prompt_id = queue_item.job.prompt_id or queue_item.generation.comfy_prompt_id
        if not prompt_id:
            return ReconciliationOutcome.UNAVAILABLE
        return await generation_service.reconcile_prompt(queue_item.generation.id, prompt_id)

    async def repair_one_optional_artifact() -> tuple[str, ...]:
        return await recovery_service.repair_completed_optional_artifacts(1)

    def generation_state_changed() -> None:
        state_sync_service.mark_dirty()
        interactive_generation_service.reconcile_after_queue_change()

    queue_worker = GenerationQueueWorker(
        dispatch_queue_repository,
        execution_service,
        app_settings,
        reconcile_handler=reconcile_queue_item,
        cancellation_adapter=cancellation_adapter,
        completed_optional_artifact_handler=repair_one_optional_artifact,
        state_changed_callback=generation_state_changed,
    )
    queue_runtime = GenerationQueueRuntime(queue_worker)
    drive_sync_worker = DriveSyncWorker(
        drive_sync_repository,
        drive_sync_service,
        app_settings,
        state_changed_callback=state_sync_service.mark_dirty,
    )
    drive_sync_runtime = DriveSyncRuntime(drive_sync_worker)
    model_transfer_worker = ModelTransferWorker(
        model_transfer_repository,
        model_preparation_service,
        app_settings,
        state_changed_callback=state_sync_service.mark_dirty,
    )
    model_transfer_runtime = ModelTransferRuntime(model_transfer_worker)
    startup_model_restore_runtime = StartupModelRestoreRuntime(
        form_state_service,
        model_preparation_service,
        app_settings,
        capability_refresh=comfyui_service.refresh_capabilities,
    )
    auto_terminate_runtime = AutoTerminateRuntime(
        AutoTerminateCoordinator(
            lifecycle_service,
            app_settings,
            startup_restore_ready=lambda: startup_model_restore_runtime.is_ready,
        ),
        app_settings,
    )
    queue_service.set_wake_callback(queue_runtime.wake)
    preset_repository = PresetRepository(session_factory)
    preset_service = PresetService(
        preset_repository,
        app_settings,
        work_gate=lifecycle_service,
        state_changed_callback=state_sync_service.mark_dirty,
    )
    recent_settings_service = RecentSettingsService(
        generation_repository,
        preset_repository,
        limit=app_settings.recent_settings_limit,
    )
    generation_diff_service = GenerationDiffService()
    mobile_status_handler = make_mobile_status_refresh_handler(
        queue_service,
        history_service,
    )
    mobile_status_poll_handler = make_mobile_status_poll_handler(
        queue_service,
        history_service,
    )

    try:
        initial_custom_size_choices = custom_size_service.selector_options()
    except GenerationCustomSizeError:
        # An older database is still usable until the normal startup migration runs.
        initial_custom_size_choices = []

    with gr.Blocks(title=APP_TITLE, css=APP_CSS) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        with gr.Tab("生成"):
            generation = build_generation_tab(
                app_settings.max_loras,
                custom_size_choices=initial_custom_size_choices,
            )
        with gr.Tab("システム"):
            system = build_system_tab(
                app_settings.comfyui_base_url,
                initial_status_markdown(),
                initial_state_sync_markdown=state_sync_markdown(
                    state_sync_service.get_status(), app_settings.timezone
                ),
            )
        with gr.Tab("キュー"):
            (
                queue_refresh,
                queue_status,
                queue_batch_filter,
                queue_jobs,
                queue_detail,
                queue_cancel,
                queue_retry,
                queue_retry_batch,
                queue_message,
                queue_ambiguous_prompt_id,
                queue_ambiguous_link,
                queue_ambiguous_fail,
            ) = build_queue_tab()
        with gr.Tab("LoRA管理"):
            lora_management = build_lora_management_tab(catalog_service)
        with gr.Tab("履歴"):
            history = build_history_tab()
        with gr.Tab("アップスケール"):
            upscale = build_upscale_tab(upscale_catalog)
        with gr.Tab("プリセット"):
            presets = build_preset_tab()
        with gr.Tab("外部metadata"):
            metadata_import = build_metadata_import_tab(app_settings.max_loras)
        with gr.Tab("同期・設定"):
            drive_sync = build_drive_sync_tab()
        with gr.Tab("モデル準備"):
            model_preparation = build_model_preparation_tab(app_settings.max_loras)

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

        health_handler = make_system_health_handler(
            system_health_service,
            app_settings.timezone,
        )
        demo.load(
            fn=health_handler,
            outputs=[system.health_markdown, system.error_history_markdown],
            concurrency_limit=1,
        )
        system.health_refresh_button.click(
            fn=health_handler,
            outputs=[system.health_markdown, system.error_history_markdown],
            concurrency_limit=1,
        )
        state_backup_event = system.state_backup_button.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[system.state_backup_button],
            queue=False,
        )
        state_backup_event.then(
            fn=make_state_backup_handler(state_sync_service, app_settings.timezone),
            outputs=[system.state_sync_markdown, system.state_sync_message],
            concurrency_limit=1,
        ).then(
            fn=lambda: gr.Button(interactive=True),
            outputs=[system.state_backup_button],
            queue=False,
        )
        system.lifecycle_refresh_button.click(
            fn=make_pod_lifecycle_readiness_handler(lifecycle_service),
            outputs=[system.lifecycle_markdown, system.lifecycle_message],
            concurrency_limit=1,
        )
        system.lifecycle_terminate_button.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[system.lifecycle_terminate_button],
            queue=False,
        ).then(
            fn=make_pod_lifecycle_terminate_handler(lifecycle_service),
            outputs=[system.lifecycle_markdown, system.lifecycle_message],
            concurrency_limit=1,
        ).then(
            fn=lambda: gr.Button(interactive=True),
            outputs=[system.lifecycle_terminate_button],
            queue=False,
        )
        system.lifecycle_toggle_button.click(
            fn=make_pod_lifecycle_toggle_handler(lifecycle_service),
            outputs=[system.lifecycle_markdown, system.lifecycle_message],
            queue=False,
        )
        demo.load(
            fn=make_pod_lifecycle_readiness_handler(lifecycle_service),
            outputs=[system.lifecycle_markdown, system.lifecycle_message],
            concurrency_limit=1,
        )

        mobile_status_inputs = [
            generation.active_generation_id,
            generation.status_card,
            generation.result_image,
            generation.result_details,
            generation.result_seed,
            generation.result_favorite,
        ]
        mobile_status_outputs = [
            generation.active_generation_id,
            generation.status_card,
            generation.result_image,
            generation.result_details,
            generation.result_seed,
            generation.result_favorite,
            generation.result_message,
        ]
        mobile_status_poll_outputs = mobile_status_outputs[1:]
        generation.status_poll_timer.tick(
            fn=mobile_status_poll_handler,
            inputs=mobile_status_inputs,
            outputs=mobile_status_poll_outputs,
            concurrency_limit=1,
        )
        demo.load(
            fn=mobile_status_handler,
            inputs=mobile_status_inputs,
            outputs=mobile_status_outputs,
            concurrency_limit=1,
        )
        demo.load(
            fn=make_refresh_handler(
                comfyui_service,
                generation,
                catalog_service,
                set_phase6_capabilities,
            ),
            inputs=[
                generation.checkpoint,
                generation.vae,
                generation.sampler,
                generation.scheduler,
                generation.upscaler,
                generation.lora_editor.state,
                generation.lora_editor.choices,
                generation.lora_category_filter,
            ],
            outputs=[system.capability_message, *capability_outputs],
            concurrency_limit=1,
        )
        generation.startup_restore_timer.tick(
            fn=make_startup_restore_handler(
                startup_model_restore_runtime,
                comfyui_service,
                generation,
                catalog_service,
                set_phase6_capabilities,
            ),
            inputs=[*capability_inputs, generation.startup_restore_applied],
            outputs=startup_restore_outputs(generation, system.capability_message),
            concurrency_limit=1,
        )
        demo.load(
            fn=make_recent_settings_handler(recent_settings_service, preset_service),
            inputs=[
                generation.checkpoint_choices,
                generation.vae_choices,
                generation.lora_editor.choices,
            ],
            outputs=[
                generation.recent_checkpoints,
                generation.recent_vaes,
                generation.recent_loras,
                generation.recent_generation_presets,
                generation.recent_prompt_presets,
                generation.recent_lora_presets,
                generation.recent_message,
            ],
        )

        drive_sync.refresh_button.click(
            fn=make_drive_sync_refresh_handler(drive_sync_service),
            outputs=[
                drive_sync.selected_job,
                drive_sync.summary,
                drive_sync.jobs,
                drive_sync.message,
            ],
        )
        drive_sync.connection_button.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[drive_sync.connection_button],
            queue=False,
        ).then(
            fn=make_drive_connection_handler(drive_sync_service),
            outputs=[
                drive_sync.connection_button,
                drive_sync.connection_status,
                drive_sync.message,
            ],
        )
        drive_sync.discovery_button.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[drive_sync.discovery_button],
            queue=False,
        ).then(
            fn=make_drive_discovery_handler(drive_sync_service),
            outputs=[drive_sync.discovery_button, drive_sync.message],
        )
        drive_sync.retry_selected_button.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[drive_sync.retry_selected_button],
            queue=False,
        ).then(
            fn=make_drive_retry_selected_handler(drive_sync_service),
            inputs=[drive_sync.selected_job],
            outputs=[drive_sync.retry_selected_button, drive_sync.message],
        )
        drive_sync.retry_failed_button.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[drive_sync.retry_failed_button],
            queue=False,
        ).then(
            fn=make_drive_retry_failed_handler(drive_sync_service),
            outputs=[drive_sync.retry_failed_button, drive_sync.message],
        )
        drive_sync.resync_button.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[drive_sync.resync_button],
            queue=False,
        ).then(
            fn=make_drive_resync_handler(drive_sync_service),
            outputs=[drive_sync.resync_button, drive_sync.message],
        )
        drive_sync.manifest_button.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[drive_sync.manifest_button],
            queue=False,
        ).then(
            fn=make_drive_manifest_handler(drive_sync_service),
            inputs=[drive_sync.manifest_date],
            outputs=[drive_sync.manifest_button, drive_sync.message],
        )
        drive_sync.failed_manifest_button.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[drive_sync.failed_manifest_button],
            queue=False,
        ).then(
            fn=make_drive_failed_manifest_handler(drive_sync_service),
            outputs=[drive_sync.failed_manifest_button, drive_sync.message],
        )

        model_preparation.refresh_button.click(
            fn=make_model_catalog_refresh_handler(model_preparation_service),
            outputs=[
                model_preparation.checkpoint,
                model_preparation.vae,
                model_preparation.loras,
                model_preparation.upscaler,
                model_preparation.message,
            ],
            concurrency_limit=1,
        )
        model_preparation.prepare_button.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[model_preparation.prepare_button],
            queue=False,
        ).then(
            fn=make_model_prepare_handler(model_preparation_service),
            inputs=[
                model_preparation.checkpoint,
                model_preparation.vae,
                model_preparation.loras,
                model_preparation.upscaler,
            ],
            outputs=[model_preparation.status, model_preparation.message],
            concurrency_limit=1,
        ).then(
            fn=lambda: gr.Button(interactive=True),
            outputs=[model_preparation.prepare_button],
            queue=False,
        )
        model_preparation.jobs_refresh_button.click(
            fn=make_model_jobs_refresh_handler(model_preparation_service),
            outputs=[model_preparation.jobs, model_preparation.status],
            concurrency_limit=1,
        ).then(
            fn=make_refresh_handler(
                comfyui_service,
                generation,
                catalog_service,
                set_phase6_capabilities,
            ),
            inputs=capability_inputs,
            outputs=[system.capability_message, *capability_outputs],
            concurrency_limit=1,
        ).then(
            fn=make_local_upscaler_refresh_handler(app_settings),
            inputs=[upscale.upscaler_name],
            outputs=[upscale.upscaler_name, upscale.catalog_message],
            queue=False,
        )
        model_preparation.cancel_button.click(
            fn=make_model_cancel_handler(model_preparation_service),
            inputs=[model_preparation.jobs],
            outputs=[model_preparation.status, model_preparation.message],
            concurrency_limit=1,
        ).then(fn=model_transfer_runtime.wake, outputs=[], queue=False)
        model_preparation.retry_button.click(
            fn=make_model_retry_handler(model_preparation_service),
            inputs=[model_preparation.jobs],
            outputs=[model_preparation.status, model_preparation.message],
            concurrency_limit=1,
        ).then(fn=model_transfer_runtime.wake, outputs=[], queue=False)
        demo.load(
            fn=make_model_jobs_refresh_handler(model_preparation_service),
            outputs=[model_preparation.jobs, model_preparation.status],
            concurrency_limit=1,
        )

        metadata_import_outputs = [
            metadata_import.import_id,
            metadata_import.preview_image,
            metadata_import.image_hash,
            metadata_import.image_dimensions,
            metadata_import.metadata_source,
            metadata_import.source_selection,
            metadata_import.confirm_sidecar_hash,
            metadata_import.select_source_button,
            metadata_import.status,
            metadata_import.warnings,
            metadata_import.unresolved,
            metadata_import.raw_metadata,
            metadata_import.settings_preview,
            metadata_import.apply_generation,
            metadata_import.apply_upscale,
            metadata_import.parse_button,
        ]
        metadata_import.parse_button.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[metadata_import.parse_button],
            queue=False,
        ).then(
            fn=make_metadata_import_handler(metadata_import_service),
            inputs=[metadata_import.image, metadata_import.sidecar],
            outputs=metadata_import_outputs,
            concurrency_limit=1,
        )
        metadata_import.apply_mapping.click(
            fn=make_metadata_mapping_handler(metadata_import_service),
            inputs=[metadata_import.import_id, metadata_import.mapping_json],
            outputs=metadata_import_outputs,
            concurrency_limit=1,
        )
        metadata_import.select_source_button.click(
            fn=make_metadata_source_selection_handler(metadata_import_service),
            inputs=[
                metadata_import.import_id,
                metadata_import.source_selection,
                metadata_import.confirm_sidecar_hash,
            ],
            outputs=metadata_import_outputs,
            concurrency_limit=1,
        )

        upscale.enqueue_button.click(
            fn=begin_upscale_enqueue,
            outputs=[upscale.enqueue_button],
            queue=False,
        ).then(
            fn=make_upscale_enqueue_details_handler(
                upscale_enqueue_service,
                preflight_service,
            ),
            inputs=[
                upscale.parent_generation_id,
                upscale.method,
                upscale.sizing_mode,
                upscale.scale_factor,
                upscale.target_width,
                upscale.target_height,
                upscale.upscaler_name,
                upscale.denoise,
                upscale.source_import_id,
            ],
            outputs=[upscale.enqueue_button, upscale.status],
            concurrency_limit=1,
        )
        upscale.latest_button.click(
            fn=make_latest_parent_selection_handler(upscale_enqueue_service),
            outputs=[upscale.parent_generation_id, upscale.source_preview, upscale.status],
            concurrency_limit=1,
        ).then(
            fn=lambda: "",
            outputs=[upscale.source_import_id],
            queue=False,
        )
        history.selected.change(
            fn=make_parent_selection_handler(upscale_enqueue_service),
            inputs=[history.selected],
            outputs=[upscale.parent_generation_id, upscale.source_preview, upscale.status],
            concurrency_limit=1,
        )
        history.selected.change(
            fn=lambda: "",
            outputs=[upscale.source_import_id],
            queue=False,
        )
        history.selected.change(
            fn=make_upscale_result_handler(upscale_enqueue_service),
            inputs=[history.selected],
            outputs=[upscale.result, upscale.comparison, upscale.status],
            concurrency_limit=1,
        )
        upscale.method.change(
            fn=make_upscale_visibility_handler(),
            inputs=[upscale.method],
            outputs=[upscale.upscaler_name, upscale.denoise],
            queue=False,
        )
        demo.load(
            fn=make_upscale_visibility_handler(),
            inputs=[upscale.method],
            outputs=[upscale.upscaler_name, upscale.denoise],
            queue=False,
        )
        upscale_plan_inputs = [
            upscale.parent_generation_id,
            upscale.method,
            upscale.sizing_mode,
            upscale.scale_factor,
            upscale.target_width,
            upscale.target_height,
            upscale.upscaler_name,
            upscale.denoise,
            upscale.source_import_id,
        ]
        for component in upscale_plan_inputs:
            component.change(
                fn=make_upscale_plan_handler(upscale_enqueue_service),
                inputs=upscale_plan_inputs,
                outputs=[upscale.plan],
                queue=False,
            )

        system.connection_button.click(
            fn=make_check_connection_handler(
                comfyui_service,
                app_settings.timezone,
                generation,
                catalog_service,
                set_phase6_capabilities,
            ),
            inputs=capability_inputs,
            outputs=[system.status_markdown, system.capability_message, *capability_outputs],
        )
        system.refresh_button.click(
            fn=make_refresh_handler(
                comfyui_service,
                generation,
                catalog_service,
                set_phase6_capabilities,
            ),
            inputs=capability_inputs,
            outputs=[system.capability_message, *capability_outputs],
        )
        generation.size_preset.change(
            fn=make_size_preset_handler(custom_size_service),
            inputs=[generation.size_preset],
            outputs=[
                generation.width,
                generation.height,
                generation.custom_size_delete_button,
                generation.custom_size_message,
            ],
            concurrency_limit=1,
        )
        generation.custom_size_delete_button.click(
            fn=make_custom_size_delete_handler(custom_size_service),
            inputs=[generation.size_preset],
            outputs=[
                generation.size_preset,
                generation.custom_size_delete_button,
                generation.custom_size_message,
            ],
            concurrency_limit=1,
        )
        generation.positive_paste_button.click(
            fn=None,
            inputs=[generation.positive_prompt],
            outputs=[generation.positive_prompt, generation.positive_clipboard_message],
            js=(
                "async value => {"
                "try {"
                "if (!navigator.clipboard) throw new Error('clipboard');"
                "const text = await navigator.clipboard.readText();"
                "return [text, 'クリップボードから貼り付けました。'];"
                "} catch (_) {"
                "return [value, 'クリップボードを利用できませんでした'];"
                "}"
                "}"
            ),
            queue=False,
        )
        generation.positive_copy_button.click(
            fn=None,
            inputs=[generation.positive_prompt],
            outputs=[generation.positive_clipboard_message],
            js=(
                "async value => {"
                "try {"
                "if (!navigator.clipboard) throw new Error('clipboard');"
                "await navigator.clipboard.writeText(value || '');"
                "return ['クリップボードへコピーしました。'];"
                "} catch (_) {"
                "return ['クリップボードを利用できませんでした'];"
                "}"
                "}"
            ),
            queue=False,
        )
        generation.negative_paste_button.click(
            fn=None,
            inputs=[generation.negative_prompt],
            outputs=[generation.negative_prompt, generation.negative_clipboard_message],
            js=(
                "async value => {"
                "try {"
                "if (!navigator.clipboard) throw new Error('clipboard');"
                "const text = await navigator.clipboard.readText();"
                "return [text, 'クリップボードから貼り付けました。'];"
                "} catch (_) {"
                "return [value, 'クリップボードを利用できませんでした'];"
                "}"
                "}"
            ),
            queue=False,
        )
        generation.negative_copy_button.click(
            fn=None,
            inputs=[generation.negative_prompt],
            outputs=[generation.negative_clipboard_message],
            js=(
                "async value => {"
                "try {"
                "if (!navigator.clipboard) throw new Error('clipboard');"
                "await navigator.clipboard.writeText(value || '');"
                "return ['クリップボードへコピーしました。'];"
                "} catch (_) {"
                "return ['クリップボードを利用できませんでした'];"
                "}"
                "}"
            ),
            queue=False,
        )
        generation.positive_clear_button.click(
            fn=lambda: "",
            outputs=[generation.positive_prompt],
            queue=False,
        )
        generation.negative_clear_button.click(
            fn=lambda: "",
            outputs=[generation.negative_prompt],
            queue=False,
        )
        batch_enqueue_event = generation.batch_enqueue_button.click(
            fn=disable_batch_enqueue_button,
            outputs=[generation.batch_enqueue_button],
            queue=False,
        )
        batch_enqueue_event.then(
            fn=make_batch_enqueue_handler(
                queue_service,
                app_settings.max_loras,
                preflight_service,
                form_state_service.save,
            ),
            inputs=[
                generation.checkpoint,
                generation.positive_prompt,
                generation.negative_prompt,
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
                generation.batch_count,
                generation.batch_seed_strategy,
                generation.batch_start_seed,
                generation.batch_seed_step,
                generation.batch_name,
                generation.upscaler,
                generation.clip_skip,
                generation.hires_fix,
                generation.hires_scale,
                generation.hires_resize_method,
                generation.hires_steps,
                generation.hires_cfg_scale,
                generation.hires_sampler,
                generation.hires_scheduler,
                generation.hires_denoise,
                generation.final_upscale,
            ],
            outputs=[
                generation.batch_enqueue_button,
                generation.progress,
                generation.batch_message,
            ],
            concurrency_limit=1,
        ).then(
            fn=mobile_status_poll_handler,
            inputs=mobile_status_inputs,
            outputs=mobile_status_poll_outputs,
            concurrency_limit=1,
        )
        demo.load(
            fn=None,
            outputs=[generation.interactive_client_local_date],
            js=(
                "() => {"
                "const now = new Date();"
                "const pad = value => String(value).padStart(2, '0');"
                "return [`${now.getFullYear()}-${pad(now.getMonth() + 1)}-"
                "${pad(now.getDate())}`];"
                "}"
            ),
            queue=False,
        )
        interactive_inputs = [
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
            generation.batch_count,
            generation.batch_size,
            generation.upscaler,
            generation.clip_skip,
            generation.hires_fix,
            generation.hires_scale,
            generation.hires_resize_method,
            generation.hires_steps,
            generation.hires_cfg_scale,
            generation.hires_sampler,
            generation.hires_scheduler,
            generation.hires_denoise,
            generation.final_upscale,
            generation.interactive_client_local_date,
            generation.workflow_template_id,
            generation.workflow_template_version,
        ]
        interactive_disable_event = generation.generate_button.click(
            fn=disable_generate_button,
            outputs=[generation.generate_button],
            queue=False,
        )
        interactive_date_event = interactive_disable_event.then(
            fn=lambda value: value,
            inputs=[generation.interactive_client_local_date],
            outputs=[generation.interactive_client_local_date],
            js=(
                "() => {"
                "const now = new Date();"
                "const pad = value => String(value).padStart(2, '0');"
                "return [`${now.getFullYear()}-${pad(now.getMonth() + 1)}-"
                "${pad(now.getDate())}`];"
                "}"
            ),
            queue=True,
        )
        interactive_start_result = interactive_date_event.then(
            fn=make_interactive_start_handler(
                interactive_generation_service,
                app_settings.max_loras,
                preflight_service,
                form_state_service.save,
            ),
            inputs=interactive_inputs,
            outputs=[
                generation.interactive_run_id,
                generation.interactive_status,
                generation.interactive_result_gallery,
                generation.batch_message,
            ],
            concurrency_limit=1,
            concurrency_id="interactive-generation",
        )
        interactive_start_result.then(
            fn=interactive_action_updates,
            inputs=[generation.interactive_status],
            outputs=[generation.generate_button, generation.interactive_cancel_button],
            queue=False,
        ).then(
            fn=make_custom_size_refresh_handler(
                custom_size_service,
                interactive_generation_service,
            ),
            inputs=[generation.interactive_run_id, generation.size_preset],
            outputs=[generation.size_preset, generation.custom_size_message],
            concurrency_limit=1,
        )
        cancel_event = generation.interactive_cancel_button.click(
            fn=lambda: gr.Button(value="キャンセル中...", interactive=False),
            outputs=[generation.interactive_cancel_button],
            queue=False,
        )
        cancel_result = cancel_event.then(
            fn=make_interactive_cancel_handler(interactive_generation_service),
            inputs=[generation.interactive_run_id],
            outputs=[
                generation.interactive_run_id,
                generation.interactive_status,
                generation.interactive_result_gallery,
                generation.batch_message,
            ],
            concurrency_limit=1,
            concurrency_id="interactive-generation",
        )
        cancel_result.then(
            fn=interactive_action_updates,
            inputs=[generation.interactive_status],
            outputs=[generation.generate_button, generation.interactive_cancel_button],
            queue=False,
        )
        restore_interactive_event = demo.load(
            fn=make_interactive_refresh_handler(interactive_generation_service),
            inputs=[generation.interactive_run_id],
            outputs=[
                generation.interactive_run_id,
                generation.interactive_status,
                generation.interactive_result_gallery,
                generation.batch_message,
            ],
            concurrency_limit=1,
            concurrency_id="interactive-generation",
        )
        restore_interactive_event.then(
            fn=interactive_action_updates,
            inputs=[generation.interactive_status],
            outputs=[generation.generate_button, generation.interactive_cancel_button],
            queue=False,
        ).then(
            fn=make_custom_size_refresh_handler(
                custom_size_service,
                interactive_generation_service,
            ),
            inputs=[generation.interactive_run_id, generation.size_preset],
            outputs=[generation.size_preset, generation.custom_size_message],
            concurrency_limit=1,
        )
        interactive_poll_event = generation.interactive_poll_timer.tick(
            fn=make_interactive_refresh_handler(interactive_generation_service),
            inputs=[generation.interactive_run_id],
            outputs=[
                generation.interactive_run_id,
                generation.interactive_status,
                generation.interactive_result_gallery,
                generation.batch_message,
            ],
            concurrency_limit=1,
            concurrency_id="interactive-generation",
        )
        interactive_poll_event.then(
            fn=interactive_action_updates,
            inputs=[generation.interactive_status],
            outputs=[generation.generate_button, generation.interactive_cancel_button],
            queue=False,
        ).then(
            fn=make_custom_size_refresh_handler(
                custom_size_service,
                interactive_generation_service,
            ),
            inputs=[generation.interactive_run_id, generation.size_preset],
            outputs=[generation.size_preset, generation.custom_size_message],
            concurrency_limit=1,
        )
        generation.interactive_result_gallery.select(
            fn=make_interactive_gallery_select_handler(),
            outputs=[generation.interactive_selected_image_index],
            queue=False,
        )
        queue_refresh.click(
            fn=make_queue_refresh_handler(queue_service),
            inputs=[queue_status, queue_batch_filter],
            outputs=[queue_jobs, queue_message],
        )
        queue_status.change(
            fn=make_queue_refresh_handler(queue_service),
            inputs=[queue_status, queue_batch_filter],
            outputs=[queue_jobs, queue_message],
        )
        queue_jobs.change(
            fn=make_queue_detail_handler(queue_service),
            inputs=[queue_jobs],
            outputs=[
                queue_detail,
                queue_ambiguous_prompt_id,
                queue_ambiguous_link,
                queue_ambiguous_fail,
            ],
        )
        queue_cancel_event = queue_cancel.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[queue_cancel],
            queue=False,
        )
        queue_cancel_event.then(
            fn=make_queue_cancel_handler(queue_service),
            inputs=[queue_jobs],
            outputs=[queue_cancel, queue_message],
            concurrency_limit=1,
        ).then(
            fn=make_queue_refresh_handler(queue_service),
            inputs=[queue_status, queue_batch_filter],
            outputs=[queue_jobs, queue_message],
        )
        ambiguous_link_event = queue_ambiguous_link.click(
            fn=lambda: (gr.Button(interactive=False), gr.Button(interactive=False)),
            outputs=[queue_ambiguous_link, queue_ambiguous_fail],
            queue=False,
        )
        ambiguous_link_event.then(
            fn=make_queue_ambiguous_link_handler(queue_service),
            inputs=[queue_jobs, queue_ambiguous_prompt_id],
            outputs=[
                queue_ambiguous_prompt_id,
                queue_ambiguous_link,
                queue_ambiguous_fail,
                queue_message,
            ],
            concurrency_limit=1,
        ).then(
            fn=make_queue_refresh_handler(queue_service),
            inputs=[queue_status, queue_batch_filter],
            outputs=[queue_jobs, queue_message],
        ).then(
            fn=make_queue_detail_handler(queue_service),
            inputs=[queue_jobs],
            outputs=[
                queue_detail,
                queue_ambiguous_prompt_id,
                queue_ambiguous_link,
                queue_ambiguous_fail,
            ],
        )
        ambiguous_fail_event = queue_ambiguous_fail.click(
            fn=lambda: (gr.Button(interactive=False), gr.Button(interactive=False)),
            outputs=[queue_ambiguous_link, queue_ambiguous_fail],
            queue=False,
        )
        ambiguous_fail_event.then(
            fn=make_queue_ambiguous_fail_handler(queue_service),
            inputs=[queue_jobs],
            outputs=[
                queue_ambiguous_prompt_id,
                queue_ambiguous_link,
                queue_ambiguous_fail,
                queue_message,
            ],
            concurrency_limit=1,
        ).then(
            fn=make_queue_refresh_handler(queue_service),
            inputs=[queue_status, queue_batch_filter],
            outputs=[queue_jobs, queue_message],
        ).then(
            fn=make_queue_detail_handler(queue_service),
            inputs=[queue_jobs],
            outputs=[
                queue_detail,
                queue_ambiguous_prompt_id,
                queue_ambiguous_link,
                queue_ambiguous_fail,
            ],
        )
        queue_retry_event = queue_retry.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[queue_retry],
            queue=False,
        )
        queue_retry_event.then(
            fn=make_queue_retry_handler(queue_service),
            inputs=[queue_jobs],
            outputs=[queue_retry, queue_message],
            concurrency_limit=1,
        ).then(
            fn=make_queue_refresh_handler(queue_service),
            inputs=[queue_status, queue_batch_filter],
            outputs=[queue_jobs, queue_message],
        )
        queue_retry_batch_event = queue_retry_batch.click(
            fn=lambda: gr.Button(interactive=False),
            outputs=[queue_retry_batch],
            queue=False,
        )
        queue_retry_batch_event.then(
            fn=make_queue_retry_batch_handler(queue_service),
            inputs=[queue_jobs],
            outputs=[queue_retry_batch, queue_message],
            concurrency_limit=1,
        ).then(
            fn=make_queue_refresh_handler(queue_service),
            inputs=[queue_status, queue_batch_filter],
            outputs=[queue_jobs, queue_message],
        )
        presets.refresh.click(
            fn=make_preset_search_handler(preset_service),
            inputs=[presets.search, presets.kind, presets.favorite_only],
            outputs=[presets.results, presets.message],
        )
        preset_form_outputs = [
            presets.results,
            presets.selected,
            presets.preset_kind,
            presets.name,
            presets.description,
            presets.favorite,
            presets.payload_summary,
            presets.prompt_apply_mode,
            presets.lora_apply_mode,
            presets.message,
        ]
        preset_selection_outputs = [
            presets.preset_kind,
            presets.name,
            presets.description,
            presets.favorite,
            presets.payload_summary,
            presets.prompt_apply_mode,
            presets.lora_apply_mode,
            presets.message,
        ]
        preset_save_inputs = [
            presets.preset_kind,
            presets.name,
            presets.description,
            presets.favorite,
            generation.positive_prompt,
            generation.negative_prompt,
            generation.width,
            generation.height,
            generation.seed_mode,
            generation.seed,
            generation.steps,
            generation.cfg_scale,
            generation.sampler,
            generation.scheduler,
            generation.checkpoint,
            generation.vae,
            generation.lora_editor.state,
            presets.prompt_apply_mode,
            presets.prompt_apply_mode,
            presets.search,
            presets.kind,
            presets.favorite_only,
        ]
        presets.results.change(
            fn=lambda selected: selected,
            inputs=[presets.results],
            outputs=[presets.selected],
        ).then(
            fn=make_preset_select_handler(preset_service),
            inputs=[presets.results],
            outputs=preset_selection_outputs,
        )
        presets.selected.change(
            fn=make_preset_select_handler(preset_service),
            inputs=[presets.selected],
            outputs=preset_selection_outputs,
        )
        presets.save_button.click(
            fn=make_preset_save_handler(preset_service, app_settings.max_loras),
            inputs=preset_save_inputs,
            outputs=preset_form_outputs,
        )
        presets.update_button.click(
            fn=make_preset_update_handler(preset_service, app_settings.max_loras),
            inputs=[presets.selected, *preset_save_inputs[1:]],
            outputs=preset_form_outputs,
        )
        presets.duplicate_button.click(
            fn=make_preset_duplicate_handler(preset_service),
            inputs=[presets.selected, presets.search, presets.kind, presets.favorite_only],
            outputs=preset_form_outputs,
        )
        presets.delete_button.click(
            fn=make_preset_delete_handler(preset_service),
            inputs=[
                presets.selected,
                presets.delete_confirmation,
                presets.search,
                presets.kind,
                presets.favorite_only,
            ],
            outputs=preset_form_outputs,
        )
        presets.favorite.change(
            fn=make_preset_favorite_handler(preset_service),
            inputs=[presets.selected, presets.favorite],
            outputs=[presets.favorite, presets.message],
        )
        presets.clear_button.click(
            fn=make_preset_clear_handler(),
            outputs=[
                presets.search,
                presets.kind,
                presets.favorite_only,
                presets.results,
                presets.selected,
                presets.preset_kind,
                presets.name,
                presets.description,
                presets.favorite,
                presets.payload_summary,
                presets.prompt_apply_mode,
                presets.lora_apply_mode,
                presets.message,
            ],
        )
        preset_apply_outputs = [
            presets.message,
            generation.checkpoint,
            generation.vae,
            generation.positive_prompt,
            generation.negative_prompt,
            generation.width,
            generation.height,
            generation.seed_mode,
            generation.seed,
            generation.steps,
            generation.cfg_scale,
            generation.sampler,
            generation.scheduler,
            generation.upscaler,
            generation.clip_skip,
            generation.hires_fix,
            generation.hires_scale,
            generation.hires_resize_method,
            generation.hires_steps,
            generation.hires_cfg_scale,
            generation.hires_sampler,
            generation.hires_scheduler,
            generation.hires_denoise,
            generation.final_upscale,
            generation.lora_editor.state,
            *component_outputs(generation.lora_editor),
            generation.lora_editor.add_button,
            generation.restored_from_generation,
            generation.regeneration_valid,
        ]
        if len(preset_apply_outputs) != preset_apply_output_count(app_settings.max_loras):
            raise RuntimeError("Preset適用イベントのoutputs数が不一致です。")
        presets.apply_button.click(
            fn=make_preset_apply_handler(
                preset_service,
                app_settings.max_loras,
                generation.lora_editor,
            ),
            inputs=[
                presets.selected,
                presets.prompt_apply_mode,
                presets.lora_apply_mode,
                generation.positive_prompt,
                generation.negative_prompt,
                generation.width,
                generation.height,
                generation.seed_mode,
                generation.seed,
                generation.steps,
                generation.cfg_scale,
                generation.sampler,
                generation.scheduler,
                generation.checkpoint,
                generation.vae,
                generation.lora_editor.state,
                generation.checkpoint_choices,
                generation.vae_choices,
                generation.lora_editor.choices,
            ],
            outputs=preset_apply_outputs,
        )
        presets.recent_refresh.click(
            fn=make_recent_settings_handler(recent_settings_service, preset_service),
            inputs=[
                generation.checkpoint_choices,
                generation.vae_choices,
                generation.lora_editor.choices,
            ],
            outputs=[
                presets.recent_checkpoints,
                presets.recent_vaes,
                presets.recent_loras,
                presets.recent_generation_presets,
                presets.recent_prompt_presets,
                presets.recent_lora_presets,
                presets.message,
            ],
        )
        generation.recent_refresh.click(
            fn=make_recent_settings_handler(recent_settings_service, preset_service),
            inputs=[
                generation.checkpoint_choices,
                generation.vae_choices,
                generation.lora_editor.choices,
            ],
            outputs=[
                generation.recent_checkpoints,
                generation.recent_vaes,
                generation.recent_loras,
                generation.recent_generation_presets,
                generation.recent_prompt_presets,
                generation.recent_lora_presets,
                generation.recent_message,
            ],
        )
        presets.recent_checkpoint_apply.click(
            fn=make_recent_checkpoint_handler(),
            inputs=[presets.recent_checkpoints, generation.checkpoint_choices],
            outputs=[generation.checkpoint, presets.message],
        )
        presets.recent_vae_apply.click(
            fn=make_recent_vae_handler(),
            inputs=[presets.recent_vaes, generation.vae_choices],
            outputs=[generation.vae, presets.message],
        )
        generation.recent_checkpoint_apply.click(
            fn=make_recent_checkpoint_handler(),
            inputs=[generation.recent_checkpoints, generation.checkpoint_choices],
            outputs=[generation.checkpoint, generation.recent_message],
        )
        generation.recent_vae_apply.click(
            fn=make_recent_vae_handler(),
            inputs=[generation.recent_vaes, generation.vae_choices],
            outputs=[generation.vae, generation.recent_message],
        )
        presets.recent_lora_add.click(
            fn=make_recent_lora_add_handler(app_settings.max_loras),
            inputs=[
                presets.recent_loras,
                generation.lora_editor.state,
                generation.lora_editor.choices,
            ],
            outputs=[
                generation.lora_editor.state,
                *component_outputs(generation.lora_editor),
                generation.lora_editor.add_button,
                presets.message,
            ],
        )
        generation.recent_lora_add.click(
            fn=make_recent_lora_add_handler(app_settings.max_loras),
            inputs=[
                generation.recent_loras,
                generation.lora_editor.state,
                generation.lora_editor.choices,
            ],
            outputs=[
                generation.lora_editor.state,
                *component_outputs(generation.lora_editor),
                generation.lora_editor.add_button,
                generation.recent_message,
            ],
        )
        recent_preset_apply_outputs = [
            generation.recent_message,
            generation.checkpoint,
            generation.vae,
            generation.positive_prompt,
            generation.negative_prompt,
            generation.width,
            generation.height,
            generation.seed_mode,
            generation.seed,
            generation.steps,
            generation.cfg_scale,
            generation.sampler,
            generation.scheduler,
            generation.upscaler,
            generation.clip_skip,
            generation.hires_fix,
            generation.hires_scale,
            generation.hires_resize_method,
            generation.hires_steps,
            generation.hires_cfg_scale,
            generation.hires_sampler,
            generation.hires_scheduler,
            generation.hires_denoise,
            generation.final_upscale,
            generation.lora_editor.state,
            *component_outputs(generation.lora_editor),
            generation.lora_editor.add_button,
            generation.restored_from_generation,
            generation.regeneration_valid,
        ]
        if len(recent_preset_apply_outputs) != preset_apply_output_count(app_settings.max_loras):
            raise RuntimeError("最近のPreset適用イベントのoutputs数が不一致です。")
        generation.recent_preset_apply.click(
            fn=make_preset_apply_handler(
                preset_service,
                app_settings.max_loras,
                generation.lora_editor,
            ),
            inputs=[
                generation.recent_generation_presets,
                presets.prompt_apply_mode,
                presets.lora_apply_mode,
                generation.positive_prompt,
                generation.negative_prompt,
                generation.width,
                generation.height,
                generation.seed_mode,
                generation.seed,
                generation.steps,
                generation.cfg_scale,
                generation.sampler,
                generation.scheduler,
                generation.checkpoint,
                generation.vae,
                generation.lora_editor.state,
                generation.checkpoint_choices,
                generation.vae_choices,
                generation.lora_editor.choices,
            ],
            outputs=recent_preset_apply_outputs,
        )
        for recent in (
            presets.recent_generation_presets,
            presets.recent_prompt_presets,
            presets.recent_lora_presets,
        ):
            recent.change(fn=lambda selected: selected, inputs=[recent], outputs=[presets.selected])

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
            fn=make_history_refresh_handler(history_service, recovery_service, reset_page=True),
            inputs=[
                history.page_state,
                history.date_filter,
                history.status_filter,
                history.kind_filter,
                history.favorite_filter,
                history.search_text,
                history.status_search,
                history.kind_search,
                history.parent_search,
                history.date_from_search,
                history.date_to_search,
                history.checkpoint_search,
                history.vae_search,
                history.lora_search,
                history.lora_search_mode,
                history.seed_search,
                history.width_search,
                history.height_search,
                history.error_code_search,
                history.sort_search,
            ],
            outputs=[
                history.page_state,
                history.thumbnail_gallery,
                history.cards,
                history.selected,
                history.page,
                history.previous_button,
                history.next_button,
                history.query_summary,
                history.message,
            ],
        )
        history.clear_button.click(
            fn=clear_history_filters,
            outputs=[
                history.page_state,
                history.date_filter,
                history.status_filter,
                history.kind_filter,
                history.favorite_filter,
                history.search_text,
                history.status_search,
                history.kind_search,
                history.parent_search,
                history.date_from_search,
                history.date_to_search,
                history.checkpoint_search,
                history.vae_search,
                history.lora_search,
                history.lora_search_mode,
                history.seed_search,
                history.width_search,
                history.height_search,
                history.error_code_search,
                history.sort_search,
                history.query_summary,
                history.selected,
                history.seed_copy,
                history.diff_view,
            ],
        ).then(
            fn=make_history_refresh_handler(history_service, recovery_service, reset_page=True),
            inputs=[
                history.page_state,
                history.date_filter,
                history.status_filter,
                history.kind_filter,
                history.favorite_filter,
                history.search_text,
                history.status_search,
                history.kind_search,
                history.parent_search,
                history.date_from_search,
                history.date_to_search,
                history.checkpoint_search,
                history.vae_search,
                history.lora_search,
                history.lora_search_mode,
                history.seed_search,
                history.width_search,
                history.height_search,
                history.error_code_search,
                history.sort_search,
            ],
            outputs=[
                history.page_state,
                history.thumbnail_gallery,
                history.cards,
                history.selected,
                history.page,
                history.previous_button,
                history.next_button,
                history.query_summary,
                history.message,
            ],
        )
        history.diff_button.click(
            fn=make_generation_diff_handler(history_service, generation_diff_service),
            inputs=[history.selected],
            outputs=[history.diff_view],
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
                history.search_text,
                history.status_search,
                history.kind_search,
                history.parent_search,
                history.date_from_search,
                history.date_to_search,
                history.checkpoint_search,
                history.vae_search,
                history.lora_search,
                history.lora_search_mode,
                history.seed_search,
                history.width_search,
                history.height_search,
                history.error_code_search,
                history.sort_search,
            ],
            outputs=[
                history.page_state,
                history.thumbnail_gallery,
                history.cards,
                history.selected,
                history.page,
                history.previous_button,
                history.next_button,
                history.query_summary,
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
                history.search_text,
                history.status_search,
                history.kind_search,
                history.parent_search,
                history.date_from_search,
                history.date_to_search,
                history.checkpoint_search,
                history.vae_search,
                history.lora_search,
                history.lora_search_mode,
                history.seed_search,
                history.width_search,
                history.height_search,
                history.error_code_search,
                history.sort_search,
            ],
            outputs=[
                history.page_state,
                history.thumbnail_gallery,
                history.cards,
                history.selected,
                history.page,
                history.previous_button,
                history.next_button,
                history.query_summary,
                history.message,
            ],
        )
        history.selected.change(
            fn=make_history_detail_handler(history_service),
            inputs=[history.selected],
            outputs=[
                history.detail,
                history.image,
                history.seed_copy,
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
            generation.upscaler,
            generation.clip_skip,
            generation.hires_fix,
            generation.hires_scale,
            generation.hires_resize_method,
            generation.hires_steps,
            generation.hires_cfg_scale,
            generation.hires_sampler,
            generation.hires_scheduler,
            generation.hires_denoise,
            generation.final_upscale,
            generation.lora_editor.state,
            *component_outputs(generation.lora_editor),
            generation.lora_editor.add_button,
            generation.restored_from_generation,
            generation.regeneration_valid,
        ]
        gallery_restore_base = make_restore_handler(history_service, app_settings.max_loras)

        def gallery_restore_delegate(
            selected: str | None,
            checkpoint_choices: object = None,
            vae_choices: object = None,
            lora_choices: object = None,
        ) -> tuple[object, ...]:
            base = gallery_restore_base(selected, checkpoint_choices, vae_choices, lora_choices)
            if not selected or base[-1] is not True:
                return (*base, gr.skip(), gr.skip(), gr.skip(), gr.skip())
            try:
                restored = history_service.restore_settings(
                    UUID(selected),
                    checkpoints=(
                        tuple(value for value in checkpoint_choices if isinstance(value, str))
                        if isinstance(checkpoint_choices, (list, tuple))
                        else None
                    ),
                    vaes=(
                        tuple(value for value in vae_choices if isinstance(value, str))
                        if isinstance(vae_choices, (list, tuple))
                        else None
                    ),
                    loras=(
                        tuple(value for value in lora_choices if isinstance(value, str))
                        if isinstance(lora_choices, (list, tuple))
                        else None
                    ),
                    max_loras=app_settings.max_loras,
                )
            except (ValueError, GenerationHistoryError):
                return (*base, gr.skip(), gr.skip(), gr.skip(), gr.skip())
            values = list(base)
            values[7] = "Fixed"
            settings = restored.settings
            return (
                *values,
                settings.batch_size,
                1,
                settings.workflow_template_id,
                settings.workflow_template_version,
            )

        metadata_import.apply_generation.click(
            fn=make_metadata_generation_apply_handler(
                metadata_import_service,
                app_settings.max_loras,
            ),
            inputs=[
                metadata_import.import_id,
                generation.checkpoint_choices,
                generation.vae_choices,
                generation.lora_editor.choices,
            ],
            outputs=restore_outputs,
        )
        metadata_import.apply_upscale.click(
            fn=make_metadata_upscale_apply_handler(metadata_import_service),
            inputs=[metadata_import.import_id],
            outputs=[
                upscale.source_import_id,
                upscale.parent_generation_id,
                upscale.source_preview,
                upscale.status,
            ],
        )
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
        generation.interactive_restore_button.click(
            fn=make_interactive_gallery_restore_handler(
                interactive_generation_service,
                gallery_restore_delegate,
            ),
            inputs=[
                generation.interactive_run_id,
                generation.interactive_selected_image_index,
                generation.checkpoint_choices,
                generation.vae_choices,
                generation.lora_editor.choices,
            ],
            outputs=[
                *restore_outputs,
                generation.batch_size,
                generation.batch_count,
                generation.workflow_template_id,
                generation.workflow_template_version,
            ],
            concurrency_limit=1,
            concurrency_id="interactive-generation",
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
            generation.upscaler,
            generation.clip_skip,
            generation.hires_fix,
            generation.hires_scale,
            generation.hires_resize_method,
            generation.hires_steps,
            generation.hires_cfg_scale,
            generation.hires_sampler,
            generation.hires_scheduler,
            generation.hires_denoise,
            generation.final_upscale,
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
            fn=make_enqueue_handler(
                queue_service,
                app_settings.max_loras,
                preflight_service,
                form_state_service.save,
            ),
            inputs=generation_inputs,
            outputs=[
                generation.generate_button,
                generation.progress,
                generation.result_image,
                generation.result_details,
                generation.regeneration_requested,
                generation.active_generation_id,
            ],
            concurrency_limit=1,
        )
        regeneration_generation_event.then(
            fn=enable_regeneration_button,
            outputs=[history.regenerate_button],
        ).then(
            fn=mobile_status_poll_handler,
            inputs=mobile_status_inputs,
            outputs=mobile_status_poll_outputs,
        )
        generation.result_edit_button.click(
            fn=make_restore_handler(history_service, app_settings.max_loras),
            inputs=[
                generation.active_generation_id,
                generation.checkpoint_choices,
                generation.vae_choices,
                generation.lora_editor.choices,
            ],
            outputs=restore_outputs,
            concurrency_limit=1,
        )
        generation.result_upscale_button.click(
            fn=make_parent_selection_handler(upscale_enqueue_service),
            inputs=[generation.active_generation_id],
            outputs=[upscale.parent_generation_id, upscale.source_preview, upscale.status],
            concurrency_limit=1,
        )
        generation.result_favorite.input(
            fn=make_history_favorite_handler(history_service),
            inputs=[generation.active_generation_id, generation.result_favorite],
            outputs=[generation.result_favorite, generation.result_message],
        )
        result_regenerate_event = generation.result_regenerate_button.click(
            fn=begin_regeneration,
            outputs=[generation.result_regenerate_button, generation.regeneration_requested],
            queue=False,
        )
        result_regenerate_event = result_regenerate_event.then(
            fn=make_restore_handler(history_service, app_settings.max_loras),
            inputs=[
                generation.active_generation_id,
                generation.checkpoint_choices,
                generation.vae_choices,
                generation.lora_editor.choices,
            ],
            outputs=restore_outputs,
            concurrency_limit=1,
        )
        result_regenerate_enqueue_event = result_regenerate_event.then(
            fn=make_enqueue_handler(
                queue_service,
                app_settings.max_loras,
                preflight_service,
                form_state_service.save,
            ),
            inputs=generation_inputs,
            outputs=[
                generation.generate_button,
                generation.progress,
                generation.result_image,
                generation.result_details,
                generation.regeneration_requested,
                generation.active_generation_id,
            ],
            concurrency_limit=1,
        )
        result_regenerate_enqueue_event.then(
            fn=lambda: gr.Button(value="同条件で再生成", interactive=True),
            outputs=[generation.result_regenerate_button],
        ).then(
            fn=mobile_status_poll_handler,
            inputs=mobile_status_inputs,
            outputs=mobile_status_poll_outputs,
            concurrency_limit=1,
        )
    demo.generation_queue_runtime = queue_runtime
    demo.drive_sync_runtime = drive_sync_runtime
    demo.model_transfer_runtime = model_transfer_runtime
    demo.auto_terminate_runtime = auto_terminate_runtime
    demo.startup_model_restore_runtime = startup_model_restore_runtime
    demo.state_sync_service = state_sync_service
    demo.stateless_reconciliation_service = stateless_reconciliation_service
    demo.interactive_generation_service = interactive_generation_service
    demo.pod_lifecycle_service = lifecycle_service
    return demo


def build_application_runtime(
    settings: Settings | None = None,
    *,
    run_stateless_reconciliation: bool = False,
    initial_remote_sha256: str | None = None,
) -> ApplicationRuntime:
    """Build the demo and obtain its unstarted process-level worker runtime."""

    demo = build_app(settings, initial_remote_sha256=initial_remote_sha256)
    runtime = getattr(demo, "generation_queue_runtime", None)
    if not isinstance(runtime, GenerationQueueRuntime):
        raise RuntimeError("generation queue runtime was not configured")
    drive_runtime = getattr(demo, "drive_sync_runtime", None)
    if not isinstance(drive_runtime, DriveSyncRuntime):
        raise RuntimeError("drive sync runtime was not configured")
    model_runtime = getattr(demo, "model_transfer_runtime", None)
    if not isinstance(model_runtime, ModelTransferRuntime):
        raise RuntimeError("model transfer runtime was not configured")
    auto_terminate_runtime = getattr(demo, "auto_terminate_runtime", None)
    if not isinstance(auto_terminate_runtime, AutoTerminateRuntime):
        raise RuntimeError("auto terminate runtime was not configured")
    startup_model_restore_runtime = getattr(demo, "startup_model_restore_runtime", None)
    if not isinstance(startup_model_restore_runtime, StartupModelRestoreRuntime):
        raise RuntimeError("startup model restore runtime was not configured")
    state_sync_service = getattr(demo, "state_sync_service", None)
    if not isinstance(state_sync_service, StateSyncService):
        raise RuntimeError("state sync service was not configured")
    stateless_reconciliation_service = getattr(demo, "stateless_reconciliation_service", None)
    if not isinstance(stateless_reconciliation_service, StatelessReconciliationService):
        raise RuntimeError("stateless reconciliation service was not configured")
    return ApplicationRuntime(
        demo=demo,
        queue_runtime=runtime,
        drive_sync_runtime=drive_runtime,
        model_transfer_runtime=model_runtime,
        auto_terminate_runtime=auto_terminate_runtime,
        startup_model_restore_runtime=startup_model_restore_runtime,
        state_sync_service=state_sync_service,
        stateless_reconciliation_service=stateless_reconciliation_service,
        run_stateless_reconciliation=run_stateless_reconciliation,
    )


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
