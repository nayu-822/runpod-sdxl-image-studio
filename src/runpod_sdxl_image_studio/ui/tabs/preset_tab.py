"""PresetのUI境界と安全なイベントハンドラ。"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from uuid import UUID

import gradio as gr

from runpod_sdxl_image_studio.domain.generation_settings import GenerationSettings
from runpod_sdxl_image_studio.domain.preset import Preset
from runpod_sdxl_image_studio.domain.preset_payload import (
    PresetKind,
    PromptApplyMode,
    SeedMode,
)
from runpod_sdxl_image_studio.services.preset_service import (
    PresetApplyResult,
    PresetService,
    PresetServiceError,
)
from runpod_sdxl_image_studio.services.recent_settings_service import (
    RecentSettingsService,
    RecentSettingsServiceError,
)
from runpod_sdxl_image_studio.ui.components.lora_editor import (
    LoraEditorComponents,
    add_lora_row,
    component_output_count_for_rows,
    lora_settings_from_state,
    normalize_lora_state,
    preserve_component_updates,
    render_state_updates,
    update_lora_row,
)


@dataclass(frozen=True)
class PresetTabComponents:
    """Preset操作に必要なGradioコンポーネント。"""

    search: gr.Textbox
    kind: gr.Dropdown
    favorite_only: gr.Checkbox
    results: gr.Dropdown
    message: gr.Markdown
    refresh: gr.Button
    preset_kind: gr.Dropdown
    name: gr.Textbox
    description: gr.Textbox
    favorite: gr.Checkbox
    selected: gr.Dropdown
    payload_summary: gr.Markdown
    prompt_apply_mode: gr.Dropdown
    lora_apply_mode: gr.Dropdown
    save_button: gr.Button
    update_button: gr.Button
    duplicate_button: gr.Button
    delete_confirmation: gr.Checkbox
    delete_button: gr.Button
    apply_button: gr.Button
    clear_button: gr.Button
    recent_checkpoints: gr.Dropdown
    recent_checkpoint_apply: gr.Button
    recent_vaes: gr.Dropdown
    recent_vae_apply: gr.Button
    recent_loras: gr.Dropdown
    recent_lora_add: gr.Button
    recent_generation_presets: gr.Dropdown
    recent_prompt_presets: gr.Dropdown
    recent_lora_presets: gr.Dropdown
    recent_refresh: gr.Button


def build_preset_tab() -> PresetTabComponents:
    """Generation画面へ接続したPreset管理UIを構築する。"""

    gr.Markdown("## プリセット")
    with gr.Row(elem_classes=["preset-actions"]):
        search = gr.Textbox(label="Preset検索")
        kind = gr.Dropdown(
            [("すべて", ""), *((item.value, item.value) for item in PresetKind)],
            value="",
            label="検索種類",
        )
        favorite_only = gr.Checkbox(label="お気に入りのみ")
        refresh = gr.Button("検索")
    results = gr.Dropdown([], label="Preset一覧", allow_custom_value=False)
    with gr.Row(elem_classes=["preset-actions"]):
        selected = gr.Dropdown([], label="選択中Preset", allow_custom_value=False)
        clear_button = gr.Button("条件・選択をクリア")
    with gr.Row(elem_classes=["preset-actions"]):
        preset_kind = gr.Dropdown(
            [(item.value, item.value) for item in PresetKind],
            value=PresetKind.GENERATION.value,
            label="Preset種類",
        )
        name = gr.Textbox(label="Preset名", max_length=100)
        favorite = gr.Checkbox(label="お気に入り")
    description = gr.Textbox(label="説明", lines=2, max_lines=4, max_length=1000)
    payload_summary = gr.Markdown("Payload未選択")
    with gr.Row(elem_classes=["preset-actions"]):
        prompt_apply_mode = gr.Dropdown(
            [
                ("置換", PromptApplyMode.REPLACE.value),
                ("先頭へ追加", PromptApplyMode.PREPEND.value),
                ("末尾へ追加", PromptApplyMode.APPEND.value),
            ],
            value=PromptApplyMode.REPLACE.value,
            label="Prompt適用方式",
        )
        lora_apply_mode = gr.Dropdown(
            [("置換", "replace"), ("末尾へ追加", "append")],
            value="replace",
            label="LoRA適用方式",
        )
    with gr.Row(elem_classes=["preset-actions"]):
        save_button = gr.Button("現在設定から保存", variant="primary")
        update_button = gr.Button("更新")
        duplicate_button = gr.Button("複製")
        apply_button = gr.Button("生成画面へ適用", variant="primary")
    with gr.Row(elem_classes=["preset-actions"]):
        delete_confirmation = gr.Checkbox(label="削除を確認しました")
        delete_button = gr.Button("削除")
    with gr.Accordion("最近使った設定", open=False):
        recent_refresh = gr.Button("最近使った設定を更新")
        recent_checkpoints = gr.Dropdown([], label="最近使ったcheckpoint")
        recent_checkpoint_apply = gr.Button("checkpointを生成画面へ反映")
        recent_vaes = gr.Dropdown([], label="最近使ったVAE")
        recent_vae_apply = gr.Button("VAEを生成画面へ反映")
        recent_loras = gr.Dropdown([], label="最近使ったLoRA")
        recent_lora_add = gr.Button("LoRAへ追加")
        recent_generation_presets = gr.Dropdown([], label="最近使ったGeneration Preset")
        recent_prompt_presets = gr.Dropdown([], label="最近使ったPrompt Preset")
        recent_lora_presets = gr.Dropdown([], label="最近使ったLoRA Preset")
    message = gr.Markdown("Presetを検索または保存してください。")
    return PresetTabComponents(
        search,
        kind,
        favorite_only,
        results,
        message,
        refresh,
        preset_kind,
        name,
        description,
        favorite,
        selected,
        payload_summary,
        prompt_apply_mode,
        lora_apply_mode,
        save_button,
        update_button,
        duplicate_button,
        delete_confirmation,
        delete_button,
        apply_button,
        clear_button,
        recent_checkpoints,
        recent_checkpoint_apply,
        recent_vaes,
        recent_vae_apply,
        recent_loras,
        recent_lora_add,
        recent_generation_presets,
        recent_prompt_presets,
        recent_lora_presets,
        recent_refresh,
    )


def make_preset_search_handler(
    service: PresetService,
) -> Callable[[str | None, str | None, bool], tuple[object, str]]:
    """低レベルRepositoryを露出させずPreset一覧を返す。"""

    def handler(text: str | None, kind: str | None, favorite_only: bool) -> tuple[object, str]:
        try:
            selected_kind = PresetKind(kind) if kind else None
            presets = service.search(text, kind=selected_kind, favorite_only=favorite_only)
            choices = _preset_choices(presets)
            return _dropdown(choices, choices[0][1] if choices else None), f"{len(choices)}件"
        except (PresetServiceError, ValueError) as exc:
            return gr.skip(), str(exc)

    return handler


def make_preset_select_handler(
    service: PresetService,
) -> Callable[[str | None], tuple[object, ...]]:
    """選択中Presetの編集値とPayload概要を表示する。"""

    def handler(selected: str | None) -> tuple[object, ...]:
        if not selected:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "Payload未選択",
                gr.skip(),
                gr.skip(),
                "",
            )
        try:
            preset = service.get(_parse_id(selected))
            return _selection_outputs(preset, "Presetを選択しました。")
        except (PresetServiceError, ValueError) as exc:
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                str(exc),
            )

    return handler


def make_preset_save_handler(
    service: PresetService,
    max_loras: int,
) -> Callable[..., tuple[object, ...]]:
    """現在の生成フォーム値からPresetを作成し、一覧を再描画する。"""

    def handler(
        kind: str,
        name: str,
        description: str,
        favorite: bool,
        positive: str,
        negative: str,
        width: float | int,
        height: float | int,
        seed_mode: str,
        seed: float | int,
        steps: float | int,
        cfg: float,
        sampler: str | None,
        scheduler: str | None,
        checkpoint: str | None,
        vae: str | None,
        lora_state: object,
        positive_mode: str,
        negative_mode: str,
        search: str | None,
        search_kind: str | None,
        favorite_only: bool,
    ) -> tuple[object, ...]:
        try:
            settings = _settings_from_form(
                positive,
                negative,
                width,
                height,
                seed_mode,
                seed,
                steps,
                cfg,
                sampler,
                scheduler,
                checkpoint,
                vae,
                lora_state,
                max_loras,
            )
            preset_kind = PresetKind(kind)
            if preset_kind is PresetKind.GENERATION:
                created = service.create_from_current_settings(
                    name,
                    settings,
                    description=description,
                    favorite=favorite,
                    seed_mode=_seed_mode(seed_mode),
                )
            elif preset_kind is PresetKind.PROMPT:
                created = service.create_prompt_preset(
                    name,
                    positive,
                    negative,
                    description=description,
                    favorite=favorite,
                    positive_mode=PromptApplyMode(positive_mode),
                    negative_mode=PromptApplyMode(negative_mode),
                )
            else:
                created = service.create_lora_preset(
                    name, settings.loras, description=description, favorite=favorite
                )
            return _management_outputs(
                service, created, search, search_kind, favorite_only, "Presetを保存しました。"
            )
        except (PresetServiceError, ValueError) as exc:
            return _preserve_management(str(exc))

    return handler


def make_preset_update_handler(
    service: PresetService,
    max_loras: int,
) -> Callable[..., tuple[object, ...]]:
    """選択中IDを明示してフォーム値を更新する。"""

    def handler(
        selected: str | None,
        name: str,
        description: str,
        favorite: bool,
        positive: str,
        negative: str,
        width: float | int,
        height: float | int,
        seed_mode: str,
        seed: float | int,
        steps: float | int,
        cfg: float,
        sampler: str | None,
        scheduler: str | None,
        checkpoint: str | None,
        vae: str | None,
        lora_state: object,
        positive_mode: str,
        negative_mode: str,
        search: str | None,
        search_kind: str | None,
        favorite_only: bool,
    ) -> tuple[object, ...]:
        if not selected:
            return _preserve_management("更新対象のPresetを選択してください。")
        try:
            settings = _settings_from_form(
                positive,
                negative,
                width,
                height,
                seed_mode,
                seed,
                steps,
                cfg,
                sampler,
                scheduler,
                checkpoint,
                vae,
                lora_state,
                max_loras,
            )
            updated = service.update_from_current_settings(
                _parse_id(selected),
                settings,
                name=name,
                description=description,
                favorite=favorite,
                seed_mode=_seed_mode(seed_mode),
                positive_mode=PromptApplyMode(positive_mode),
                negative_mode=PromptApplyMode(negative_mode),
            )
            return _management_outputs(
                service, updated, search, search_kind, favorite_only, "Presetを更新しました。"
            )
        except (PresetServiceError, ValueError) as exc:
            return _preserve_management(str(exc))

    return handler


def make_preset_duplicate_handler(
    service: PresetService,
) -> Callable[[str | None, str | None, str | None, bool], tuple[object, ...]]:
    """選択Presetを自動生成した別名で複製する。"""

    def handler(
        selected: str | None, search: str | None, search_kind: str | None, favorite_only: bool
    ) -> tuple[object, ...]:
        if not selected:
            return _preserve_management("複製対象のPresetを選択してください。")
        try:
            duplicate = service.duplicate(_parse_id(selected))
            return _management_outputs(
                service, duplicate, search, search_kind, favorite_only, "Presetを複製しました。"
            )
        except (PresetServiceError, ValueError) as exc:
            return _preserve_management(str(exc))

    return handler


def make_preset_delete_handler(
    service: PresetService,
) -> Callable[[str | None, bool, str | None, str | None, bool], tuple[object, ...]]:
    """確認済みのPresetだけを削除し、選択と編集値を解除する。"""

    def handler(
        selected: str | None,
        confirmed: bool,
        search: str | None,
        search_kind: str | None,
        favorite_only: bool,
    ) -> tuple[object, ...]:
        if not selected:
            return _preserve_management("削除対象のPresetを選択してください。")
        if not confirmed:
            return _preserve_management("削除確認にチェックを入れてください。")
        try:
            service.delete(_parse_id(selected))
            choices = _search_choices(service, search, search_kind, favorite_only)
            return (
                _dropdown(choices),
                gr.Dropdown(value=None),
                gr.Dropdown(value=PresetKind.GENERATION.value),
                gr.Textbox(value=""),
                gr.Textbox(value=""),
                gr.Checkbox(value=False),
                "Payload未選択",
                gr.Dropdown(value=PromptApplyMode.REPLACE.value),
                gr.Dropdown(value="replace"),
                "Presetを削除しました。",
            )
        except (PresetServiceError, ValueError) as exc:
            return _preserve_management(str(exc))

    return handler


def make_preset_favorite_handler(
    service: PresetService,
) -> Callable[[str | None, bool], tuple[object, str]]:
    """お気に入り変更を選択中Presetへ明示的に保存する。"""

    def handler(selected: str | None, favorite: bool) -> tuple[object, str]:
        if not selected:
            return gr.skip(), "お気に入り変更対象を選択してください。"
        try:
            service.set_favorite(_parse_id(selected), favorite)
            return favorite, "お気に入りを保存しました。"
        except (PresetServiceError, ValueError) as exc:
            return gr.skip(), str(exc)

    return handler


def make_preset_clear_handler() -> Callable[[], tuple[object, ...]]:
    """検索条件と選択中Presetを初期化する。"""

    def handler() -> tuple[object, ...]:
        return (
            gr.Textbox(value=""),
            gr.Dropdown(value=""),
            gr.Checkbox(value=False),
            gr.Dropdown(choices=[], value=None),
            gr.Dropdown(value=None),
            gr.Dropdown(value=PresetKind.GENERATION.value),
            gr.Textbox(value=""),
            gr.Textbox(value=""),
            gr.Checkbox(value=False),
            "Payload未選択",
            gr.Dropdown(value=PromptApplyMode.REPLACE.value),
            gr.Dropdown(value="replace"),
            "条件と選択をクリアしました。",
        )

    return handler


def make_preset_apply_handler(
    service: PresetService,
    max_loras: int,
    lora_editor: LoraEditorComponents | None = None,
) -> Callable[..., tuple[object, ...]]:
    """Presetを生成フォームへ反映する。ここでは生成処理を呼び出さない。"""

    def handler(
        selected: str | None,
        prompt_mode: str,
        lora_mode: str,
        positive: str,
        negative: str,
        width: float | int,
        height: float | int,
        seed_mode: str,
        seed: float | int,
        steps: float | int,
        cfg: float,
        sampler: str | None,
        scheduler: str | None,
        checkpoint: str | None,
        vae: str | None,
        lora_state: object,
        checkpoint_choices: object,
        vae_choices: object,
        lora_choices: object,
    ) -> tuple[object, ...]:
        if not selected:
            return _preserve_apply("適用するPresetを選択してください。", max_loras, lora_editor)
        try:
            current = _settings_from_form(
                positive,
                negative,
                width,
                height,
                seed_mode,
                seed,
                steps,
                cfg,
                sampler,
                scheduler,
                checkpoint,
                vae,
                lora_state,
                max_loras,
            )
            result = service.apply(
                _parse_id(selected),
                current_settings=current,
                available_checkpoints=_string_choices(checkpoint_choices),
                available_vaes=_string_choices(vae_choices),
                available_loras=_string_choices(lora_choices),
                max_loras=max_loras,
                prompt_mode=PromptApplyMode(prompt_mode),
                lora_mode=lora_mode,
            )
            return _apply_outputs(
                result,
                checkpoint_choices,
                vae_choices,
                lora_choices,
                max_loras,
            )
        except (PresetServiceError, ValueError) as exc:
            return _preserve_apply(str(exc), max_loras, lora_editor)

    return handler


def make_recent_settings_handler(
    service: RecentSettingsService,
    preset_service: PresetService,
) -> Callable[..., tuple[object, ...]]:
    """設定上限付きの最近値をDBから取得し、missingを表示する。"""

    def handler(
        checkpoint_choices: object, vae_choices: object, lora_choices: object
    ) -> tuple[object, ...]:
        try:
            recent = service.get_recent()
            checkpoint_values = _recent_choices(
                recent.checkpoints, _string_choices(checkpoint_choices)
            )
            vae_values = _recent_choices(recent.vaes, _string_choices(vae_choices))
            lora_values = _recent_choices(recent.loras, _string_choices(lora_choices))
            return (
                _recent_dropdown(checkpoint_values),
                _recent_dropdown(vae_values),
                _recent_dropdown(lora_values),
                _recent_dropdown(
                    _preset_id_choices(
                        preset_service, recent.generation_presets, PresetKind.GENERATION
                    )
                ),
                _recent_dropdown(
                    _preset_id_choices(preset_service, recent.prompt_presets, PresetKind.PROMPT)
                ),
                _recent_dropdown(
                    _preset_id_choices(preset_service, recent.lora_presets, PresetKind.LORA)
                ),
                "最近使った設定を更新しました。",
            )
        except (RecentSettingsServiceError, PresetServiceError, ValueError):
            return (
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                gr.skip(),
                "最近使った設定を取得できませんでした。",
            )

    return handler


def make_recent_checkpoint_handler() -> Callable[[str | None, object], tuple[object, str]]:
    """能力一覧に存在する最近のcheckpointだけを生成フォームへ反映する。"""

    def handler(selected: str | None, choices: object) -> tuple[object, str]:
        if not selected:
            return gr.skip(), "反映するcheckpointを選択してください。"
        available = _string_choices(choices)
        if available is None:
            return gr.skip(), "checkpoint一覧が未取得のため反映できません。"
        if selected not in available:
            return gr.skip(), f"checkpointが現在利用できません: {selected}"
        return gr.Dropdown(value=selected), "最近使ったcheckpointを反映しました。"

    return handler


def make_recent_vae_handler() -> Callable[[str | None, object], tuple[object, str]]:
    """能力一覧に存在する最近のVAEだけを生成フォームへ反映する。"""

    def handler(selected: str | None, choices: object) -> tuple[object, str]:
        if selected is None:
            return gr.skip(), "反映するVAEを選択してください。"
        available = _string_choices(choices)
        if available is None:
            return gr.skip(), "VAE一覧が未取得のため反映できません。"
        if selected not in available:
            return gr.skip(), f"VAEが現在利用できません: {selected}"
        return gr.Dropdown(value=selected), "最近使ったVAEを反映しました。"

    return handler


def make_recent_lora_add_handler(
    max_loras: int,
) -> Callable[[str | None, object, object], tuple[object, ...]]:
    """選択済みの最近LoRAを明示操作で末尾へ追加する。"""

    def handler(selected: str | None, state: object, choices: object) -> tuple[object, ...]:
        if not selected:
            return _preserve_recent_lora(
                state, choices, max_loras, "追加するLoRAを選択してください。"
            )
        available = _string_choices(choices)
        if available is None:
            return _preserve_recent_lora(
                state, choices, max_loras, "LoRA一覧が未取得のため追加できません。"
            )
        if selected not in available:
            return _preserve_recent_lora(
                state, choices, max_loras, f"LoRAが現在利用できません: {selected}"
            )
        try:
            current = lora_settings_from_state(state, max_loras)
            if selected in {item.name for item in current}:
                return _preserve_recent_lora(
                    state, choices, max_loras, "同じLoRAは重複追加できません。"
                )
            if len(current) >= max_loras:
                return _preserve_recent_lora(
                    state, choices, max_loras, "LoRA数が上限を超えています。"
                )
            rows = normalize_lora_state(state, max_loras)
            empty_index = next(
                (index for index, row in enumerate(rows) if not row.get("lora_name")),
                None,
            )
            if empty_index is not None:
                row_index = empty_index
                updated = update_lora_row(state, row_index, selected, 1.0, 1.0, max_loras)
            else:
                expanded = add_lora_row(state, max_loras)
                updated = update_lora_row(
                    expanded, len(expanded) - 1, selected, 1.0, 1.0, max_loras
                )
            return render_state_updates(updated, choices, max_loras) + (
                "最近使ったLoRAを末尾へ追加しました。",
            )
        except ValueError:
            return _preserve_recent_lora(state, choices, max_loras, "LoRAを追加できませんでした。")

    return handler


def seed_copy_value(seed: int) -> str:
    """履歴snapshotの実使用seedを整数文字列として表示する。"""

    return str(int(seed))


def preset_id(value: str | None) -> UUID | None:
    """選択値を安全にUUIDへ変換する。"""

    if not value:
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _settings_from_form(
    positive: str,
    negative: str,
    width: float | int,
    height: float | int,
    seed_mode: str,
    seed: float | int,
    steps: float | int,
    cfg: float,
    sampler: str | None,
    scheduler: str | None,
    checkpoint: str | None,
    vae: str | None,
    lora_state: object,
    max_loras: int,
) -> GenerationSettings:
    return GenerationSettings(
        positive_prompt=positive or "",
        negative_prompt=negative or "",
        width=int(width),
        height=int(height),
        seed=-1 if seed_mode == "Random" or seed_mode == SeedMode.RANDOM.value else int(seed),
        steps=int(steps),
        cfg_scale=float(cfg),
        sampler_name=sampler or "",
        scheduler_name=scheduler or "",
        checkpoint_name=checkpoint or "",
        vae_name=vae,
        loras=lora_settings_from_state(lora_state, max_loras),
    )


def _seed_mode(value: str) -> SeedMode:
    translated = {
        "Random": SeedMode.RANDOM,
        "Fixed": SeedMode.FIXED,
        "Previous seed": SeedMode.PREVIOUS,
    }
    return translated[value] if value in translated else SeedMode(value)


def _apply_outputs(
    result: PresetApplyResult,
    checkpoint_choices: object,
    vae_choices: object,
    lora_choices: object,
    max_loras: int,
) -> tuple[object, ...]:
    settings = result.settings
    state = [
        {
            "row_id": f"preset-{index}",
            "lora_name": lora.name,
            "model_strength": lora.model_strength,
            "clip_strength": lora.clip_strength,
        }
        for index, lora in enumerate(settings.loras)
    ] or [{"row_id": "preset-0", "lora_name": None, "model_strength": 1.0, "clip_strength": 1.0}]
    lora_updates = render_state_updates(state, lora_choices, max_loras, clear_unavailable=False)
    mode = (
        result.seed_mode.value
        if result.seed_mode is not None
        else (SeedMode.FIXED.value if settings.seed >= 0 else SeedMode.RANDOM.value)
    )
    ui_mode = {
        SeedMode.RANDOM.value: "Random",
        SeedMode.FIXED.value: "Fixed",
        SeedMode.PREVIOUS.value: "Previous seed",
    }[mode]
    warning = " / ".join(result.warnings)
    return (
        "Presetを適用しました。" + (f" 警告: {warning}" if warning else ""),
        _model_update(checkpoint_choices, settings.checkpoint_name),
        _model_update(vae_choices, settings.vae_name, include_embedded_vae=True),
        settings.positive_prompt,
        settings.negative_prompt,
        settings.width,
        settings.height,
        ui_mode,
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
        gr.skip(),
        gr.skip(),
    )


def _preserve_apply(
    message: str,
    max_loras: int,
    lora_editor: LoraEditorComponents | None = None,
) -> tuple[object, ...]:
    """Preset適用の全出力を、正常系と同じ構造で保持する。"""

    if lora_editor is not None:
        lora_outputs = (
            gr.skip(),
            *preserve_component_updates(lora_editor),
            gr.skip(),
        )
    else:
        lora_outputs = tuple(
            gr.skip() for _ in range(2 + component_output_count_for_rows(max_loras))
        )
    return (
        message,
        *(gr.skip() for _ in range(23)),
        *lora_outputs,
        gr.skip(),
        gr.skip(),
    )


def _preserve_recent_lora(
    state: object,
    choices: object,
    max_loras: int,
    message: str,
) -> tuple[object, ...]:
    """最近LoRA追加の失敗時にStateと各行を変更しない。"""

    del state, choices
    return (
        *(gr.skip() for _ in range(2 + component_output_count_for_rows(max_loras))),
        message,
    )


def preset_apply_output_count(max_loras: int) -> int:
    """Preset適用イベントの正常系・異常系で共有する出力数。"""

    lora_output_count = 2 + component_output_count_for_rows(max_loras)
    return 1 + 23 + lora_output_count + 2


def _management_outputs(
    service: PresetService,
    preset: Preset,
    search: str | None,
    search_kind: str | None,
    favorite_only: bool,
    message: str,
) -> tuple[object, ...]:
    choices = _search_choices(service, search, search_kind, favorite_only)
    selection = _selection_outputs(preset, message)
    return (
        _dropdown(choices, str(preset.id)),
        gr.Dropdown(value=str(preset.id)),
        *selection[:-1],
        selection[-1],
    )


def _preserve_management(message: str) -> tuple[object, ...]:
    return (gr.skip(),) * 9 + (message,)


def _selection_outputs(preset: Preset, message: str) -> tuple[object, ...]:
    positive_mode = PromptApplyMode.REPLACE.value
    if hasattr(preset.payload, "positive_mode"):
        positive_mode = preset.payload.positive_mode.value
    return (
        gr.Dropdown(value=preset.kind.value),
        gr.Textbox(value=preset.name),
        gr.Textbox(value=preset.description or ""),
        gr.Checkbox(value=preset.favorite),
        _payload_summary(preset),
        gr.Dropdown(value=positive_mode),
        gr.Dropdown(value="replace"),
        message,
    )


def _payload_summary(preset: Preset) -> str:
    payload = preset.payload.model_dump(mode="json")
    safe = html.escape(str(payload))
    return f"**{html.escape(preset.kind.value)} payload**\n\n`{safe}`"


def _preset_choices(presets: Sequence[Preset]) -> list[tuple[str, str]]:
    return [(f"{preset.name} ({preset.kind.value})", str(preset.id)) for preset in presets]


def _search_choices(
    service: PresetService, text: str | None, kind: str | None, favorite: bool
) -> list[tuple[str, str]]:
    selected_kind = PresetKind(kind) if kind else None
    return _preset_choices(service.search(text, kind=selected_kind, favorite_only=favorite))


def _dropdown(choices: Sequence[tuple[str, str]], value: str | None = None) -> gr.Dropdown:
    return gr.Dropdown(choices=list(choices), value=value)


def _recent_dropdown(choices: Sequence[tuple[str, str]]) -> gr.Dropdown:
    """最近設定のchoicesを選択値なしのDropdown更新へ変換する。"""

    return gr.Dropdown(choices=list(choices), value=None)


def _parse_id(value: str) -> UUID:
    parsed = preset_id(value)
    if parsed is None:
        raise ValueError("Preset IDが不正です。")
    return parsed


def _string_choices(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], str):
            result.append(item[1])
    return tuple(result)


def _recent_choices(
    values: Sequence[str], available: tuple[str, ...] | None
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for value in values:
        missing = available is not None and value not in available
        result.append((f"{value}（現在利用不可）" if missing else value, value))
    return result


def _model_update(
    choices: object,
    value: str | None,
    *,
    include_embedded_vae: bool = False,
) -> gr.Dropdown:
    """現在選択中のmissingモデルも消さずに画面へ残す。"""

    values = _string_choices(choices) or ()
    options: list[tuple[str, str]] = [("Checkpoint内蔵VAE", "")] if include_embedded_vae else []
    options.extend((item, item) for item in values)
    if value and value not in values:
        options.append((f"{value}（現在利用不可）", value))
    return gr.Dropdown(choices=options, value=value)


def _preset_id_choices(
    service: PresetService, ids: Sequence[UUID], kind: PresetKind
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for preset_id_value in ids:
        try:
            preset = service.get(preset_id_value)
            if preset.kind is kind:
                result.append((preset.name, str(preset.id)))
        except (PresetServiceError, ValueError):
            continue
    return result


__all__ = [
    "PresetTabComponents",
    "build_preset_tab",
    "make_preset_apply_handler",
    "make_preset_clear_handler",
    "make_preset_delete_handler",
    "make_preset_duplicate_handler",
    "make_preset_favorite_handler",
    "make_preset_save_handler",
    "make_preset_search_handler",
    "make_preset_select_handler",
    "make_preset_update_handler",
    "make_recent_settings_handler",
    "preset_apply_output_count",
    "preset_id",
    "seed_copy_value",
]
