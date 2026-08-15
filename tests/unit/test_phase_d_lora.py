from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import gradio as gr
import pytest

from runpod_sdxl_image_studio.domain.generation_form_state import (
    FormSeedMode,
    GenerationFormStateSnapshot,
)
from runpod_sdxl_image_studio.domain.lora import LoraSetting
from runpod_sdxl_image_studio.services.lora_trigger_service import (
    LoraTriggerResolutionError,
    resolve_effective_positive_prompt,
)
from runpod_sdxl_image_studio.ui.components.lora_editor import (
    auto_trigger_lora_names,
    lora_settings_from_state,
    normalize_lora_state,
)
from runpod_sdxl_image_studio.ui.tabs.lora_management_tab import (
    make_add_to_generation_handler,
)


def test_old_lora_editor_state_defaults_auto_trigger_to_false() -> None:
    state = normalize_lora_state(
        [{"row_id": "old", "lora_name": "style.safetensors"}],
        2,
    )

    assert state[0]["auto_add_trigger_words"] is False
    assert auto_trigger_lora_names(state, 2) == ()
    assert lora_settings_from_state(state, 2)[0].auto_add_trigger_words is False


def test_trigger_resolution_is_exact_ordered_deduplicated_and_negative_safe() -> None:
    calls: list[str] = []

    class Catalog:
        def get_by_file_name(self, file_name: str) -> object:
            calls.append(file_name)
            return SimpleNamespace(
                file_name=file_name,
                is_missing=False,
                trigger_words=("hero", "blue eyes"),
            )

    loras = (
        LoraSetting(name="first.safetensors", order=1, auto_add_trigger_words=True),
        LoraSetting(name="second.safetensors", order=0, auto_add_trigger_words=True),
        LoraSetting(name="off.safetensors", order=2, auto_add_trigger_words=False),
    )

    effective = resolve_effective_positive_prompt("1girl, hero", loras, Catalog())

    assert effective == "1girl, hero, blue eyes"
    assert calls == ["second.safetensors", "first.safetensors"]


def test_trigger_resolution_does_not_lookup_when_disabled_and_fails_closed_when_missing() -> None:
    class Catalog:
        def get_by_file_name(self, file_name: str) -> None:
            raise AssertionError(f"unexpected lookup: {file_name}")

    assert (
        resolve_effective_positive_prompt(
            "positive",
            (LoraSetting(name="style.safetensors", auto_add_trigger_words=False),),
            Catalog(),
        )
        == "positive"
    )

    class MissingCatalog:
        def get_by_file_name(self, file_name: str) -> object:
            return None

    with pytest.raises(LoraTriggerResolutionError):
        resolve_effective_positive_prompt(
            "positive",
            (LoraSetting(name="missing.safetensors", auto_add_trigger_words=True),),
            MissingCatalog(),
        )


def test_form_state_round_trip_preserves_auto_trigger_names_and_old_payload_defaults() -> None:
    snapshot = GenerationFormStateSnapshot.from_ui(
        positive_prompt="positive, trigger",
        negative_prompt="negative",
        seed_mode=FormSeedMode.FIXED,
        seed=42,
        width=1024,
        height=1024,
        steps=28,
        cfg_scale=5.5,
        sampler_name="euler",
        scheduler_name="normal",
        checkpoint_name="checkpoint.safetensors",
        vae_name=None,
        loras=(LoraSetting(name="style.safetensors", auto_add_trigger_words=True),),
        auto_trigger_lora_names=("style.safetensors",),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    restored = GenerationFormStateSnapshot.from_json(snapshot.to_json())
    old_payload = snapshot.model_dump(mode="json")
    old_payload.pop("auto_trigger_lora_names")

    assert restored.auto_trigger_lora_names == ("style.safetensors",)
    assert GenerationFormStateSnapshot.model_validate(old_payload).auto_trigger_lora_names == ()


def test_lora_card_add_applies_recommended_strength_without_duplicate_or_unavailable() -> None:
    selected_id = UUID(int=1)
    metadata = SimpleNamespace(
        id=selected_id,
        file_name="style.safetensors",
        is_missing=False,
        recommended_model_strength=0.7,
        recommended_clip_strength=0.6,
    )

    class Catalog:
        def get_metadata(self, metadata_id: UUID) -> object:
            assert metadata_id == selected_id
            return metadata

    handler = make_add_to_generation_handler(Catalog(), 2)  # type: ignore[arg-type]
    state = [{"row_id": "empty", "lora_name": None}]
    with gr.Blocks():
        result = handler(str(selected_id), state, ["style.safetensors"])

    assert result[0] == "生成に追加しました。"
    assert result[1][0]["lora_name"] == "style.safetensors"
    assert result[1][0]["model_strength"] == 0.7
    assert result[1][0]["clip_strength"] == 0.6
    assert result[1][0]["auto_add_trigger_words"] is False

    duplicate = handler(str(selected_id), result[1], ["style.safetensors"])
    assert "すでに" in duplicate[0]

    unavailable = SimpleNamespace(
        id=UUID(int=2),
        file_name="missing.safetensors",
        is_missing=True,
        recommended_model_strength=None,
        recommended_clip_strength=None,
    )

    class MissingCatalog:
        def get_metadata(self, metadata_id: UUID) -> object:
            return unavailable

    with gr.Blocks():
        blocked = make_add_to_generation_handler(MissingCatalog(), 2)(
            str(unavailable.id), state, ["missing.safetensors"]
        )
    assert "利用できない" in blocked[0]
