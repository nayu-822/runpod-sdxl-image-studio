"""Application service for generation and prompt differences."""

from __future__ import annotations

from difflib import SequenceMatcher
from uuid import UUID

from runpod_sdxl_image_studio.domain.generation import Generation
from runpod_sdxl_image_studio.domain.generation_diff import (
    ChangeType,
    GenerationDiff,
    LoraChange,
    PromptTokenChange,
    ValueChange,
)
from runpod_sdxl_image_studio.domain.generation_snapshot import GenerationSettingsSnapshot

MAX_DIFF_TEXT_LENGTH = 20_000


class GenerationDiffError(ValueError):
    """差分入力が安全な範囲を超えている。"""


class GenerationDiffService:
    """snapshotを変更せず、tag単位の差分を生成する。"""

    def compare(self, source: Generation, target: Generation) -> GenerationDiff:
        return self.compare_snapshots(
            source.id, source.settings_snapshot, target.id, target.settings_snapshot
        )

    def compare_snapshots(
        self,
        source_generation_id: UUID,
        source: GenerationSettingsSnapshot,
        target_generation_id: UUID,
        target: GenerationSettingsSnapshot,
    ) -> GenerationDiff:
        prompt_values = (
            source.positive_prompt,
            source.negative_prompt,
            target.positive_prompt,
            target.negative_prompt,
        )
        if any(len(value) > MAX_DIFF_TEXT_LENGTH for value in prompt_values):
            raise GenerationDiffError("prompt is too long to compare")
        return GenerationDiff(
            source_generation_id=source_generation_id,
            target_generation_id=target_generation_id,
            positive_prompt_changes=_token_diff(source.positive_prompt, target.positive_prompt),
            negative_prompt_changes=_token_diff(source.negative_prompt, target.negative_prompt),
            setting_changes=_setting_diff(source, target),
            lora_changes=_lora_diff(source, target),
        )


def _tokens(prompt: str) -> list[str]:
    return [token.strip() for token in prompt.split(",") if token.strip()]


def _token_diff(before: str, after: str) -> tuple[PromptTokenChange, ...]:
    left, right = _tokens(before), _tokens(after)
    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    changes: list[PromptTokenChange] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            changes.extend(
                PromptTokenChange(
                    value=left[i],
                    change_type=ChangeType.UNCHANGED,
                    before_index=i,
                    after_index=j1 + i - i1,
                )
                for i in range(i1, i2)
            )
        elif tag == "delete":
            changes.extend(
                PromptTokenChange(
                    value=left[i], change_type=ChangeType.REMOVED, before_index=i, after_index=None
                )
                for i in range(i1, i2)
            )
        elif tag == "insert":
            changes.extend(
                PromptTokenChange(
                    value=right[j], change_type=ChangeType.ADDED, before_index=None, after_index=j
                )
                for j in range(j1, j2)
            )
        else:
            changes.extend(
                PromptTokenChange(
                    value=left[i], change_type=ChangeType.REMOVED, before_index=i, after_index=None
                )
                for i in range(i1, i2)
            )
            changes.extend(
                PromptTokenChange(
                    value=right[j], change_type=ChangeType.ADDED, before_index=None, after_index=j
                )
                for j in range(j1, j2)
            )
    if sorted(left) == sorted(right) and left != right:
        return tuple(
            PromptTokenChange(
                value=value,
                change_type=ChangeType.REORDERED,
                before_index=left.index(value),
                after_index=right.index(value),
            )
            for value in right
        )
    return tuple(changes)


def _setting_diff(
    before: GenerationSettingsSnapshot, after: GenerationSettingsSnapshot
) -> tuple[ValueChange, ...]:
    fields = (
        "checkpoint_name",
        "vae_name",
        "seed",
        "width",
        "height",
        "steps",
        "cfg_scale",
        "sampler_name",
        "scheduler_name",
    )
    return tuple(
        ValueChange(field, getattr(before, field), getattr(after, field), ChangeType.CHANGED)
        for field in fields
        if getattr(before, field) != getattr(after, field)
    )


def _lora_diff(
    before: GenerationSettingsSnapshot, after: GenerationSettingsSnapshot
) -> tuple[LoraChange, ...]:
    left = {item.name: item for item in before.loras}
    right = {item.name: item for item in after.loras}
    changes: list[LoraChange] = []
    for name in dict.fromkeys((*left.keys(), *right.keys())):
        old, new = left.get(name), right.get(name)
        if old is None and new is not None:
            changes.append(
                LoraChange(
                    name,
                    ChangeType.ADDED,
                    after_order=new.order,
                    after_model_strength=new.model_strength,
                    after_clip_strength=new.clip_strength,
                )
            )
        elif new is None and old is not None:
            changes.append(
                LoraChange(
                    name,
                    ChangeType.REMOVED,
                    before_order=old.order,
                    before_model_strength=old.model_strength,
                    before_clip_strength=old.clip_strength,
                )
            )
        elif old is not None and new is not None:
            change_type = ChangeType.UNCHANGED
            if (
                old.order != new.order
                and old.model_strength == new.model_strength
                and old.clip_strength == new.clip_strength
            ):
                change_type = ChangeType.REORDERED
            elif (old.order, old.model_strength, old.clip_strength) != (
                new.order,
                new.model_strength,
                new.clip_strength,
            ):
                change_type = ChangeType.CHANGED
            if change_type is not ChangeType.UNCHANGED:
                changes.append(
                    LoraChange(
                        name,
                        change_type,
                        old.order,
                        new.order,
                        old.model_strength,
                        new.model_strength,
                        old.clip_strength,
                        new.clip_strength,
                    )
                )
    return tuple(changes)
