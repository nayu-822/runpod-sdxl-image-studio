"""生成状態カードのGradioコンポーネント。"""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr


@dataclass(frozen=True)
class GenerationStatusCardComponents:
    """状態pollと生成結果表示に共有するコンポーネント。"""

    card: gr.Markdown
    active_generation_id: gr.State
    poll_timer: gr.Timer


def build_generation_status_card() -> GenerationStatusCardComponents:
    """Build a bounded DB-backed status card and its polling state."""

    card = gr.Markdown(
        "### 生成ステータス\n**状態:** `待機中`",
        elem_classes=["generation-status-card"],
    )
    active_generation_id = gr.State(None)
    poll_timer = gr.Timer(value=5, active=True)
    return GenerationStatusCardComponents(card, active_generation_id, poll_timer)


__all__ = ["GenerationStatusCardComponents", "build_generation_status_card"]
