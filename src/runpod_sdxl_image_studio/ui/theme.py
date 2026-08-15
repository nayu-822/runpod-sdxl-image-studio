"""RunPod SDXL Image Studioの標準モダンダークテーマ。"""

from __future__ import annotations

import gradio as gr


def modern_dark_theme() -> gr.themes.Base:
    """Return the app-wide dark theme with stable visual tokens."""

    return gr.themes.Base().set(
        body_background_fill="#0B0D12",
        body_background_fill_dark="#0B0D12",
        body_text_color="#F5F7FB",
        body_text_color_dark="#F5F7FB",
        body_text_color_subdued="#9AA4B2",
        body_text_color_subdued_dark="#9AA4B2",
        background_fill_primary="#12151C",
        background_fill_primary_dark="#12151C",
        background_fill_secondary="#181C24",
        background_fill_secondary_dark="#181C24",
        block_background_fill="#12151C",
        block_background_fill_dark="#12151C",
        block_border_color="rgba(255,255,255,0.08)",
        block_border_color_dark="rgba(255,255,255,0.08)",
        block_radius="16px",
        container_radius="16px",
        input_background_fill="#181C24",
        input_background_fill_dark="#181C24",
        input_background_fill_focus="#202633",
        input_background_fill_focus_dark="#202633",
        input_border_color="rgba(255,255,255,0.08)",
        input_border_color_dark="rgba(255,255,255,0.08)",
        input_border_color_focus="#7C5CFF",
        input_border_color_focus_dark="#7C5CFF",
        border_color_accent="#7C5CFF",
        border_color_accent_dark="#7C5CFF",
        color_accent="#7C5CFF",
        color_accent_soft="#202633",
        color_accent_soft_dark="#202633",
        button_primary_background_fill="#7C5CFF",
        button_primary_background_fill_dark="#7C5CFF",
        button_primary_background_fill_hover="#8B73FF",
        button_primary_background_fill_hover_dark="#8B73FF",
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_dark="#FFFFFF",
        button_large_radius="15px",
        button_medium_radius="12px",
        button_small_radius="10px",
        button_large_padding="0.75rem 1rem",
        button_medium_padding="0.6rem 0.85rem",
        button_small_padding="0.45rem 0.7rem",
        block_title_text_color="#F5F7FB",
        block_title_text_color_dark="#F5F7FB",
        accordion_text_color="#F5F7FB",
        accordion_text_color_dark="#F5F7FB",
        panel_background_fill="#12151C",
        panel_background_fill_dark="#12151C",
        panel_border_color="rgba(255,255,255,0.08)",
        panel_border_color_dark="rgba(255,255,255,0.08)",
    )


__all__ = ["modern_dark_theme"]
