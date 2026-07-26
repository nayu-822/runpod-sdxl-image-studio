"""Public generation-tab module for the Phase 1B UI."""

from .system_tab import (
    GenerationTabComponents,
    build_generation_tab,
    make_generate_handler,
    size_preset_values,
)

__all__ = [
    "GenerationTabComponents",
    "build_generation_tab",
    "make_generate_handler",
    "size_preset_values",
]
