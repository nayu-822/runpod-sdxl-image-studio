"""Phase 8のレスポンシブUI用CSS。"""

from __future__ import annotations

MOBILE_UI_CSS = """
.gradio-container {
  box-sizing: border-box;
  max-width: 1200px !important;
  width: 100% !important;
  padding: 1rem clamp(0.75rem, 2vw, 1.5rem) !important;
}

.gradio-container *,
.gradio-container *::before,
.gradio-container *::after {
  box-sizing: border-box;
  min-width: 0;
}

.gradio-container .row,
.tab-nav,
[role="tablist"] {
  min-width: 0 !important;
  max-width: 100% !important;
}

.tab-nav,
[role="tablist"] {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 0.25rem !important;
}

.tab-nav button,
[role="tablist"] button {
  min-width: 0 !important;
  min-height: 44px !important;
  white-space: normal !important;
}

.generation-layout {
  align-items: start !important;
  gap: 1rem !important;
}

.system-health-layout {
  align-items: stretch !important;
  gap: 1rem !important;
}

.system-health-column,
.system-error-column,
.system-health-section,
.system-error-section {
  min-width: 0 !important;
  width: 100% !important;
}

.generation-primary,
.generation-preview,
.generation-advanced,
.generation-batch,
.prompt-editor,
.lora-card-row,
.lora-card {
  min-width: 0 !important;
  width: 100% !important;
}

.generation-preview {
  display: flex !important;
  flex-direction: column !important;
  gap: 0.75rem !important;
}

.model-preparation-selection,
.model-preparation-actions {
  min-width: 0 !important;
  width: 100% !important;
  align-items: stretch !important;
  gap: 0.5rem !important;
  flex-wrap: wrap !important;
}

.model-preparation-selection {
  display: grid !important;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 16rem), 1fr));
}

.model-preparation-actions {
  display: grid !important;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 9rem), 1fr));
}

.comparison-layout {
  display: grid !important;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.75rem !important;
  min-width: 0 !important;
}

.generation-status-card {
  border: 1px solid var(--border-color-primary, #d9e1ea);
  border-radius: 0.75rem;
  padding: 0.75rem;
}

.generation-sticky-action {
  position: sticky !important;
  bottom: 0;
  z-index: 20;
  width: 100% !important;
  padding: 0.5rem 0;
  background: var(--body-background-fill, #ffffff);
  padding-bottom: calc(0.5rem + env(safe-area-inset-bottom, 0px));
}

.generation-sticky-action button,
.mobile-tap-button button,
.lora-actions button,
.history-actions button,
.system-actions button,
.queue-actions button,
.drive-actions button,
.metadata-actions button,
.preset-actions button {
  min-height: 44px !important;
}

.generation-sticky-action button {
  width: 100% !important;
  font-size: 1.05rem !important;
}

.prompt-editor textarea {
  min-height: 7rem !important;
  resize: vertical !important;
}

.prompt-editor.negative textarea {
  min-height: 5rem !important;
}

.prompt-actions,
.model-preparation-actions,
.size-dimensions,
.seed-controls,
.result-actions,
.recent-actions,
.system-actions,
.lora-strengths,
.lora-actions,
.history-filter,
.history-actions,
.queue-actions,
.drive-actions,
.metadata-actions,
.preset-actions {
  align-items: stretch !important;
  gap: 0.5rem !important;
  flex-wrap: wrap !important;
}

.size-dimensions,
.seed-controls,
.result-actions,
.system-actions,
.lora-strengths,
.lora-actions,
.history-filter,
.history-actions,
.queue-actions,
.drive-actions,
.metadata-actions,
.preset-actions {
  display: grid !important;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 9rem), 1fr));
}

.prompt-actions {
  display: flex !important;
}

.prompt-actions button {
  flex: 1 1 10rem;
  min-width: 0 !important;
}

.lora-card {
  gap: 0.5rem !important;
  padding: 0.75rem !important;
  border: 1px solid var(--border-color-primary, #d9e1ea);
  border-radius: 0.75rem;
  background: var(--background-fill-secondary, #f8fafc);
}

.lora-card .wrap,
.lora-card .form {
  min-width: 0 !important;
  width: 100% !important;
}

.lora-card .lora-name {
  width: 100% !important;
}

.lora-actions button {
  width: 100% !important;
  min-width: 0 !important;
  padding-inline: 0.35rem !important;
}

.history-gallery,
.comparison-gallery {
  min-width: 0 !important;
  width: 100% !important;
}

.comparison-gallery .grid-wrap {
  grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
}

.history-gallery img,
.comparison-gallery img {
  max-width: 100%;
  object-fit: contain;
}

.filter-stack,
.advanced-history-filter,
.drive-sync-section,
.queue-section,
.metadata-section,
.preset-section {
  min-width: 0 !important;
  width: 100% !important;
}

@media (min-width: 1024px) {
  .generation-layout {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  }

  .system-health-layout {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  }

  .comparison-layout {
    grid-template-columns: minmax(0, 1fr);
  }

  .history-gallery .grid-wrap {
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
  }
}

@media (min-width: 640px) and (max-width: 1023px) {
  .generation-layout {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr);
  }

  .system-health-layout {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 639px) {
  .gradio-container {
    padding-inline: 0.75rem !important;
  }

  .generation-layout {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr);
    gap: 0.75rem !important;
  }

  .system-health-layout {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr);
    gap: 0.75rem !important;
  }

  .size-dimensions,
  .seed-controls,
  .lora-strengths {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .history-filter,
  .model-preparation-actions,
  .history-actions,
  .system-actions,
  .queue-actions,
  .drive-actions,
  .metadata-actions,
  .preset-actions {
    grid-template-columns: minmax(0, 1fr);
  }

  .result-actions {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .lora-actions {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .history-gallery .grid-wrap {
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  }

  .comparison-gallery .grid-wrap {
    grid-template-columns: minmax(0, 1fr) !important;
  }

  .history-filter > *,
  .model-preparation-actions > *,
  .history-actions > *,
  .system-actions > *,
  .queue-actions > *,
  .drive-actions > *,
  .metadata-actions > *,
  .preset-actions > * {
    width: 100% !important;
  }
}

@media (max-width: 359px) {
  .size-dimensions,
  .seed-controls,
  .lora-strengths,
  .result-actions {
    grid-template-columns: minmax(0, 1fr);
  }

  .lora-actions {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

button:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible {
  outline: 3px solid #2563eb;
  outline-offset: 2px;
}
"""


def mobile_ui_css() -> str:
    """Return the shared CSS string used by the Gradio composition root."""

    return MOBILE_UI_CSS


__all__ = ["MOBILE_UI_CSS", "mobile_ui_css"]
