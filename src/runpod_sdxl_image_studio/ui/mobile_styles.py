"""Phase Cのモダンダーク・レスポンシブUI用CSS。"""

from __future__ import annotations

MOBILE_UI_CSS = """
:root {
  color-scheme: dark;
  --studio-background: #0B0D12;
  --studio-surface: #12151C;
  --studio-surface-secondary: #181C24;
  --studio-surface-elevated: #202633;
  --studio-text: #F5F7FB;
  --studio-text-secondary: #9AA4B2;
  --studio-border: rgba(255, 255, 255, 0.08);
  --studio-accent: #7C5CFF;
  --studio-accent-hover: #8B73FF;
  --studio-success: #22C55E;
  --studio-warning: #F59E0B;
  --studio-danger: #EF4444;
}

html,
body {
  background: var(--studio-background) !important;
  color: var(--studio-text) !important;
  color-scheme: dark !important;
}

.gradio-container {
  box-sizing: border-box;
  max-width: 1440px !important;
  width: 100% !important;
  min-height: 100vh;
  padding: 1rem clamp(0.75rem, 2vw, 2rem) !important;
  background: var(--studio-background) !important;
  color: var(--studio-text) !important;
  overflow: clip;
}

.gradio-container *,
.gradio-container *::before,
.gradio-container *::after {
  box-sizing: border-box;
  min-width: 0;
}

.studio-header {
  margin: 0 0 1rem !important;
}

.studio-header h2 {
  margin: 0 !important;
  color: var(--studio-text) !important;
  font-size: clamp(1.25rem, 2vw, 1.65rem) !important;
  letter-spacing: -0.02em;
}

.gradio-container .row,
.gradio-container .column,
.tab-nav,
[role="tablist"] {
  min-width: 0 !important;
  max-width: 100% !important;
}

.tab-nav,
[role="tablist"] {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 0.35rem !important;
  border-bottom: 1px solid var(--studio-border) !important;
}

.tab-nav button,
[role="tablist"] button {
  min-width: 0 !important;
  min-height: 44px !important;
  padding: 0.55rem 0.8rem !important;
  border-radius: 10px 10px 0 0 !important;
  color: var(--studio-text-secondary) !important;
  white-space: normal !important;
}

.tab-nav button.selected,
[role="tablist"] button[aria-selected="true"] {
  color: var(--studio-text) !important;
  background: var(--studio-surface-secondary) !important;
  border-color: var(--studio-accent) !important;
}

.generation-layout {
  display: grid !important;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.65fr);
  align-items: start !important;
  gap: clamp(1rem, 2vw, 1.5rem) !important;
}

.generation-primary,
.generation-preview,
.generation-advanced,
.generation-batch,
.prompt-section,
.prompt-editor,
.lora-editor-section,
.recent-settings,
.generation-status-surface {
  min-width: 0 !important;
  width: 100% !important;
}

.generation-primary,
.generation-preview {
  gap: 1rem !important;
}

.generation-primary > .block,
.generation-preview > .block,
.generation-advanced,
.generation-batch,
.generation-status-surface,
.lora-editor-section,
.recent-settings {
  border: 1px solid var(--studio-border) !important;
  border-radius: 16px !important;
  background: var(--studio-surface) !important;
  box-shadow: none !important;
}

.prompt-section {
  padding: 0.25rem 0 !important;
}

.prompt-section > .prose h3 {
  margin: 0.3rem 0 0.55rem !important;
  color: var(--studio-text) !important;
}

.prompt-editor textarea {
  min-height: 9rem !important;
  resize: vertical !important;
  border-radius: 12px !important;
  background: var(--studio-surface-secondary) !important;
  color: var(--studio-text) !important;
}

.prompt-editor.negative textarea {
  min-height: 6.5rem !important;
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

.prompt-actions {
  display: flex !important;
  justify-content: flex-end !important;
  margin: 0.25rem 0 0.5rem !important;
}

.prompt-actions button {
  flex: 0 1 auto;
  min-width: 4.5rem !important;
  color: var(--studio-text-secondary) !important;
  background: transparent !important;
  border-color: transparent !important;
}

.prompt-actions button:hover {
  color: var(--studio-text) !important;
  background: var(--studio-surface-elevated) !important;
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
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 8rem), 1fr));
}

.compact-controls {
  display: grid !important;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.75rem !important;
}

.lora-summary {
  margin: 0.5rem 0 0.25rem !important;
  padding: 0.65rem 0.8rem !important;
  border-radius: 12px !important;
  background: var(--studio-surface-secondary) !important;
  color: var(--studio-text) !important;
}

.lora-summary strong {
  color: var(--studio-text-secondary) !important;
  font-size: 0.85rem;
  font-weight: 600;
}

.lora-summary code {
  display: inline-block;
  margin: 0.2rem 0.25rem 0 0;
  padding: 0.18rem 0.45rem;
  border-radius: 999px;
  background: var(--studio-surface-elevated);
  color: var(--studio-text);
  font-size: 0.8rem;
}

.lora-card {
  gap: 0.5rem !important;
  padding: 0.75rem !important;
  border: 1px solid var(--studio-border) !important;
  border-radius: 12px !important;
  background: var(--studio-surface-secondary) !important;
}

.lora-card .wrap,
.lora-card .form,
.lora-card-row {
  min-width: 0 !important;
  width: 100% !important;
}

.lora-card .lora-name {
  width: 100% !important;
}

.lora-catalog-gallery,
.lora-catalog-gallery .grid-wrap,
.lora-catalog-gallery .gallery-item {
  min-width: 0 !important;
  max-width: 100% !important;
}

.lora-catalog-gallery .grid-wrap {
  display: grid !important;
  grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
  gap: 0.75rem !important;
  overflow: clip !important;
}

.lora-catalog-gallery .gallery-item {
  min-height: 44px !important;
  overflow: hidden !important;
  border: 1px solid var(--studio-border) !important;
  border-radius: 14px !important;
  background: var(--studio-surface-secondary) !important;
}

.lora-catalog-gallery img {
  aspect-ratio: 1 / 1 !important;
  width: 100% !important;
  object-fit: cover !important;
}

.lora-catalog-gallery figcaption {
  min-height: 44px !important;
  padding: 0.45rem !important;
  color: var(--studio-text) !important;
  white-space: pre-line !important;
}

.lora-detail {
  margin-top: 0.75rem !important;
}

.lora-actions button,
.mobile-tap-button button,
.history-actions button,
.system-actions button,
.queue-actions button,
.drive-actions button,
.metadata-actions button,
.preset-actions button {
  min-height: 44px !important;
}

.generation-sticky-action {
  position: sticky !important;
  bottom: 0;
  z-index: 20;
  width: 100% !important;
  margin-top: 0.75rem !important;
  padding: 0.65rem 0 !important;
  background: linear-gradient(
    to bottom,
    rgba(11, 13, 18, 0),
    var(--studio-background) 35%
  ) !important;
  padding-bottom: calc(0.65rem + env(safe-area-inset-bottom, 0px)) !important;
}

.generation-sticky-action button {
  width: 100% !important;
  min-height: 52px !important;
  border-radius: 15px !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
}

.generation-sticky-action button + button {
  margin-top: 0.5rem !important;
}

.generation-status-surface {
  padding: 0.9rem !important;
}

.generation-status-card {
  border: 0 !important;
  border-radius: 12px !important;
  padding: 0 !important;
  background: transparent !important;
}

.generation-status-card h3 {
  margin: 0 !important;
  color: var(--studio-text) !important;
}

.interactive-result-gallery,
.interactive-result-gallery .grid-wrap,
.interactive-result-gallery .gallery-item,
.history-gallery,
.comparison-gallery {
  min-width: 0 !important;
  max-width: 100% !important;
  width: 100% !important;
}

.interactive-result-gallery .grid-wrap,
.history-gallery .grid-wrap,
.comparison-gallery .grid-wrap {
  overflow: clip !important;
}

.interactive-result-gallery img,
.history-gallery img,
.comparison-gallery img {
  max-width: 100%;
  border-radius: 12px;
  object-fit: contain;
}

.interactive-result-gallery .grid-wrap {
  grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  gap: 0.65rem !important;
}

.validation-message {
  color: var(--studio-warning) !important;
}

.model-preparation-selection,
.model-preparation-actions,
.system-health-column,
.system-error-column,
.system-health-section,
.system-error-section,
.filter-stack,
.advanced-history-filter,
.drive-sync-section,
.queue-section,
.metadata-section,
.preset-section {
  min-width: 0 !important;
  width: 100% !important;
}

button,
input,
textarea,
select {
  color-scheme: dark !important;
}

button:focus-visible,
input:focus-visible,
textarea:focus-visible,
select:focus-visible {
  outline: 3px solid var(--studio-accent-hover) !important;
  outline-offset: 2px;
}

@media (min-width: 1024px) {
  .generation-layout {
    grid-template-columns: minmax(420px, 0.8fr) minmax(0, 1.6fr);
  }

  .history-gallery .grid-wrap {
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
  }
}

@media (min-width: 640px) and (max-width: 1023px) {
  .generation-layout {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 639px) {
  .gradio-container {
    padding-inline: 0.75rem !important;
  }

  .generation-layout {
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

  .lora-catalog-gallery .grid-wrap {
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  }

  .comparison-gallery .grid-wrap {
    grid-template-columns: minmax(0, 1fr) !important;
  }
}

@media (max-width: 359px) {
  .compact-controls,
  .size-dimensions,
  .seed-controls,
  .lora-strengths,
  .result-actions {
    grid-template-columns: minmax(0, 1fr);
  }

  .lora-catalog-gallery .grid-wrap {
    grid-template-columns: minmax(0, 1fr) !important;
  }
}
"""


def mobile_ui_css() -> str:
    """Return the shared CSS string used by the Gradio composition root."""

    return MOBILE_UI_CSS


__all__ = ["MOBILE_UI_CSS", "mobile_ui_css"]
