# Phase 2A implementation

Phase 2A adds the bounded foundation for fixed-workflow SDXL generation with multiple LoRAs and VAE selection.

## Included

- `LoraSetting` validates relative names, strengths from `-2.0` to `2.0`, non-negative order, duplicate names, and duplicate order values.
- `GenerationSettings` carries `vae_name` and an ordered tuple of LoRA settings.
- The workflow adapter adds `LoraLoader` nodes only for selected LoRAs and chains model and CLIP outputs in order.
- The adapter adds a standard `VAELoader` only for an external VAE. `None` keeps the checkpoint's VAE output.
- Generation prevalidation checks capability membership, optional node availability, and `Settings.max_loras` before `/prompt`.
- The Gradio UI provides a bounded, mobile-friendly LoRA editor with add, remove, and reorder controls, model/CLIP strengths, and a checkpoint-internal VAE option.
- Capability refresh preserves still-valid VAE and LoRA selections and clears removed selections.

## Manual check

1. Start the app with `python -m runpod_sdxl_image_studio.app`.
2. Connect to ComfyUI and refresh capabilities.
3. Confirm the VAE selector contains `Checkpoint内蔵VAE` and external VAE names.
4. Add two LoRAs, set strengths, reorder them, and generate an image.
5. Confirm the result details show the VAE and LoRA order/strengths.
6. Confirm invalid or unavailable selections fail safely and the Generate button becomes usable again.

## Deferred to later phases

LoRA metadata such as trigger words, categories, favorites, recommended strengths, previews, search, and presets remain out of scope. SQLite/Alembic history, batch generation, queue management, metadata sidecars, Google Drive/rclone, upscaling, img2img, and RunPod API integration are also deferred.
