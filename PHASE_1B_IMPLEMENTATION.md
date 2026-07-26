# Phase 1B implementation notes

Phase 1B provides the first single-image SDXL generation path using the repository-controlled
`sdxl_txt2img` API workflow.

- The Gradio 5 form accepts prompts, size, seed mode, steps, CFG, checkpoint, sampler, and scheduler.
- Random seed `-1` is resolved once in the application service and the resolved seed is returned with the result.
- ComfyUI `/prompt`, WebSocket progress, bounded `/history/{prompt_id}` recovery polling, and `/view` image retrieval are adapter responsibilities.
- Output images are validated with Pillow and stored atomically under the configured data directory using an Asia/Tokyo date directory.
- Jobs and generation results are intentionally in memory for this phase.

Out of scope for Phase 1B: LoRA application, batch generation, VAE switching, SQLite/Alembic persistence,
PNG metadata, sidecar JSON, history search, presets, Google Drive/rclone synchronization, and upscaling.

Manual check:

1. Start ComfyUI with the required SDXL checkpoint and standard nodes.
2. Run `python -m runpod_sdxl_image_studio.app`.
3. Open the Gradio page, press the system connection check, then refresh capabilities.
4. Select a checkpoint, enter a prompt, and press Generate.
5. Confirm progress, the generated image, the resolved seed, and the local date-based output file.
