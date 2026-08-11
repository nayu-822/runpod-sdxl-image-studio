# RunPod production deployment

This directory describes the Phase 13 deployment of the existing RunPod
ComfyUI application and Image Studio runtime. It does not create Pods through
the RunPod API. Phase 12's self-only Safe Auto-Terminate remains the only
RunPod lifecycle API operation in this repository.

## Image contents and boundaries

The image is built from the pinned base image
runpod/comfyui:1.4.4-cuda12.8. It contains the repository-controlled Image
Studio source and Python package metadata, Alembic migrations, workflow
definitions, the Image Studio Python 3.11+ virtual environment at
/opt/image-studio-venv, the pinned rclone 1.74.2 binary verified against the
upstream SHA256SUMS file, and the deployment scripts.

It intentionally does not contain checkpoints, LoRA, VAE, upscaler files,
generated images, SQLite state, rclone.conf, OAuth tokens, API keys, cookies,
or .env files. Models and state are restored by the existing Phase 11/10
application services after startup.

The application is installed editable under /opt/image-studio. This keeps
the repository root, alembic.ini, alembic/, and workflows/ available to the
existing startup code without a package-data workaround. The Image Studio
venv is separate from the ComfyUI venv.

## Build and publish

Build on a machine with Docker and publish a traceable tag. Do not publish or
configure latest for a production Template.

~~~bash
git rev-parse --short HEAD
docker build --platform linux/amd64 -t runpod-sdxl-image-studio:phase13-<git-short-sha> .
docker tag runpod-sdxl-image-studio:phase13-<git-short-sha> <registry-user>/runpod-sdxl-image-studio:phase13-<git-short-sha>
docker push <registry-user>/runpod-sdxl-image-studio:phase13-<git-short-sha>
~~~

The build downloads only the pinned rclone release and its checksum file at
image-build time. Runtime bootstrap never runs git clone, pip install, apt,
curl | bash, or a model download.

## rclone Secret

Create a RunPod Secret named image_studio_rclone_config_b64 containing the
base64 encoding of the local rclone.conf. Do not commit the value or put it
in a Docker build argument. The decoded file is created at runtime as
/run/image-studio/rclone.conf with a 0700 parent and 0600 file mode.

PowerShell:

~~~powershell
$bytes = [IO.File]::ReadAllBytes('.\rclone.conf')
$secret = [Convert]::ToBase64String($bytes)
Set-Clipboard $secret
~~~

Linux/macOS:

~~~bash
RCLONE_CONFIG_B64="$(base64 -w0 ./rclone.conf)"
echo "base64 value prepared"
unset RCLONE_CONFIG_B64
~~~

Paste the value only into RunPod Secrets. In the Template environment use:

~~~dotenv
IMAGE_STUDIO_RCLONE_CONFIG_B64={{ RUNPOD_SECRET_image_studio_rclone_config_b64 }}
RCLONE_CONFIG=/run/image-studio/rclone.conf
RCLONE_REMOTE=drive
~~~

Bootstrap rejects an empty or invalid value, an existing symlink at the
configured path or parent, and any write/decode failure. It writes to a
temporary file, applies 0600, and atomically replaces the final path. It then
verifies that RCLONE_REMOTE appears in rclone listremotes; remote names and
command success are used only for validation, never the config contents. If
state sync, remote model preparation, or a configured remote is enabled
without a valid config, startup fails fast.

## Template settings

Copy template.env.example into the RunPod Template environment and replace
only the Secret placeholder through the RunPod Secret mechanism. The
recommended production Template is:

| Setting | Value |
| --- | --- |
| Name | SDXL Image Studio |
| Category | NVIDIA |
| Container image | fixed phase13-<git-short-sha> tag |
| Container disk | 50 GB |
| Volume disk | 0 GB |
| Network volume | none |
| HTTP port | 7860/http only |
| Public Template | off |
| Docker Entrypoint | unchanged |
| Docker Start Command | unchanged; image CMD invokes bootstrap |

Do not expose 8188/http, 8080/http, or 8888/http in production. ComfyUI is
reached by Image Studio through 127.0.0.1:8188. Port 22/tcp may be added
temporarily for debugging but is not part of the production Template. RunPod
supplies RUNPOD_POD_ID and RUNPOD_API_KEY; do not add or persist them in the
Template file.

## Startup flow

The bootstrap order is deliberately narrow:

1. Materialize and validate the optional rclone config.
2. Start the base image's existing /start.sh in the background.
3. Poll only http://127.0.0.1:8188/system_stats until ready, with a bounded
   IMAGE_STUDIO_BOOTSTRAP_COMFYUI_TIMEOUT_SECONDS timeout.
4. Start /opt/image-studio-venv/bin/runpod-sdxl-image-studio from
   /opt/image-studio.
5. Supervise both processes. An unexpected exit stops the other process and
   exits non-zero; the container is not kept alive with sleep infinity.
6. On SIGTERM/SIGINT, forward a graceful TERM and wait up to
   IMAGE_STUDIO_BOOTSTRAP_SHUTDOWN_GRACE_SECONDS before a bounded fallback.

The Docker health check probes only the user-facing Image Studio endpoint at
http://127.0.0.1:7860/. A health-check failure does not mutate SQLite or issue
a RunPod DELETE.

The bootstrap does not run state restore, Alembic migrations, model downloads,
/prompt, Drive sync, or Auto-Terminate. The existing application startup path
performs restore verification and Alembic upgrade; Phase 11's model transfer
worker handles exact on-demand model preparation; Phase 12 handles
Safe Auto-Terminate after its readiness and final-backup checks.

## Local and container smoke checks

The GPU-free smoke test verifies Python, package import, Gradio major version,
rclone availability, repository files, executable scripts, and a temporary
SQLite upgrade_database plus integrity check. It does not require a GPU,
ComfyUI, RunPod, Google Drive, or a real rclone remote.

~~~bash
bash -n deploy/runpod/bootstrap.sh
bash -n deploy/runpod/smoke-test.sh
pytest tests/unit/test_phase13_runpod_deployment.py
~~~

After a successful image build:

~~~bash
docker run --rm --entrypoint /bin/bash runpod-sdxl-image-studio:phase13-<git-short-sha> /opt/image-studio/deploy/runpod/smoke-test.sh
~~~

If Docker is unavailable, report the Docker build and container smoke test as
not executed. Static checks and local tests do not prove that an image was
built or that a real RunPod Template works.

## Fresh Pod verification

Deploy Fresh Pod A with IMAGE_STUDIO_AUTO_TERMINATE_ENABLED=false and verify:

1. /start.sh starts the base services and ComfyUI reaches local readiness.
2. Image Studio is reachable through the RunPod proxy on 7860.
3. rclone remote validation succeeds without exposing config contents.
4. Google Drive state restore completes or fails closed with a safe message.
5. The previous form state is restored and exact required models are prepared.
6. No startup /prompt is submitted.
7. A user Generate succeeds.
8. Image and metadata reach SYNCED, the manifest reaches SYNCED, and the
   final state backup is clean.

Then deploy Fresh Pod B from the same fixed Template tag and verify state,
form settings, and exact model preparation are restored without a new startup
generation. Only after A and B pass should auto-terminate be enabled on a
separate verification Pod:

~~~dotenv
IMAGE_STUDIO_AUTO_TERMINATE_ENABLED=true
~~~

Verify Generation completion, Drive and manifest synchronization, final state
backup, grace period, SAFE TO TERMINATE, one self-only DELETE, and the
existing Phase 12 fail-closed behavior for ambiguous responses. A live Pod
test is separate from CI and local Mock/Fake tests.

## Troubleshooting

- ComfyUI timeout: inspect base image logs and confirm the fixed local
  endpoint is available. Do not replace it with an external or user-supplied
  URL.
- rclone config failure: recreate the Secret from a local config, confirm the
  exact Secret name, and verify the drive: remote locally. Never print
  rclone.conf or use rclone config show in diagnostics.
- model unavailable: keep the application fail-closed and use the Phase 11
  model preparation UI. Do not copy models into the image or substitute a
  different checkpoint automatically.
- state restore failure: leave the application under its existing fail-closed
  write protection and inspect authorized application logs. Bootstrap does not
  download latest.json or reconcile SQLite state.
- container exits: inspect the safe bootstrap message for which supervised
  process ended. The bootstrap intentionally does not hide an unexpected exit
  behind an infinite sleep.
