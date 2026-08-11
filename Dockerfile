FROM runpod/comfyui:1.4.4-cuda12.8

ARG RCLONE_VERSION=1.74.2

ENV IMAGE_STUDIO_ROOT=/opt/image-studio \
    IMAGE_STUDIO_VENV=/opt/image-studio-venv \
    PYTHONUNBUFFERED=1

WORKDIR /opt/image-studio

# Keep the ComfyUI environment owned by the RunPod base image.  Image Studio
# gets an isolated Python environment so Gradio and application dependencies
# cannot alter ComfyUI's runtime packages.
RUN python3.12 -m venv "$IMAGE_STUDIO_VENV" \
    && "$IMAGE_STUDIO_VENV/bin/pip" install --upgrade pip

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
COPY workflows ./workflows
COPY deploy/runpod ./deploy/runpod

RUN "$IMAGE_STUDIO_VENV/bin/pip" install --no-cache-dir -e "$IMAGE_STUDIO_ROOT" \
    && chmod 0755 /opt/image-studio/deploy/runpod/bootstrap.sh \
    && chmod 0755 /opt/image-studio/deploy/runpod/smoke-test.sh

# Pin and verify the rclone release during image build.  No runtime download
# or installer script is used, and the archive is unpacked without trusting a
# path supplied by the archive.
RUN set -eux; \
    mkdir -p /tmp/rclone; \
    curl --fail --silent --show-error --location --retry 3 \
        --output "/tmp/rclone/rclone-v${RCLONE_VERSION}-linux-amd64.zip" \
        "https://downloads.rclone.org/v${RCLONE_VERSION}/rclone-v${RCLONE_VERSION}-linux-amd64.zip"; \
    curl --fail --silent --show-error --location --retry 3 \
        --output /tmp/rclone/SHA256SUMS \
        "https://downloads.rclone.org/v${RCLONE_VERSION}/SHA256SUMS"; \
    cd /tmp/rclone; \
    grep "  rclone-v${RCLONE_VERSION}-linux-amd64.zip$" SHA256SUMS > checksum; \
    sha256sum --check --status checksum; \
    cd /opt/image-studio; \
    python3 -c 'import pathlib, sys, zipfile; archive = pathlib.Path(sys.argv[1]); target = pathlib.Path("/usr/local/bin/rclone"); bundle = zipfile.ZipFile(archive); names = [name for name in bundle.namelist() if name.endswith("/rclone") or name == "rclone"]; assert len(names) == 1 and not pathlib.PurePosixPath(names[0]).is_absolute(), "unexpected rclone archive layout"; target.write_bytes(bundle.read(names[0])); bundle.close(); target.chmod(0o755)' "/tmp/rclone/rclone-v${RCLONE_VERSION}-linux-amd64.zip"; \
    rclone version; \
    rm -rf /tmp/rclone

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=3 \
    CMD curl --fail --silent --show-error --max-time 4 \
        http://127.0.0.1:7860/ > /dev/null || exit 1

# The base image's entrypoint is intentionally left untouched.  The base
# /start.sh is launched by bootstrap before the application is started.
CMD ["/opt/image-studio/deploy/runpod/bootstrap.sh"]
