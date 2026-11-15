# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39 AS builder
WORKDIR /build
RUN pip install --no-cache-dir build==1.6.1 setuptools==80.9.0 wheel==0.45.1
COPY pyproject.toml setup.py README.md LICENSE MANIFEST.in ./
COPY mnemosyne/ mnemosyne/
COPY hermes_memory_provider/ hermes_memory_provider/
RUN python -m build --wheel --no-isolation --outdir /wheels

FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39
ARG VERSION=dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="Mnemosyne MCP Server" \
      org.opencontainers.image.source="https://github.com/mnemosyne-oss/mnemosyne" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$REVISION
ENV HOME=/home/mnemosyne \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MNEMOSYNE_DATA_DIR=/data \
    MNEMOSYNE_FASTEMBED_CACHE_DIR=/cache/fastembed
COPY deploy/docker/requirements.lock /opt/mnemosyne/requirements.lock
COPY --from=builder /wheels/ /wheels/
RUN pip install --no-cache-dir --only-binary=:all: -c /opt/mnemosyne/requirements.lock /wheels/*.whl \
    && pip install --no-cache-dir --only-binary=:all: -c /opt/mnemosyne/requirements.lock 'mnemosyne-memory[mcp,embeddings]' \
    && rm -rf /wheels \
    && groupadd --gid 10001 mnemosyne \
    && useradd --uid 10001 --gid 10001 --home-dir /home/mnemosyne --create-home mnemosyne \
    && mkdir -p /data /cache/fastembed \
    && chown -R 10001:10001 /data /cache
COPY deploy/docker/healthcheck.py /opt/mnemosyne/healthcheck.py
USER 10001:10001
WORKDIR /home/mnemosyne
VOLUME ["/data", "/cache"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=12s --start-period=30s --retries=3 \
    CMD ["python", "/opt/mnemosyne/healthcheck.py"]
ENTRYPOINT ["mnemosyne", "mcp"]
CMD []
