# Mnemosyne MCP Server — Docker Image
#
# Run the Mnemosyne MCP server in a container for easy deployment
# with any MCP-compatible client.
#
# Build:
#   docker build -t mnemosyne-mcp .
#
# Run (stdio — for Claude Desktop, Cursor, etc.):
#   docker run -i --rm -v mnemosyne-data:/data mnemosyne-mcp
#
# Run (SSE — for web clients):
#   docker run -d --rm -p 8080:8080 \
#     -v mnemosyne-data:/data \
#     -e MNEMOSYNE_MCP_TOKEN=my-secret \
#     mnemosyne-mcp --transport sse --host 0.0.0.0 --port 8080
#
# With custom data directory:
#   docker run -i --rm \
#     -v /host/path/to/data:/data \
#     -e MNEMOSYNE_DATA_DIR=/data \
#     mnemosyne-mcp

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Mnemosyne MCP Server"
LABEL org.opencontainers.image.description="Universal memory layer MCP server for any AI agent"
LABEL org.opencontainers.image.source="https://github.com/AxDSan/mnemosyne"
LABEL org.opencontainers.image.licenses="MIT"

# Install this checkout with MCP + SSE extras so locally built images contain
# the source that was used to build them.
COPY . /opt/mnemosyne
RUN pip install --no-cache-dir "/opt/mnemosyne[mcp]"

# Default data directory (overridable via MNEMOSYNE_DATA_DIR env var)
ENV MNEMOSYNE_DATA_DIR=/data
VOLUME /data

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "from mnemosyne import recall; recall('health', top_k=1)" || exit 1

# Default: stdio transport (for Claude Desktop, Cursor, etc.)
ENTRYPOINT ["mnemosyne", "mcp"]
CMD []
