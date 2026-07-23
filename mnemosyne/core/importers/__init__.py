"""
Mnemosyne Cross-Provider Importers

Import memories from other AI memory providers into Mnemosyne.

Supported providers:
    mem0        — Mem0 (cloud + self-hosted)
    letta       — Letta (formerly MemGPT)
    zep         — Zep enterprise memory
    cognee      — Cognee graph memory
    honcho      — Honcho entity memory
    supermemory — SuperMemory cloud API
    hindsight   — Hindsight/Hermes memory provider
    holographic — Hermes Holographic Memory (SQLite + FTS5 + HRR)

CLI usage:
    hermes mnemosyne import --from mem0 --api-key sk-xxx
    hermes mnemosyne import --file export.json --dry-run
    hermes mnemosyne import --list-providers
    hermes mnemosyne import --from mem0 --generate-script

Agentic migration:
    hermes mnemosyne import --from zep --agentic
"""

__all__ = [
    "AgenticImporter",
    "BaseImporter",
    "CogneeImporter",
    "HindsightImporter",
    "HolographicImporter",
    "HonchoImporter",
    "ImporterResult",
    "LettaImporter",
    "Mem0Importer",
    "SuperMemoryImporter",
    "ZepImporter",
    "generate_agent_instructions",
    "generate_docs_instructions",
    "generate_migration_script",
    "import_from_file",
    "import_from_hindsight",
    "import_from_holographic",
    "import_from_mem0",
    "import_from_provider",
    "list_providers",
]

from .base import BaseImporter, ImporterResult, import_from_file
from .mem0 import Mem0Importer, import_from_mem0
from .letta import LettaImporter
from .zep import ZepImporter
from .cognee import CogneeImporter
from .honcho import HonchoImporter
from .supermemory import SuperMemoryImporter
from .hindsight import HindsightImporter, import_from_hindsight
from .holographic import HolographicImporter, import_from_holographic
from .agentic import (
    AgenticImporter,
    generate_migration_script,
    generate_agent_instructions,
    generate_docs_instructions,
)


# Registry of supported providers
PROVIDERS = {
    "mem0": {
        "name": "Mem0",
        "class": Mem0Importer,
        "module": "mem0",
        "docs": "https://docs.mem0.ai",
        "env_key": "MEM0_API_KEY",
        "pypi_package": "mem0ai",
        "description": "Memory platform with 24 vector store backends. Supports user/agent/app scoping.",
    },
    "letta": {
        "name": "Letta (MemGPT)",
        "class": LettaImporter,
        "module": "letta",
        "docs": "https://docs.letta.com",
        "env_key": "LETTA_API_KEY",
        "pypi_package": "letta-client",
        "description": "Agent OS with hierarchical memory blocks. Export via .af AgentFile format.",
    },
    "zep": {
        "name": "Zep",
        "class": ZepImporter,
        "module": "zep",
        "docs": "https://docs.getzep.com",
        "env_key": "ZEP_API_KEY",
        "pypi_package": "zep-cloud",
        "description": "Enterprise temporal knowledge graph. Session-based with user/thread model.",
    },
    "cognee": {
        "name": "Cognee",
        "class": CogneeImporter,
        "module": "cognee",
        "docs": "https://docs.cognee.ai",
        "env_key": None,
        "pypi_package": "cognee",
        "description": "Graph-based memory with Kùzu + LanceDB + SQLite. Nodes/edges map to episodic/triples.",
    },
    "honcho": {
        "name": "Honcho",
        "class": HonchoImporter,
        "module": "honcho",
        "docs": "https://docs.honcho.dev",
        "env_key": None,
        "pypi_package": "honcho-ai",
        "description": "Entity-centric memory by Plastic Labs. Workspace → Peer → Session → Message model.",
    },
    "supermemory": {
        "name": "SuperMemory",
        "class": SuperMemoryImporter,
        "module": "supermemory",
        "docs": "https://supermemory.ai/docs",
        "env_key": "SUPERMEMORY_API_KEY",
        "pypi_package": "supermemory",
        "description": "Cloud memory API with container tags and document management.",
    },
    "hindsight": {
        "name": "Hindsight",
        "class": HindsightImporter,
        "module": "hindsight",
        "docs": "https://github.com/vectorize-io/hindsight",
        "env_key": None,
        "pypi_package": None,
        "description": "Hermes memory provider. Imports consolidated memories into episodic storage while preserving timestamps and metadata.",
    },
    "holographic": {
        "name": "Hermes Holographic Memory",
        "class": HolographicImporter,
        "module": "holographic",
        "docs": "https://github.com/NousResearch/hermes-agent/tree/main/plugins/memory/holographic",
        "env_key": None,
        "pypi_package": None,
        "description": "Hermes SQLite memory plugin with FTS5 + HRR vectors, entity linking, trust scoring, and category banks. Imports from ~/.hermes/memory_store.db.",
    },
}


def list_providers() -> list:
    """Return list of supported provider names."""
    return list(PROVIDERS.keys())


def get_provider_info(name: str) -> dict:
    """Get metadata for a supported provider."""
    return PROVIDERS.get(name, {})


def import_from_provider(provider: str, mnemosyne, dry_run: bool = False,
                         session_id: str = None, channel_id: str = None,
                         **kwargs) -> ImporterResult:
    """Import memories from a supported provider into Mnemosyne.

    Args:
        provider: Provider name (e.g., 'mem0', 'letta').
        mnemosyne: Mnemosyne instance to import into.
        dry_run: If True, validate and transform but don't write.
        session_id: Override session for imported memories.
        channel_id: Channel to assign imported memories to.
        **kwargs: Provider-specific arguments (api_key, user_id, etc.)

    Returns:
        ImporterResult with import statistics.

    Raises:
        ValueError: If provider is not supported.
    """
    provider_info = PROVIDERS.get(provider)
    if not provider_info:
        supported = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unsupported provider: '{provider}'. Supported: {supported}"
        )

    importer_cls = provider_info["class"]
    importer = importer_cls(**kwargs)
    return importer.run(
        mnemosyne,
        dry_run=dry_run,
        session_id=session_id,
        channel_id=channel_id,
    )


def generate_script(provider: str, **kwargs) -> str:
    """Generate a migration script for the given provider."""
    provider_info = PROVIDERS.get(provider)
    if not provider_info:
        supported = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unsupported provider: '{provider}'. Supported: {supported}"
        )
    return generate_migration_script(provider, **kwargs)
