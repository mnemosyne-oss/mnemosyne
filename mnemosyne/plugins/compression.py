"""
CompressionPlugin for Mnemosyne
==============================

Applies rust_cave_001 compression to episodic memories during consolidation,
only when enabled via config and memory content exceeds threshold.

This plugin replaces the legacy MNEMOSYNE_USE_CAVEMAN environment variable.
"""

import os
import logging
from typing import Dict, Any
from mnemosyne.core.plugins import MnemosynePlugin
from mnemosyne.core import rust_cave_001

logger = logging.getLogger(__name__)


class CompressionPlugin(MnemosynePlugin):
    """
    Plugin that applies memory compression during consolidation.
    
    Enabled via: mnemosyne.plugins.compression.enabled: true
    
    Applies compression to tier-3 memories with content > TIER3_MAX_CHARS (default 300).
    Falls back to legacy MNEMOSYNE_USE_CAVEMAN env var if present.
    """
    
    name = "compression"
    version = "1.0.0"
    enabled = False
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Check legacy env var for backward compatibility
        legacy_enabled = os.environ.get("MNEMOSYNE_USE_CAVEMAN", "").strip().lower() in ("1", "true", "yes", "on")
        if legacy_enabled:
            logger.warning(
                "MNEMOSYNE_USE_CAVEMAN is deprecated. Use mnemosyne.plugins.compression.enabled in config instead. "
                "This will be removed in v2.1."
            )
            self.enabled = True
        
        # Override with config if provided
        if config and config.get("enabled") is not None:
            self.enabled = bool(config["enabled"])
        
        # Cache threshold from global config
        self.threshold = int(os.environ.get("MNEMOSYNE_TIER3_MAX_CHARS", "300"))
        
    def initialize(self) -> None:
        """Called once when plugin is loaded."""
        super().initialize()
        logger.info("CompressionPlugin initialized (enabled=%s, threshold=%d)", self.enabled, self.threshold)
        
    def shutdown(self) -> None:
        """Called once when plugin is unloaded."""
        super().shutdown()
        logger.info("CompressionPlugin shutdown")
        
    def on_consolidate(self, summary: Dict[str, Any]) -> None:
        """
        Apply compression to tier-3 memories during consolidation.
        
        Only runs if plugin is enabled.
        
        Args:
            summary: Consolidation summary from beam.py
        """
        if not self.enabled:
            return
        
        # Extract source memories from summary
        source_ids = summary.get("source_wm_ids", [])
        if not source_ids:
            return
        
        # For each source memory, if tier == 3 and content > threshold, compress
        # Note: This assumes the summary includes the full content of each source memory
        # If not, we'd need to query the DB — but beam.py's consolidation loop already has it.
        # We'll assume summary['source_memories'] contains the full text of each memory.
        # If not, we'll fall back to a DB lookup in a future version.
        
        source_memories = summary.get("source_memories", [])
        for mem in source_memories:
            if mem.get("tier") == 3 and len(mem.get("content", "")) > self.threshold:
                try:
                    compressed = rust_cave_001.compress(mem["content"])
                    if compressed and len(compressed) < len(mem["content"]):
                        mem["content"] = compressed
                        logger.debug("Compressed memory %s: %d → %d chars", 
                                   mem.get("id", "unknown"), 
                                   len(mem["content"]), 
                                   len(compressed))
                except Exception as e:
                    logger.warning("Compression failed for memory %s: %s", 
                                 mem.get("id", "unknown"), str(e))
                    # Preserve original content on failure
        
        # Update summary with modified memories
        summary["source_memories"] = source_memories
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize plugin metadata."""
        base = super().to_dict()
        base.update({
            "threshold": self.threshold,
        })
        return base