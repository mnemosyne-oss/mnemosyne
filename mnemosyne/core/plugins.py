"""
Mnemosyne Plugin Architecture
===============================
Extensible plugin system for Mnemosyne memory operations.

Plugins can hook into memory lifecycle events:
- on_remember: called when a memory is stored
- on_recall: called when a memory is recalled
- on_consolidate: called during sleep/consolidation
- on_invalidate: called when a memory is invalidated

Plugin discovery loads plugins from ~/.hermes/mnemosyne/plugins/
and built-in plugins are always available.
"""

import abc
import importlib
import importlib.util
import inspect
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

# Plugin directory under ~/.hermes
DEFAULT_PLUGIN_DIR = Path.home() / ".hermes" / "mnemosyne" / "plugins"

logger = logging.getLogger(__name__)


class MnemosynePlugin(abc.ABC):
    """
    Base class for all Mnemosyne plugins.

    Subclasses must implement the four lifecycle hooks.
    Each hook receives the relevant data and can perform
    side effects (logging, metrics, filtering, etc.).
    """

    name: str = ""
    version: str = "1.0.0"
    enabled: bool = True

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._initialized = False

    def initialize(self) -> None:
        """Called once when the plugin is loaded."""
        self._initialized = True

    def shutdown(self) -> None:
        """Called once when the plugin is unloaded."""
        self._initialized = False

    @abc.abstractmethod
    def on_remember(self, memory: Dict[str, Any]) -> None:
        """Called when a memory is stored."""
        ...

    @abc.abstractmethod
    def on_recall(self, memory: Dict[str, Any]) -> None:
        """Called when a memory is recalled."""
        ...

    @abc.abstractmethod
    def on_consolidate(self, summary: Dict[str, Any]) -> None:
        """Called during sleep/consolidation."""
        ...

    @abc.abstractmethod
    def on_invalidate(self, memory_id: str) -> None:
        """Called when a memory is invalidated."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize plugin metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled,
            "initialized": self._initialized,
            "config": self.config,
        }


class LoggingPlugin(MnemosynePlugin):
    """
    Built-in plugin that logs all memory operations.
    """

    name = "logging"
    version = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.log_level = self.config.get("log_level", "INFO")
        self._memory_log: List[Dict[str, Any]] = []
        self._max_entries = self.config.get("max_entries", 10000)

    def on_remember(self, memory: Dict[str, Any]) -> None:
        entry = {
            "event": "remember",
            "timestamp": datetime.now().isoformat(),
            "memory_id": memory.get("id"),
            "content_preview": self._preview(memory.get("content", "")),
        }
        self._append(entry)
        logger.log(getattr(logging, self.log_level, logging.INFO),
                   "[LoggingPlugin] remember: %s", entry)

    def on_recall(self, memory: Dict[str, Any]) -> None:
        entry = {
            "event": "recall",
            "timestamp": datetime.now().isoformat(),
            "memory_id": memory.get("id"),
            "content_preview": self._preview(memory.get("content", "")),
        }
        self._append(entry)
        logger.log(getattr(logging, self.log_level, logging.INFO),
                   "[LoggingPlugin] recall: %s", entry)

    def on_consolidate(self, summary: Dict[str, Any]) -> None:
        entry = {
            "event": "consolidate",
            "timestamp": datetime.now().isoformat(),
            "summary_preview": self._preview(summary.get("summary", "")),
            "source_count": len(summary.get("source_wm_ids", [])),
        }
        self._append(entry)
        logger.log(getattr(logging, self.log_level, logging.INFO),
                   "[LoggingPlugin] consolidate: %s", entry)

    def on_invalidate(self, memory_id: str) -> None:
        entry = {
            "event": "invalidate",
            "timestamp": datetime.now().isoformat(),
            "memory_id": memory_id,
        }
        self._append(entry)
        logger.log(getattr(logging, self.log_level, logging.INFO),
                   "[LoggingPlugin] invalidate: %s", entry)

    def _preview(self, content: str, max_len: int = 80) -> str:
        if len(content) <= max_len:
            return content
        return content[:max_len] + "..."

    def _append(self, entry: Dict[str, Any]) -> None:
        self._memory_log.append(entry)
        if len(self._memory_log) > self._max_entries:
            self._memory_log.pop(0)

    def get_log(self) -> List[Dict[str, Any]]:
        """Return the in-memory log entries."""
        return list(self._memory_log)

    def clear_log(self) -> None:
        """Clear the in-memory log."""
        self._memory_log.clear()


class MetricsPlugin(MnemosynePlugin):
    """
    Built-in plugin that collects performance metrics for memory operations.
    """

    name = "metrics"
    version = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._counters: Dict[str, int] = {
            "remember": 0,
            "recall": 0,
            "consolidate": 0,
            "invalidate": 0,
        }
        self._timings: Dict[str, List[float]] = {
            "remember": [],
            "recall": [],
            "consolidate": [],
            "invalidate": [],
        }
        self._max_timing_samples = self.config.get("max_timing_samples", 1000)

    def on_remember(self, memory: Dict[str, Any]) -> None:
        self._counters["remember"] += 1

    def on_recall(self, memory: Dict[str, Any]) -> None:
        self._counters["recall"] += 1

    def on_consolidate(self, summary: Dict[str, Any]) -> None:
        self._counters["consolidate"] += 1

    def on_invalidate(self, memory_id: str) -> None:
        self._counters["invalidate"] += 1

    def record_timing(self, event: str, duration_ms: float) -> None:
        """Record the duration of an operation."""
        if event not in self._timings:
            self._timings[event] = []
        self._timings[event].append(duration_ms)
        if len(self._timings[event]) > self._max_timing_samples:
            self._timings[event].pop(0)

    def get_counters(self) -> Dict[str, int]:
        """Return event counters."""
        return dict(self._counters)

    def get_timings(self, event: str) -> List[float]:
        """Return timing samples for an event."""
        return list(self._timings.get(event, []))

    def get_average_timing(self, event: str) -> Optional[float]:
        """Return average timing for an event."""
        samples = self._timings.get(event, [])
        if not samples:
            return None
        return sum(samples) / len(samples)

    def reset(self) -> None:
        """Reset all counters and timings."""
        for key in self._counters:
            self._counters[key] = 0
        for key in self._timings:
            self._timings[key].clear()

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of all metrics."""
        summary = {
            "counters": self.get_counters(),
            "averages": {},
        }
        for event in self._timings:
            avg = self.get_average_timing(event)
            summary["averages"][event] = avg
        return summary


class FilterPlugin(MnemosynePlugin):
    """
    Built-in plugin that filters memories based on custom rules.

    Rules are callables registered via add_rule().
    Each rule receives the memory dict and returns True to allow,
    False to block. Blocked memories are tracked but not passed
    to downstream plugins.
    """

    name = "filter"
    version = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._rules: List[Callable[[Dict[str, Any]], bool]] = []
        self._blocked: List[Dict[str, Any]] = []
        self._max_blocked = self.config.get("max_blocked", 1000)

    def add_rule(self, rule: Callable[[Dict[str, Any]], bool]) -> None:
        """Register a filtering rule."""
        self._rules.append(rule)

    def remove_rule(self, rule: Callable[[Dict[str, Any]], bool]) -> None:
        """Unregister a filtering rule."""
        if rule in self._rules:
            self._rules.remove(rule)

    def clear_rules(self) -> None:
        """Remove all filtering rules."""
        self._rules.clear()

    def on_remember(self, memory: Dict[str, Any]) -> None:
        if not self._passes(memory):
            self._block(memory)

    def on_recall(self, memory: Dict[str, Any]) -> None:
        if not self._passes(memory):
            self._block(memory)

    def on_consolidate(self, summary: Dict[str, Any]) -> None:
        if not self._passes(summary):
            self._block(summary)

    def on_invalidate(self, memory_id: str) -> None:
        pass

    def _passes(self, item: Dict[str, Any]) -> bool:
        for rule in self._rules:
            try:
                if not rule(item):
                    return False
            except Exception:
                return False
        return True

    def _block(self, item: Dict[str, Any]) -> None:
        self._blocked.append({
            "timestamp": datetime.now().isoformat(),
            "item": item,
        })
        if len(self._blocked) > self._max_blocked:
            self._blocked.pop(0)

    def get_blocked(self) -> List[Dict[str, Any]]:
        """Return all blocked items."""
        return list(self._blocked)

    def is_blocked(self, memory_id: str) -> bool:
        """Check if a memory ID has been blocked."""
        for entry in self._blocked:
            item = entry.get("item", {})
            if item.get("id") == memory_id:
                return True
        return False


class CompressionPlugin(MnemosynePlugin):
    """
    Built-in plugin for optional pre-compression of memory content
    before LLM summarization/consolidation.

    When enabled and a compression provider (e.g. ``rust_cave_001``)
    is available, each memory line is compressed before chunking.
    This reduces context window pressure for small local LLMs.

    Config:
        enabled (bool): Enable compression (default: False).
        provider (str): Compression backend name (default: "caveman").
            Currently only "caveman" (``rust_cave_001``) is supported.
        threshold_chars (int): Minimum character count below which
            compression is skipped (default: 20).

    Env-var fallback (deprecated):
        ``MNEMOSYNE_USE_CAVEMAN=1`` still works but shows a deprecation
        warning. Prefer the config/plugin path.
    """

    name = "compression"
    version = "1.0.0"
    enabled = False  # Opt-in; must be explicitly enabled via config or deprecated env var

    def __init__(self, config: dict = None):
        super().__init__(config)
        self._provider = self.config.get("provider", "caveman")
        self._threshold = int(self.config.get("threshold_chars", 20))
        self._caveman_available = False
        self._caveman_import_attempted = False

        # Apply enabled from config (overrides class-level default)
        self.enabled = bool(self.config.get("enabled", False))

        # --- Deprecated env-var fallback ---
        env_val = os.environ.get("MNEMOSYNE_USE_CAVEMAN", "").lower()
        if env_val in ("1", "true", "yes"):
            import warnings
            warnings.warn(
                "MNEMOSYNE_USE_CAVEMAN env var is deprecated. "
                "Use the compression plugin config instead: "
                "'mnemosyne.plugins.compression.enabled: true'",
                DeprecationWarning,
                stacklevel=2,
            )
            self.enabled = True

    def _lazy_init_caveman(self) -> None:
        """Try importing rust_cave_001 once; cache result."""
        if self._caveman_import_attempted:
            return
        self._caveman_import_attempted = True
        try:
            from rust_cave_001 import compress  # noqa: F811
            self._caveman_available = True
        except ImportError:
            logger.debug("CompressionPlugin: rust_cave_001 not installed")
        except Exception:
            logger.warning("CompressionPlugin: unexpected import error", exc_info=True)

    def compress_lines(self, lines: list[str]) -> list[str]:
        """Compress a list of memory content lines in-place.

        Returns the compressed lines (or originals on any failure).
        No-op if the provider is not available.
        """
        if not self.enabled:
            return lines
        if self._provider != "caveman":
            logger.warning("CompressionPlugin: unknown provider %r, skipping", self._provider)
            return lines

        self._lazy_init_caveman()
        if not self._caveman_available:
            return lines

        from rust_cave_001 import compress  # type: ignore[import-untyped]

        compressed: list[str] = []
        for line in lines:
            if len(line) < self._threshold:
                compressed.append(line)
                continue
            try:
                c = compress(line)
                compressed.append(c if len(c) > 2 else line)
            except Exception:
                compressed.append(line)
        return compressed

    # --- Lifecycle hooks (minimal; compression is called explicitly) ---

    def on_remember(self, memory: dict) -> None:
        pass

    def on_recall(self, memory: dict) -> None:
        pass

    def on_consolidate(self, summary: dict) -> None:
        pass

    def on_invalidate(self, memory_id: str) -> None:
        pass


class PluginManager:
    """
    Register, load, and manage Mnemosyne plugins.

    Supports:
    - Built-in plugins (LoggingPlugin, MetricsPlugin, FilterPlugin, CompressionPlugin)
    - External plugins discovered from ~/.hermes/mnemosyne/plugins/
    - Manual registration of plugin classes
    """

    def __init__(self, plugin_dir: Path = None):
        self._registry: Dict[str, Type[MnemosynePlugin]] = {}
        self._instances: Dict[str, MnemosynePlugin] = {}
        self._plugin_dir = plugin_dir or DEFAULT_PLUGIN_DIR

        # Register built-in plugins
        self.register_plugin("logging", LoggingPlugin)
        self.register_plugin("metrics", MetricsPlugin)
        self.register_plugin("filter", FilterPlugin)
        self.register_plugin("compression", CompressionPlugin)

    def register_plugin(self, name: str, plugin_class: Type[MnemosynePlugin]) -> None:
        """
        Register a plugin class by name.

        Args:
            name: Unique identifier for the plugin.
            plugin_class: Subclass of MnemosynePlugin.

        Raises:
            TypeError: If plugin_class is not a subclass of MnemosynePlugin.
            ValueError: If name is already registered.
        """
        if not inspect.isclass(plugin_class) or not issubclass(plugin_class, MnemosynePlugin):
            raise TypeError(f"plugin_class must be a subclass of MnemosynePlugin, got {plugin_class}")
        if name in self._registry:
            raise ValueError(f"Plugin '{name}' is already registered")
        self._registry[name] = plugin_class

    def load_plugin(self, name: str, config: Dict[str, Any] = None) -> MnemosynePlugin:
        """
        Instantiate and initialize a registered plugin.

        Args:
            name: Registered plugin name.
            config: Optional configuration dict passed to the plugin.

        Returns:
            The initialized plugin instance.

        Raises:
            ValueError: If the plugin is not registered.
            RuntimeError: If the plugin is already loaded.
        """
        if name not in self._registry:
            raise ValueError(f"Plugin '{name}' is not registered")
        if name in self._instances:
            raise RuntimeError(f"Plugin '{name}' is already loaded")

        plugin_class = self._registry[name]
        instance = plugin_class(config=config or {})
        instance.initialize()
        self._instances[name] = instance
        logger.info("Loaded plugin: %s v%s", instance.name, instance.version)
        return instance

    def unload_plugin(self, name: str) -> None:
        """
        Cleanup and remove a loaded plugin.

        Args:
            name: Loaded plugin name.

        Raises:
            ValueError: If the plugin is not loaded.
        """
        if name not in self._instances:
            raise ValueError(f"Plugin '{name}' is not loaded")
        instance = self._instances.pop(name)
        instance.shutdown()
        logger.info("Unloaded plugin: %s", name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all registered plugins with their load status.

        Returns:
            List of dicts with keys: name, class, loaded, instance.
        """
        result = []
        for name, plugin_class in self._registry.items():
            loaded = name in self._instances
            result.append({
                "name": name,
                "class": plugin_class.__name__,
                "loaded": loaded,
                "instance": self._instances.get(name),
            })
        return result

    def get_plugin(self, name: str) -> Optional[MnemosynePlugin]:
        """Return a loaded plugin instance, or None if not loaded.

        Lazy-loads registered-but-unloaded plugins on first access so callers
        that hold a reference can test `.enabled` without a separate load call.
        """
        if name in self._instances:
            return self._instances[name]
        if name in self._registry:
            return self.load_plugin(name)
        return None

    def is_loaded(self, name: str) -> bool:
        """Check if a plugin is currently loaded."""
        return name in self._instances

    def is_registered(self, name: str) -> bool:
        """Check if a plugin class is registered."""
        return name in self._registry

    def load_all(self, configs: Dict[str, Dict[str, Any]] = None) -> List[MnemosynePlugin]:
        """
        Load all registered plugins.

        Args:
            configs: Optional mapping of plugin name -> config dict.

        Returns:
            List of loaded plugin instances.
        """
        configs = configs or {}
        loaded = []
        for name in self._registry:
            if name not in self._instances:
                instance = self.load_plugin(name, config=configs.get(name, {}))
                loaded.append(instance)
        return loaded

    def unload_all(self) -> None:
        """Unload all loaded plugins."""
        for name in list(self._instances.keys()):
            self.unload_plugin(name)

    def discover_plugins(self) -> List[str]:
        """
        Discover and register plugins from the plugin directory.

        Scans ~/.hermes/mnemosyne/plugins/ for Python files and
        registers any MnemosynePlugin subclasses found.

        Returns:
            List of newly registered plugin names.
        """
        discovered: List[str] = []
        if not self._plugin_dir.exists():
            return discovered

        for file_path in self._plugin_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    file_path.stem, str(file_path)
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[file_path.stem] = module
                spec.loader.exec_module(module)

                for attr_name in dir(module):
                    obj = getattr(module, attr_name)
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, MnemosynePlugin)
                        and obj is not MnemosynePlugin
                        and not obj.__name__.startswith("_")
                    ):
                        plugin_name = getattr(obj, "name", None) or obj.__name__.lower()
                        if plugin_name not in self._registry:
                            self.register_plugin(plugin_name, obj)
                            discovered.append(plugin_name)
            except Exception as exc:
                logger.warning("Failed to load plugin from %s: %s", file_path, exc)

        return discovered

    def notify_remember(self, memory: Dict[str, Any]) -> None:
        """Notify all loaded plugins of a remember event."""
        for instance in self._instances.values():
            if instance.enabled:
                try:
                    instance.on_remember(memory)
                except Exception as exc:
                    logger.error("Plugin %s on_remember error: %s", instance.name, exc)

    def notify_recall(self, memory: Dict[str, Any]) -> None:
        """Notify all loaded plugins of a recall event."""
        for instance in self._instances.values():
            if instance.enabled:
                try:
                    instance.on_recall(memory)
                except Exception as exc:
                    logger.error("Plugin %s on_recall error: %s", instance.name, exc)

    def notify_consolidate(self, summary: Dict[str, Any]) -> None:
        """Notify all loaded plugins of a consolidate event."""
        for instance in self._instances.values():
            if instance.enabled:
                try:
                    instance.on_consolidate(summary)
                except Exception as exc:
                    logger.error("Plugin %s on_consolidate error: %s", instance.name, exc)

    def notify_invalidate(self, memory_id: str) -> None:
        """Notify all loaded plugins of an invalidate event."""
        for instance in self._instances.values():
            if instance.enabled:
                try:
                    instance.on_invalidate(memory_id)
                except Exception as exc:
                    logger.error("Plugin %s on_invalidate error: %s", instance.name, exc)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_all()
        return False


# Global plugin manager instance for convenience
_default_manager: Optional[PluginManager] = None


def get_manager() -> PluginManager:
    """Get or create the global PluginManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = PluginManager()
    return _default_manager


def reset_manager() -> None:
    """Reset the global PluginManager (useful in tests)."""
    global _default_manager
    if _default_manager is not None:
        _default_manager.unload_all()
    _default_manager = None
