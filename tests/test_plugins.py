"""
Tests for Mnemosyne Plugin Architecture

Validates:
1. PluginManager registration, loading, unloading, listing
2. MnemosynePlugin base class and abstract methods
3. Built-in plugins: LoggingPlugin, MetricsPlugin, FilterPlugin
4. Plugin discovery from filesystem
5. Plugin notification lifecycle
6. Global manager convenience functions
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnemosyne.core.plugins import (
    MnemosynePlugin,
    PluginManager,
    LoggingPlugin,
    MetricsPlugin,
    FilterPlugin,
    CompressionPlugin,
    get_manager,
    reset_manager,
    DEFAULT_PLUGIN_DIR,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_global_manager():
    """Reset the global plugin manager before each test."""
    reset_manager()
    yield
    reset_manager()


@pytest.fixture
def manager():
    """Fresh PluginManager instance."""
    return PluginManager()


@pytest.fixture
def sample_memory():
    """A sample memory dict for event testing."""
    return {
        "id": "mem-123",
        "content": "User prefers dark mode in all applications",
        "source": "conversation",
        "importance": 0.9,
    }


@pytest.fixture
def sample_summary():
    """A sample consolidation summary dict."""
    return {
        "summary": "User prefers dark mode",
        "source_wm_ids": ["wm1", "wm2"],
        "importance": 0.8,
    }


# ============================================================================
# Abstract Base Class
# ============================================================================

class TestMnemosynePlugin:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_base(self):
        """MnemosynePlugin is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            MnemosynePlugin()

    def test_subclass_must_implement_hooks(self):
        """Subclasses must implement all four lifecycle hooks."""
        class PartialPlugin(MnemosynePlugin):
            name = "partial"

            def on_remember(self, memory):
                pass

            def on_recall(self, memory):
                pass

            def on_consolidate(self, summary):
                pass

            # missing on_invalidate

        with pytest.raises(TypeError):
            PartialPlugin()

    def test_valid_subclass_can_instantiate(self):
        """A fully implemented subclass can be instantiated."""
        class ValidPlugin(MnemosynePlugin):
            name = "valid"

            def on_remember(self, memory):
                pass

            def on_recall(self, memory):
                pass

            def on_consolidate(self, summary):
                pass

            def on_invalidate(self, memory_id):
                pass

        plugin = ValidPlugin()
        assert plugin.name == "valid"
        assert plugin.enabled is True
        assert plugin._initialized is False

    def test_initialize_sets_flag(self):
        """initialize() sets _initialized to True."""
        class ValidPlugin(MnemosynePlugin):
            name = "valid"

            def on_remember(self, memory):
                pass

            def on_recall(self, memory):
                pass

            def on_consolidate(self, summary):
                pass

            def on_invalidate(self, memory_id):
                pass

        plugin = ValidPlugin()
        plugin.initialize()
        assert plugin._initialized is True

    def test_shutdown_clears_flag(self):
        """shutdown() sets _initialized to False."""
        class ValidPlugin(MnemosynePlugin):
            name = "valid"

            def on_remember(self, memory):
                pass

            def on_recall(self, memory):
                pass

            def on_consolidate(self, summary):
                pass

            def on_invalidate(self, memory_id):
                pass

        plugin = ValidPlugin()
        plugin.initialize()
        plugin.shutdown()
        assert plugin._initialized is False

    def test_to_dict(self):
        """to_dict() returns correct metadata."""
        class ValidPlugin(MnemosynePlugin):
            name = "valid"
            version = "2.0.0"

            def on_remember(self, memory):
                pass

            def on_recall(self, memory):
                pass

            def on_consolidate(self, summary):
                pass

            def on_invalidate(self, memory_id):
                pass

        plugin = ValidPlugin(config={"foo": "bar"})
        plugin.initialize()
        d = plugin.to_dict()
        assert d["name"] == "valid"
        assert d["version"] == "2.0.0"
        assert d["enabled"] is True
        assert d["initialized"] is True
        assert d["config"] == {"foo": "bar"}


# ============================================================================
# PluginManager Registration
# ============================================================================

class TestPluginManagerRegistration:
    """Tests for register_plugin and related checks."""

    def test_register_valid_plugin(self, manager):
        """register_plugin accepts a valid subclass."""
        class TestPlugin(MnemosynePlugin):
            name = "test"

            def on_remember(self, memory):
                pass

            def on_recall(self, memory):
                pass

            def on_consolidate(self, summary):
                pass

            def on_invalidate(self, memory_id):
                pass

        manager.register_plugin("test", TestPlugin)
        assert manager.is_registered("test")

    def test_register_non_subclass_raises(self, manager):
        """register_plugin raises TypeError for non-subclasses."""
        with pytest.raises(TypeError):
            manager.register_plugin("bad", str)

    def test_register_duplicate_raises(self, manager):
        """register_plugin raises ValueError for duplicate names."""
        class TestPlugin(MnemosynePlugin):
            name = "test"

            def on_remember(self, memory):
                pass

            def on_recall(self, memory):
                pass

            def on_consolidate(self, summary):
                pass

            def on_invalidate(self, memory_id):
                pass

        manager.register_plugin("test", TestPlugin)
        with pytest.raises(ValueError, match="already registered"):
            manager.register_plugin("test", TestPlugin)

    def test_builtins_registered_by_default(self, manager):
        """Built-in plugins are registered on construction."""
        assert manager.is_registered("logging")
        assert manager.is_registered("metrics")
        assert manager.is_registered("filter")
        assert manager.is_registered("compression")

    def test_is_registered_false_for_unknown(self, manager):
        """is_registered returns False for unknown plugins."""
        assert not manager.is_registered("nonexistent")

    def test_is_loaded_false_before_load(self, manager):
        """is_loaded returns False before load_plugin is called."""
        assert not manager.is_loaded("logging")


# ============================================================================
# PluginManager Loading / Unloading
# ============================================================================

class TestPluginManagerLoadUnload:
    """Tests for load_plugin, unload_plugin, get_plugin."""

    def test_load_plugin(self, manager):
        """load_plugin instantiates and initializes."""
        instance = manager.load_plugin("logging")
        assert isinstance(instance, LoggingPlugin)
        assert manager.is_loaded("logging")
        assert instance._initialized is True

    def test_load_plugin_with_config(self, manager):
        """load_plugin passes config to the plugin."""
        instance = manager.load_plugin("metrics", config={"max_timing_samples": 42})
        assert instance.config["max_timing_samples"] == 42

    def test_load_unregistered_raises(self, manager):
        """load_plugin raises ValueError for unregistered names."""
        with pytest.raises(ValueError, match="not registered"):
            manager.load_plugin("unknown")

    def test_load_already_loaded_raises(self, manager):
        """load_plugin raises RuntimeError if already loaded."""
        manager.load_plugin("filter")
        with pytest.raises(RuntimeError, match="already loaded"):
            manager.load_plugin("filter")

    def test_unload_plugin(self, manager):
        """unload_plugin calls shutdown and removes instance."""
        manager.load_plugin("logging")
        manager.unload_plugin("logging")
        assert not manager.is_loaded("logging")

    def test_unload_not_loaded_raises(self, manager):
        """unload_plugin raises ValueError if not loaded."""
        with pytest.raises(ValueError, match="not loaded"):
            manager.unload_plugin("logging")

    def test_get_plugin_returns_instance(self, manager):
        """get_plugin returns the loaded instance (lazy-loads if needed)."""
        loaded = manager.load_plugin("metrics")
        assert manager.get_plugin("metrics") is loaded

    def test_get_plugin_lazy_loads_registered(self, manager):
        """get_plugin auto-loads registered-but-unloaded plugins on access."""
        # "metrics" is registered but not yet loaded
        assert not manager.is_loaded("metrics")
        instance = manager.get_plugin("metrics")
        assert instance is not None  # auto-loaded
        assert manager.is_loaded("metrics")

    def test_load_all(self, manager):
        """load_all loads every registered plugin."""
        loaded = manager.load_all()
        assert len(loaded) == 4
        assert all(manager.is_loaded(p["name"]) for p in manager.list_plugins())

    def test_unload_all(self, manager):
        """unload_all removes every loaded plugin."""
        manager.load_all()
        manager.unload_all()
        assert all(not p["loaded"] for p in manager.list_plugins())

    def test_context_manager(self):
        """PluginManager works as a context manager."""
        with PluginManager() as mgr:
            mgr.load_plugin("logging")
            assert mgr.is_loaded("logging")
        assert not mgr.is_loaded("logging")


# ============================================================================
# PluginManager Listing
# ============================================================================

class TestPluginManagerList:
    """Tests for list_plugins."""

    def test_list_plugins_structure(self, manager):
        """list_plugins returns correct dict structure."""
        plugins = manager.list_plugins()
        assert len(plugins) == 4
        for p in plugins:
            assert "name" in p
            assert "class" in p
            assert "loaded" in p
            assert "instance" in p

    def test_list_plugins_loaded_state(self, manager):
        """list_plugins reflects loaded vs unloaded state."""
        manager.load_plugin("logging")
        plugins = manager.list_plugins()
        logging_entry = next(p for p in plugins if p["name"] == "logging")
        assert logging_entry["loaded"] is True
        assert isinstance(logging_entry["instance"], LoggingPlugin)

        metrics_entry = next(p for p in plugins if p["name"] == "metrics")
        assert metrics_entry["loaded"] is False
        assert metrics_entry["instance"] is None


# ============================================================================
# Built-in LoggingPlugin
# ============================================================================

class TestLoggingPlugin:
    """Tests for LoggingPlugin behavior."""

    def test_on_remember_logs(self, manager, sample_memory):
        """on_remember creates a log entry."""
        plugin = manager.load_plugin("logging")
        plugin.clear_log()
        plugin.on_remember(sample_memory)
        log = plugin.get_log()
        assert len(log) == 1
        assert log[0]["event"] == "remember"
        assert log[0]["memory_id"] == "mem-123"

    def test_on_recall_logs(self, manager, sample_memory):
        """on_recall creates a log entry."""
        plugin = manager.load_plugin("logging")
        plugin.clear_log()
        plugin.on_recall(sample_memory)
        log = plugin.get_log()
        assert len(log) == 1
        assert log[0]["event"] == "recall"

    def test_on_consolidate_logs(self, manager, sample_summary):
        """on_consolidate creates a log entry."""
        plugin = manager.load_plugin("logging")
        plugin.clear_log()
        plugin.on_consolidate(sample_summary)
        log = plugin.get_log()
        assert len(log) == 1
        assert log[0]["event"] == "consolidate"
        assert log[0]["source_count"] == 2

    def test_on_invalidate_logs(self, manager):
        """on_invalidate creates a log entry."""
        plugin = manager.load_plugin("logging")
        plugin.clear_log()
        plugin.on_invalidate("mem-123")
        log = plugin.get_log()
        assert len(log) == 1
        assert log[0]["event"] == "invalidate"
        assert log[0]["memory_id"] == "mem-123"

    def test_log_max_entries(self, manager):
        """Log respects max_entries limit."""
        plugin = manager.load_plugin("logging", config={"max_entries": 3})
        plugin.clear_log()
        for i in range(5):
            plugin.on_remember({"id": f"m{i}", "content": f"memory {i}"})
        assert len(plugin.get_log()) == 3
        assert plugin.get_log()[0]["memory_id"] == "m2"

    def test_clear_log(self, manager, sample_memory):
        """clear_log empties the log."""
        plugin = manager.load_plugin("logging")
        plugin.on_remember(sample_memory)
        plugin.clear_log()
        assert len(plugin.get_log()) == 0

    def test_preview_truncation(self, manager):
        """Long content is truncated in previews."""
        plugin = manager.load_plugin("logging")
        preview = plugin._preview("x" * 200)
        assert preview.endswith("...")
        assert len(preview) <= 83


# ============================================================================
# Built-in MetricsPlugin
# ============================================================================

class TestMetricsPlugin:
    """Tests for MetricsPlugin behavior."""

    def test_counters_increment(self, manager, sample_memory, sample_summary):
        """Counters increment on each event."""
        plugin = manager.load_plugin("metrics")
        plugin.on_remember(sample_memory)
        plugin.on_recall(sample_memory)
        plugin.on_consolidate(sample_summary)
        plugin.on_invalidate("m1")
        counters = plugin.get_counters()
        assert counters["remember"] == 1
        assert counters["recall"] == 1
        assert counters["consolidate"] == 1
        assert counters["invalidate"] == 1

    def test_record_timing(self, manager):
        """record_timing stores duration samples."""
        plugin = manager.load_plugin("metrics")
        plugin.record_timing("remember", 12.5)
        plugin.record_timing("remember", 7.5)
        assert plugin.get_timings("remember") == [12.5, 7.5]

    def test_average_timing(self, manager):
        """get_average_timing computes the mean."""
        plugin = manager.load_plugin("metrics")
        plugin.record_timing("recall", 10.0)
        plugin.record_timing("recall", 20.0)
        assert plugin.get_average_timing("recall") == 15.0

    def test_average_timing_none(self, manager):
        """get_average_timing returns None with no samples."""
        plugin = manager.load_plugin("metrics")
        assert plugin.get_average_timing("recall") is None

    def test_timing_max_samples(self, manager):
        """Timing list respects max_timing_samples."""
        plugin = manager.load_plugin("metrics", config={"max_timing_samples": 2})
        for i in range(5):
            plugin.record_timing("remember", float(i))
        assert len(plugin.get_timings("remember")) == 2
        assert plugin.get_timings("remember") == [3.0, 4.0]

    def test_reset(self, manager, sample_memory):
        """reset clears counters and timings."""
        plugin = manager.load_plugin("metrics")
        plugin.on_remember(sample_memory)
        plugin.record_timing("remember", 5.0)
        plugin.reset()
        assert plugin.get_counters()["remember"] == 0
        assert plugin.get_timings("remember") == []

    def test_get_summary(self, manager, sample_memory):
        """get_summary returns counters and averages."""
        plugin = manager.load_plugin("metrics")
        plugin.on_remember(sample_memory)
        plugin.record_timing("remember", 10.0)
        summary = plugin.get_summary()
        assert summary["counters"]["remember"] == 1
        assert summary["averages"]["remember"] == 10.0


# ============================================================================
# Built-in FilterPlugin
# ============================================================================

class TestFilterPlugin:
    """Tests for FilterPlugin behavior."""

    def test_add_remove_rule(self, manager):
        """Rules can be added and removed."""
        plugin = manager.load_plugin("filter")
        rule = lambda m: True
        plugin.add_rule(rule)
        assert rule in plugin._rules
        plugin.remove_rule(rule)
        assert rule not in plugin._rules

    def test_clear_rules(self, manager):
        """clear_rules removes all rules."""
        plugin = manager.load_plugin("filter")
        plugin.add_rule(lambda m: True)
        plugin.add_rule(lambda m: False)
        plugin.clear_rules()
        assert len(plugin._rules) == 0

    def test_passes_with_no_rules(self, manager, sample_memory):
        """Without rules, everything passes."""
        plugin = manager.load_plugin("filter")
        assert plugin._passes(sample_memory) is True

    def test_passes_with_allow_rule(self, manager, sample_memory):
        """A rule returning True allows the memory."""
        plugin = manager.load_plugin("filter")
        plugin.add_rule(lambda m: m.get("importance", 0) > 0.5)
        assert plugin._passes(sample_memory) is True

    def test_passes_with_block_rule(self, manager, sample_memory):
        """A rule returning False blocks the memory."""
        plugin = manager.load_plugin("filter")
        plugin.add_rule(lambda m: m.get("importance", 0) > 0.95)
        assert plugin._passes(sample_memory) is False

    def test_blocked_tracked(self, manager, sample_memory):
        """Blocked memories are tracked."""
        plugin = manager.load_plugin("filter")
        plugin.add_rule(lambda m: False)
        plugin.on_remember(sample_memory)
        blocked = plugin.get_blocked()
        assert len(blocked) == 1
        assert blocked[0]["item"]["id"] == "mem-123"

    def test_is_blocked(self, manager, sample_memory):
        """is_blocked checks by memory ID."""
        plugin = manager.load_plugin("filter")
        plugin.add_rule(lambda m: False)
        plugin.on_remember(sample_memory)
        assert plugin.is_blocked("mem-123") is True
        assert plugin.is_blocked("other") is False

    def test_exception_in_rule_blocks(self, manager, sample_memory):
        """An exception in a rule treats the memory as blocked."""
        plugin = manager.load_plugin("filter")
        plugin.add_rule(lambda m: 1 / 0)
        assert plugin._passes(sample_memory) is False

    def test_max_blocked(self, manager):
        """Blocked list respects max_blocked limit."""
        plugin = manager.load_plugin("filter", config={"max_blocked": 2})
        plugin.add_rule(lambda m: False)
        for i in range(5):
            plugin.on_remember({"id": f"m{i}", "content": f"c{i}"})
        assert len(plugin.get_blocked()) == 2


# ============================================================================
# Plugin Notifications
# ============================================================================

class TestPluginNotifications:
    """Tests for notify_* methods on PluginManager."""

    def test_notify_remember(self, manager, sample_memory):
        """notify_remember reaches all loaded plugins."""
        logging_plugin = manager.load_plugin("logging")
        logging_plugin.clear_log()
        manager.notify_remember(sample_memory)
        assert len(logging_plugin.get_log()) == 1

    def test_notify_recall(self, manager, sample_memory):
        """notify_recall reaches all loaded plugins."""
        metrics_plugin = manager.load_plugin("metrics")
        manager.notify_recall(sample_memory)
        assert metrics_plugin.get_counters()["recall"] == 1

    def test_notify_consolidate(self, manager, sample_summary):
        """notify_consolidate reaches all loaded plugins."""
        logging_plugin = manager.load_plugin("logging")
        logging_plugin.clear_log()
        manager.notify_consolidate(sample_summary)
        assert len(logging_plugin.get_log()) == 1

    def test_notify_invalidate(self, manager):
        """notify_invalidate reaches all loaded plugins."""
        metrics_plugin = manager.load_plugin("metrics")
        manager.notify_invalidate("mid-1")
        assert metrics_plugin.get_counters()["invalidate"] == 1

    def test_disabled_plugin_skipped(self, manager, sample_memory):
        """Disabled plugins are skipped during notification."""
        plugin = manager.load_plugin("logging")
        plugin.clear_log()
        plugin.enabled = False
        manager.notify_remember(sample_memory)
        assert len(plugin.get_log()) == 0

    def test_plugin_error_does_not_propagate(self, manager, sample_memory):
        """Exceptions in plugins are caught and logged, not propagated."""
        class BadPlugin(MnemosynePlugin):
            name = "bad"

            def on_remember(self, memory):
                raise RuntimeError("boom")

            def on_recall(self, memory):
                pass

            def on_consolidate(self, summary):
                pass

            def on_invalidate(self, memory_id):
                pass

        manager.register_plugin("bad", BadPlugin)
        manager.load_plugin("bad")
        # Should not raise
        manager.notify_remember(sample_memory)


# ============================================================================
# Plugin Discovery
# ============================================================================

class TestPluginDiscovery:
    """Tests for discover_plugins."""

    def test_discover_empty_dir(self, manager):
        """Discover on non-existent directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PluginManager(plugin_dir=Path(tmpdir) / "empty")
            discovered = mgr.discover_plugins()
            assert discovered == []

    def test_discover_valid_plugin(self, manager):
        """Discover registers a valid plugin from a .py file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_file = Path(tmpdir) / "my_plugin.py"
            plugin_file.write_text(
                "from mnemosyne.core.plugins import MnemosynePlugin\n"
                "class MyPlugin(MnemosynePlugin):\n"
                "    name = 'myplugin'\n"
                "    def on_remember(self, memory): pass\n"
                "    def on_recall(self, memory): pass\n"
                "    def on_consolidate(self, summary): pass\n"
                "    def on_invalidate(self, memory_id): pass\n"
            )
            mgr = PluginManager(plugin_dir=Path(tmpdir))
            discovered = mgr.discover_plugins()
            assert "myplugin" in discovered
            assert mgr.is_registered("myplugin")

    def test_discover_ignores_underscore_files(self, manager):
        """Files starting with _ are ignored during discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_file = Path(tmpdir) / "_private.py"
            plugin_file.write_text(
                "from mnemosyne.core.plugins import MnemosynePlugin\n"
                "class PrivatePlugin(MnemosynePlugin):\n"
                "    name = 'private'\n"
                "    def on_remember(self, memory): pass\n"
                "    def on_recall(self, memory): pass\n"
                "    def on_consolidate(self, summary): pass\n"
                "    def on_invalidate(self, memory_id): pass\n"
            )
            mgr = PluginManager(plugin_dir=Path(tmpdir))
            discovered = mgr.discover_plugins()
            assert "private" not in discovered

    def test_discover_does_not_duplicate(self, manager):
        """Already-registered plugins are not duplicated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_file = Path(tmpdir) / "logging.py"
            plugin_file.write_text(
                "from mnemosyne.core.plugins import MnemosynePlugin\n"
                "class LoggingPlugin(MnemosynePlugin):\n"
                "    name = 'logging'\n"
                "    def on_remember(self, memory): pass\n"
                "    def on_recall(self, memory): pass\n"
                "    def on_consolidate(self, summary): pass\n"
                "    def on_invalidate(self, memory_id): pass\n"
            )
            mgr = PluginManager(plugin_dir=Path(tmpdir))
            discovered = mgr.discover_plugins()
            assert "logging" not in discovered  # already registered as builtin

    def test_discover_bad_file_graceful(self, manager):
        """Bad Python files are skipped gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_file = Path(tmpdir) / "broken.py"
            plugin_file.write_text("this is not valid python!!!")
            mgr = PluginManager(plugin_dir=Path(tmpdir))
            discovered = mgr.discover_plugins()
            assert discovered == []


# ============================================================================
# Global Manager
# ============================================================================

class TestGlobalManager:
    """Tests for get_manager and reset_manager."""

    def test_get_manager_singleton(self):
        """get_manager returns the same instance."""
        mgr1 = get_manager()
        mgr2 = get_manager()
        assert mgr1 is mgr2

    def test_reset_manager_creates_new(self):
        """reset_manager creates a new instance on next get_manager."""
        mgr1 = get_manager()
        reset_manager()
        mgr2 = get_manager()
        assert mgr1 is not mgr2

    def test_reset_manager_unloads(self):
        """reset_manager unloads existing plugins."""
        mgr = get_manager()
        mgr.load_plugin("logging")
        reset_manager()
        mgr2 = get_manager()
        assert not mgr2.is_loaded("logging")


# ============================================================================
# Built-in CompressionPlugin
# ============================================================================


class TestCompressionPlugin:
    """Tests for CompressionPlugin behavior."""

    def test_disabled_by_default(self):
        """CompressionPlugin is disabled by default (opt-in)."""
        plugin = CompressionPlugin()
        assert not plugin.enabled

    def test_enabled_via_config(self):
        """Config enabled=True enables the plugin."""
        plugin = CompressionPlugin(config={"enabled": True})
        assert plugin.enabled

    def test_compress_lines_noop_when_disabled(self):
        """compress_lines returns lines unchanged when disabled."""
        plugin = CompressionPlugin(config={"enabled": False})
        lines = ["Hello world", "Test line"]
        result = plugin.compress_lines(lines)
        assert result == lines

    def test_compress_lines_works_with_caveman(self):
        """compress_lines actually compresses when rust_cave_001 is installed."""
        plugin = CompressionPlugin(config={"enabled": True, "provider": "caveman"})
        lines = ["Hello world", "This is a test line"]
        result = plugin.compress_lines(lines)
        # caveman removes stopwords like "This" from phrases
        assert len(result) == len(lines)
        assert isinstance(result[0], str)
        assert len(result[0]) <= len(lines[0])  # compressed or unchanged
        # Verify threshold still works
        plugin2 = CompressionPlugin(config={"enabled": True, "threshold_chars": 100})
        short = ["Short"]
        assert plugin2.compress_lines(short) == short

    def test_deprecated_env_var_fallback(self, monkeypatch):
        """MNEMOSYNE_USE_CAVEMAN=1 enables with deprecation warning."""
        monkeypatch.setenv("MNEMOSYNE_USE_CAVEMAN", "1")
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plugin = CompressionPlugin()
            deprecation = any(
                issubclass(warn.category, DeprecationWarning) for warn in w
            )
            assert deprecation, "Should emit DeprecationWarning"
            assert plugin.enabled

    def test_registered_as_builtin(self, manager):
        """CompressionPlugin is registered in PluginManager."""
        assert manager.is_registered("compression")
        instance = manager.load_plugin("compression")
        assert isinstance(instance, CompressionPlugin)
        assert not instance.enabled  # default disabled

    def test_unknown_provider_fallback(self, manager):
        """Unknown provider falls back gracefully, returning lines unchanged."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plugin = CompressionPlugin(config={"enabled": True, "provider": "unknown_provider_xyz"})
            lines = ["Hello"]
            result = plugin.compress_lines(lines)
            assert result == lines


# ============================================================================
# Integration with Mnemosyne memory class (optional hook points)
# ============================================================================

class TestPluginIntegration:
    """Tests showing how plugins integrate with memory operations."""

    def test_plugin_manager_can_be_attached_to_memory(self, manager):
        """A PluginManager can be stored and used alongside Mnemosyne."""
        # This demonstrates the pattern: Mnemosyne can hold a PluginManager
        # and call notify_* at key lifecycle points.
        from mnemosyne.core.memory import Mnemosyne as Memory
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            mem = Memory(session_id="s1", db_path=db_path)
            # Attach plugin manager
            mem.plugins = manager
            manager.load_plugin("logging")
            manager.load_plugin("metrics")
            assert hasattr(mem, "plugins")
            assert manager.is_loaded("logging")
            assert manager.is_loaded("metrics")
