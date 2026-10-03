"""Public plugin discovery regression tests for module identity and rollback."""
import importlib
import importlib.util
import logging
from pathlib import Path
import sys

import pytest

from mnemosyne.core.plugins import MnemosynePlugin, PluginManager, _plugin_module_key


@pytest.fixture(autouse=True)
def module_cleanup():
    """Restore exactly the module entries owned by these discovery fixtures."""
    before = dict(sys.modules)
    yield
    keys = {k for k in sys.modules if k.startswith('_mnemosyne_plugin_')}
    keys.update(k for k in before if k.startswith('_mnemosyne_plugin_'))
    # Include bare fixture stems so a broken loader cannot poison later tests.
    keys.update(('logging', 'a-b', 'a_b', 'shared', 'plugin', 'alias', 'unicode-λ'))
    for key in keys:
        if key in before:
            sys.modules[key] = before[key]
        else:
            sys.modules.pop(key, None)


def source(name):
    """Return a minimal plugin whose defining module must remain importable."""
    return (
        'from mnemosyne.core.plugins import MnemosynePlugin\n'
        'class ExamplePlugin(MnemosynePlugin):\n'
        f'    name = {name!r}\n'
        '    def on_remember(self, memory): pass\n'
        '    def on_recall(self, query, results): pass\n'
        '    def on_consolidate(self, summary): pass\n'
        '    def on_invalidate(self, memory_id): pass\n'
    )


def store(path, content):
    """Write a fresh fixture and remove bytecode before rediscovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    cache = Path(importlib.util.cache_from_source(str(path)))
    cache.unlink(missing_ok=True)
    return path


def install_preexisting(path, manager):
    """An externally registered module has not been cached by discovery."""
    key = _plugin_module_key(path)
    spec = importlib.util.spec_from_file_location(key, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    manager.register_plugin(module.ExamplePlugin.name, module.ExamplePlugin)
    return module


def test_stdlib_module_survives_discovery(tmp_path):
    """A plugin called logging.py must not replace the real logging module."""
    p = store(tmp_path / 'logging.py', source('fixture'))
    manager = PluginManager(plugin_dir=tmp_path)
    assert 'fixture' in manager.discover_plugins()
    assert sys.modules['logging'] is logging
    module = importlib.import_module(_plugin_module_key(p))
    assert manager._registry['fixture'] is module.ExamplePlugin
    assert module.ExamplePlugin.__module__ == module.__spec__.name == module.__name__


@pytest.mark.parametrize('names', [('a-b.py', 'a_b.py'), ('left/shared.py', 'right/shared.py'), ('unicode-λ.py', 'plugin.py')])
def test_distinct_paths_have_distinct_importable_modules(tmp_path, names):
    """Punctuation, directory and Unicode differences cannot collapse keys."""
    a, b = (store(tmp_path / n, source(label)) for n, label in zip(names, ('first', 'second')))
    ma, mb = PluginManager(plugin_dir=a.parent), PluginManager(plugin_dir=b.parent)
    assert 'first' in ma.discover_plugins()
    assert 'second' in mb.discover_plugins()
    ka, kb = _plugin_module_key(a), _plugin_module_key(b)
    assert ka != kb
    first, second = sys.modules[ka], sys.modules[kb]
    assert first.ExamplePlugin.name == 'first'
    assert second.ExamplePlugin.name == 'second'
    store(b, "raise RuntimeError('rediscovery failure')\n")
    mb.discover_plugins()
    assert sys.modules[kb] is second
    # Separate directories must never disturb each other's successful load.
    if a.parent != b.parent:
        assert sys.modules[ka] is first


def test_canonical_aliases_share_identity(tmp_path, monkeypatch):
    """Relative paths and symlinks resolve to the same canonical identity."""
    p = store(tmp_path / 'plugin.py', source('fixture'))
    monkeypatch.chdir(tmp_path)
    assert _plugin_module_key(Path('plugin.py')) == _plugin_module_key(p)
    alias = tmp_path / 'alias.py'
    try:
        alias.symlink_to(p)
    except OSError:
        pytest.skip('symlink creation unavailable')
    assert _plugin_module_key(alias) == _plugin_module_key(p)


FAILURES = {
    'exec': "raise RuntimeError('execution failed')\n",
    'dir': source('partial') + "def __dir__(): raise RuntimeError('inspection failed')\n",
    'getattr': source('partial') + "def __dir__(): return ['ExamplePlugin', 'missing']\ndef __getattr__(name): raise RuntimeError('attribute failed')\n",
    'interrupt': "raise KeyboardInterrupt('interrupted import')\n",
    'exit': "raise SystemExit(7)\n",
}


@pytest.mark.parametrize('failure', FAILURES)
@pytest.mark.parametrize('prior', ['absent', 'none', 'working'])
def test_failed_load_restores_exact_previous_state(tmp_path, failure, prior):
    """Execution and inspection failures restore absent, None or live entries."""
    p = store(tmp_path / 'plugin.py', source('fixture'))
    manager = PluginManager(plugin_dir=tmp_path)
    key = _plugin_module_key(p)
    if prior == 'working':
        install_preexisting(p, manager)
    elif prior == 'none':
        sys.modules[key] = None
    previous = sys.modules.get(key)
    registry = dict(manager._registry)
    store(p, FAILURES[failure])
    if failure in ('interrupt', 'exit'):
        with pytest.raises(KeyboardInterrupt if failure == 'interrupt' else SystemExit):
            manager.discover_plugins()
    else:
        assert manager.discover_plugins() == []
    assert (key in sys.modules) == (prior != 'absent')
    assert sys.modules.get(key) is previous
    assert manager._registry == registry


@pytest.mark.parametrize('remove', [False, True])
@pytest.mark.parametrize('phase', ['exec', 'scan'])
def test_failure_does_not_clobber_a_replacement(tmp_path, remove, phase):
    """A failed load may not restore over a newer module or deliberate deletion."""
    p = store(tmp_path / 'plugin.py', source('fixture'))
    manager = PluginManager(plugin_dir=tmp_path)
    install_preexisting(p, manager)
    key = _plugin_module_key(p)
    change = 'del sys.modules[__name__]' if remove else "sys.modules[__name__] = types.ModuleType('replacement')"
    code = 'import sys, types\n'
    if phase == 'scan':
        code += 'def __dir__():\n    ' + change + "\n    raise RuntimeError('failure')\n"
    else:
        code += change + "\nraise RuntimeError('failure')\n"
    store(p, code)
    assert manager.discover_plugins() == []
    if remove:
        assert key not in sys.modules
    else:
        assert sys.modules[key].__name__ == 'replacement'


def test_registration_failure_rolls_back_owned_entries(tmp_path, monkeypatch):
    """Partial registration must not leave classes backed by a failed module."""
    p = store(tmp_path / 'plugin.py', source('first') + 'class SecondPlugin(ExamplePlugin):\n    name = "second"\n')
    manager = PluginManager(plugin_dir=tmp_path)
    before = dict(manager._registry)
    register = manager.register_plugin

    def fail_second(name, cls):
        register(name, cls)
        if name == 'second':
            raise RuntimeError('registration failed')

    monkeypatch.setattr(manager, 'register_plugin', fail_second)
    assert manager.discover_plugins() == []
    assert manager._registry == before
    assert _plugin_module_key(p) not in sys.modules


def test_registry_replacement_during_failure_is_preserved(tmp_path, monkeypatch):
    """Rollback may remove only registry entries installed by the failed load."""
    p = store(tmp_path / 'plugin.py', source('fixture'))
    manager = PluginManager(plugin_dir=tmp_path)
    class Replacement(MnemosynePlugin):
        name = 'replacement'

    replacement = Replacement

    def replace_then_fail(name, cls):
        manager._registry[name] = replacement
        raise RuntimeError('registration failed after replacement')

    monkeypatch.setattr(manager, 'register_plugin', replace_then_fail)
    assert manager.discover_plugins() == []
    assert manager._registry['fixture'] is replacement
    assert _plugin_module_key(p) not in sys.modules


@pytest.mark.parametrize('changed', [False, True])
def test_successful_rediscovery_preserves_module_registry_and_instance(tmp_path, changed):
    p = store(tmp_path / 'plugin.py', source('fixture'))
    manager = PluginManager(plugin_dir=tmp_path)
    assert manager.discover_plugins() == ['fixture']
    key = _plugin_module_key(p)
    original = sys.modules[key]
    cls = manager._registry['fixture']
    if changed:
        store(p, "raise AssertionError('successful paths must not execute again')\n")
    assert manager.discover_plugins() == []
    assert sys.modules[key] is original
    assert original.ExamplePlugin is cls is manager._registry['fixture']
    assert type(manager.load_plugin('fixture')) is cls
    manager.unload_all()


def test_new_manager_reuses_canonical_module_and_registers_classes(tmp_path):
    p = store(tmp_path / 'plugin.py', source('fixture'))
    first = PluginManager(plugin_dir=tmp_path)
    assert first.discover_plugins() == ['fixture']
    module = sys.modules[_plugin_module_key(p)]
    store(p, "raise AssertionError('second manager must reuse successful module')\n")
    second = PluginManager(plugin_dir=tmp_path)
    assert second.discover_plugins() == ['fixture']
    assert sys.modules[_plugin_module_key(p)] is module
    assert second._registry['fixture'] is first._registry['fixture'] is module.ExamplePlugin
    assert type(second.load_plugin('fixture')) is module.ExamplePlugin
    second.unload_all()


def test_canonical_alias_discovery_executes_once(tmp_path):
    p = store(tmp_path / 'plugin.py', source('fixture'))
    alias = tmp_path / 'alias.py'
    try:
        alias.symlink_to(p)
    except OSError:
        pytest.skip('symlink creation unavailable')
    manager = PluginManager(plugin_dir=tmp_path)
    assert manager.discover_plugins() == ['fixture']
    module = sys.modules[_plugin_module_key(p)]
    assert manager._registry['fixture'] is module.ExamplePlugin
