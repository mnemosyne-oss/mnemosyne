---
name: Bug Report
about: Report a bug in Mnemosyne
title: '[BUG] '
labels: bug
assignees: ''
---

## Description

A clear and concise description of the bug.

## Environment

- **OS:** (e.g. Ubuntu 22.04, macOS 14, Windows 11)
- **Python version:** (output of `python --version`)
- **Mnemosyne version/commit:** (output of `git rev-parse --short HEAD` or version string)
- **Installation method:** `pip install -e .` / install script / manual clone
- **Hermes version:** (if using the Hermes plugin)

## Reproduction Steps

1. Step one
2. Step two
3. Step three

Minimal code or command that triggers the bug:

```python
# Paste minimal reproduction here
```

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened. Include full error messages, tracebacks, or output.

```
Paste logs / traceback here
```

## Mnemosyne Status

Please run the following and paste the output:

```bash
hermes mnemosyne stats
```

Or if using the Python API directly:

```python
from mnemosyne import get_stats
print(get_stats())
```

## Optional: Database Info

If relevant, include:

```bash
ls -la ~/.hermes/mnemosyne/data/
sqlite3 ~/.hermes/mnemosyne/data/mnemosyne.db "SELECT COUNT(*) FROM working_memory; SELECT COUNT(*) FROM episodic_memory;"
```

## Checklist

- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I can reproduce this bug consistently
- [ ] I have included the full traceback / error message
- [ ] I have included my environment details
