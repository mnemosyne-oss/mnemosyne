"""
Mnemosyne - The Zero-Dependency AI Memory System

A native, sub-millisecond memory system for AI agents using SQLite.
No HTTP, no servers, no API keys — just Python and SQLite.

Example:
    >>> from mnemosyne import remember, recall
    >>> remember("User prefers dark mode", importance=0.9)
    >>> results = recall("user preferences")
"""

__version__ = "1.0.0"
__author__ = "FluxSpeak AI"
__license__ = "MIT"

from .core.memory import (
    Mnemosyne,
    remember,
    recall,
    get_context,
    get_stats,
    forget,
    update,
)

__all__ = [
    "Mnemosyne",
    "remember",
    "recall",
    "get_context",
    "get_stats",
    "forget",
    "update",
]
