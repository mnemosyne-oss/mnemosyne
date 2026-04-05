"""
Mnemosyne Plugin Tools for Hermes

Tool implementations that wrap Mnemosyne core functionality.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
plugin_dir = Path(__file__).parent
sys.path.insert(0, str(plugin_dir.parent))

from mnemosyne.core.memory import Mnemosyne

# Global memory instance
_memory_instance = None


def _get_memory():
    """Get or create global memory instance"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = Mnemosyne(session_id="hermes_default")
    return _memory_instance


# Tool Schemas (for Hermes tool registration)
REMEMBER_SCHEMA = {
    "name": "mnemosyne_remember",
    "description": "Store a memory in Mnemosyne local database. Use for important facts, preferences, or context to remember later.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The information to remember"
            },
            "importance": {
                "type": "number",
                "description": "Importance from 0.0 to 1.0 (0.9+ for critical facts)"
            },
            "source": {
                "type": "string",
                "description": "Source of the memory (preference, fact, conversation, etc.)"
            }
        },
        "required": ["content"]
    }
}

RECALL_SCHEMA = {
    "name": "mnemosyne_recall",
    "description": "Search memories in Mnemosyne. Use to recall previous context or facts about the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5
            }
        },
        "required": ["query"]
    }
}

STATS_SCHEMA = {
    "name": "mnemosyne_stats",
    "description": "Get Mnemosyne memory statistics",
    "parameters": {
        "type": "object",
        "properties": {}
    }
}


# Tool Handlers
def mnemosyne_remember(args: dict, **kwargs) -> str:
    """Store a memory"""
    try:
        content = args.get("content", "").strip()
        importance = args.get("importance", 0.5)
        source = args.get("source", "conversation")
        
        if not content:
            return json.dumps({"error": "Content is required"})
        
        mem = _get_memory()
        memory_id = mem.remember(content, source=source, importance=importance)
        
        return json.dumps({
            "status": "stored",
            "id": memory_id,
            "content_preview": content[:80] + "..." if len(content) > 80 else content
        })
        
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_recall(args: dict, **kwargs) -> str:
    """Search memories"""
    try:
        query = args.get("query", "").strip()
        top_k = args.get("top_k", 5)
        
        if not query:
            return json.dumps({"error": "Query is required"})
        
        mem = _get_memory()
        results = mem.recall(query, top_k=top_k)
        
        return json.dumps({
            "query": query,
            "results_count": len(results),
            "results": results
        })
        
    except Exception as e:
        return json.dumps({"error": str(e)})


def mnemosyne_stats(args: dict, **kwargs) -> str:
    """Get memory statistics"""
    try:
        mem = _get_memory()
        stats = mem.get_stats()
        
        return json.dumps(stats)
        
    except Exception as e:
        return json.dumps({"error": str(e)})
