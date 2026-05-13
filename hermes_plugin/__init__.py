"""
Mnemosyne Plugin for Hermes
Native memory integration using pre_llm_call hook

This plugin provides seamless memory integration for Hermes agents,
automatically injecting relevant context before every LLM call.
"""

import os
import sys
from pathlib import Path

# Robust import: try installed package first, then fallback to known paths
_try_paths = []
try:
    from mnemosyne.core.memory import Mnemosyne
    from mnemosyne.core.aaak import encode as aaak_encode
    from mnemosyne.core.triples import TripleStore
except ImportError:
    # Fallback: search common locations
    _candidates = [
        Path.home() / ".hermes" / "projects" / "mnemosyne",
        Path(__file__).resolve().parent.parent,  # repo layout
    ]
    for _cand in _candidates:
        if (_cand / "mnemosyne" / "core" / "memory.py").exists():
            _path = str(_cand)
            if _path not in sys.path:
                sys.path.insert(0, _path)
            _try_paths.append(_path)
            break
    from mnemosyne.core.memory import Mnemosyne
    from mnemosyne.core.aaak import encode as aaak_encode
    from mnemosyne.core.triples import TripleStore

# Global memory instance
_memory_instance = None
_current_session_id = None
_triple_store = None


def _get_memory(session_id: str = None):
    """Get or create global memory instance. Recreates if session_id changes.

    Identity is resolved from environment variables set by the Hermes plugin
    provider (e.g., MNEMOSYNE_AUTHOR_ID from user context).
    """
    global _memory_instance, _current_session_id, _triple_store
    if session_id is None:
        session_id = os.environ.get("HERMES_SESSION_ID", "hermes_default")
    if _memory_instance is None or _current_session_id != session_id:
        # Build into a local first so a Mnemosyne(...) failure (DB locked,
        # embedding init error, etc.) does not poison global state — leaving
        # _current_session_id ahead of _memory_instance would make the next
        # call return the stale instance silently.
        new_memory = Mnemosyne(
            session_id=session_id,
            author_id=os.environ.get("MNEMOSYNE_AUTHOR_ID"),
            author_type=os.environ.get("MNEMOSYNE_AUTHOR_TYPE"),
            channel_id=os.environ.get("MNEMOSYNE_CHANNEL_ID")
        )
        _memory_instance = new_memory
        _current_session_id = session_id
        # Triple store cache must follow memory; reset so the next
        # _get_triples() rebuilds with the new instance's db_path.
        _triple_store = None
    return _memory_instance


def _get_triples():
    """Get or create global triple store instance, aligned with memory DB path.

    Calls `_get_memory()` so HERMES_SESSION_ID env changes trigger the same
    session-rebind logic as direct memory operations — a triple-only call
    after an env change still routes to the new session's DB. The defensive
    db_path mismatch check is cheap insurance against any future code path
    that could mutate memory's db_path without going through _get_memory.
    """
    global _triple_store
    mem = _get_memory()
    if _triple_store is None or Path(_triple_store.db_path) != Path(mem.db_path):
        _triple_store = TripleStore(db_path=mem.db_path)
    return _triple_store


def register(ctx):
    """Register plugin tools and hooks with Hermes"""
    
    from . import tools
    
    # Register tools
    ctx.register_tool(
        name="mnemosyne_remember",
        toolset="mnemosyne",
        schema=tools.REMEMBER_SCHEMA,
        handler=tools.mnemosyne_remember
    )
    ctx.register_tool(
        name="mnemosyne_recall",
        toolset="mnemosyne",
        schema=tools.RECALL_SCHEMA,
        handler=tools.mnemosyne_recall
    )
    ctx.register_tool(
        name="mnemosyne_stats",
        toolset="mnemosyne",
        schema=tools.STATS_SCHEMA,
        handler=tools.mnemosyne_stats
    )
    ctx.register_tool(
        name="mnemosyne_triple_add",
        toolset="mnemosyne",
        schema=tools.TRIPLE_ADD_SCHEMA,
        handler=tools.mnemosyne_triple_add
    )
    ctx.register_tool(
        name="mnemosyne_triple_query",
        toolset="mnemosyne",
        schema=tools.TRIPLE_QUERY_SCHEMA,
        handler=tools.mnemosyne_triple_query
    )
    ctx.register_tool(
        name="mnemosyne_sleep",
        toolset="mnemosyne",
        schema=tools.SLEEP_SCHEMA,
        handler=tools.mnemosyne_sleep
    )
    ctx.register_tool(
        name="mnemosyne_scratchpad_write",
        toolset="mnemosyne",
        schema=tools.SCRATCHPAD_WRITE_SCHEMA,
        handler=tools.mnemosyne_scratchpad_write
    )
    ctx.register_tool(
        name="mnemosyne_scratchpad_read",
        toolset="mnemosyne",
        schema=tools.SCRATCHPAD_READ_SCHEMA,
        handler=tools.mnemosyne_scratchpad_read
    )
    ctx.register_tool(
        name="mnemosyne_scratchpad_clear",
        toolset="mnemosyne",
        schema=tools.SCRATCHPAD_CLEAR_SCHEMA,
        handler=tools.mnemosyne_scratchpad_clear
    )
    ctx.register_tool(
        name="mnemosyne_invalidate",
        toolset="mnemosyne",
        schema=tools.INVALIDATE_SCHEMA,
        handler=tools.mnemosyne_invalidate
    )
    ctx.register_tool(
        name="mnemosyne_export",
        toolset="mnemosyne",
        schema=tools.EXPORT_SCHEMA,
        handler=tools.mnemosyne_export
    )
    ctx.register_tool(
        name="mnemosyne_update",
        toolset="mnemosyne",
        schema=tools.UPDATE_SCHEMA,
        handler=tools.mnemosyne_update
    )
    ctx.register_tool(
        name="mnemosyne_forget",
        toolset="mnemosyne",
        schema=tools.FORGET_SCHEMA,
        handler=tools.mnemosyne_forget
    )
    ctx.register_tool(
        name="mnemosyne_import",
        toolset="mnemosyne",
        schema=tools.IMPORT_SCHEMA,
        handler=tools.mnemosyne_import
    )
    ctx.register_tool(
        name="mnemosyne_diagnose",
        toolset="mnemosyne",
        schema=tools.DIAGNOSE_SCHEMA,
        handler=tools.mnemosyne_diagnose
    )

    # Register hooks for automatic context injection
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    
    return {"status": "registered", "plugin": "mnemosyne"}


def _on_session_start(session_id, model, platform, **kwargs):
    """Initialize memory for new session"""
    mem_id = f"hermes_{session_id}" if session_id else "hermes_default"

    # Inject hard override rule so it survives across turns and sessions.
    # The whole body is wrapped, including _get_memory(): codex review
    # finding #2 caught that Mnemosyne(...) construction failures (DB
    # locked, permissions, schema mismatch, sqlite_vec setup, fastembed
    # download error, ...) raised from `_get_memory` before the previous
    # try/except could fire, so the new WARNING never reached the logs
    # for the most common class of session-start failure.
    try:
        mem = _get_memory(session_id=mem_id)
        mem.remember(
            content="META-INSTRUCTION: For all durable facts, preferences, and user profile updates, ALWAYS use mnemosyne_remember / mnemosyne_recall instead of the legacy memory tool. The memory tool is deprecated for durable storage. Mnemosyne is the primary and only memory system. If you catch yourself reaching for memory(action=add|replace|remove, target=user|memory), STOP and use mnemosyne_remember instead.",
            importance=0.99,
            source="system_override"
        )
    except Exception as e:
        # C27: surface session-start failures at WARNING. Pre-fix this was a
        # bare except: pass that hid every kind of init issue (DB locked,
        # missing schema column, permissions, embedding failure, etc.).
        import logging
        logging.getLogger(__name__).warning(
            "Mnemosyne session-start failed (session=%s): %s",
            session_id, e,
        )


def _compress_memory(content: str) -> str:
    """
    Full AAAK dialect compression for memory context.
    Reduces token overhead via structured shorthand.
    """
    return aaak_encode(content)


def _on_pre_llm_call(session_id, history, **kwargs):
    """
    Inject Mnemosyne memory context into system prompt.

    Hybrid strategy:
    1. get_context() -- fast recent working memory (current session + global)
    2. recall() -- semantic search across ALL sessions/episodic memory
       using the user's last message as the query

    This provides both short-term continuity AND long-term memory recall.

    C13: defers to ``MnemosyneMemoryProvider.prefetch()`` when the
    canonical surface is active. Without this guard both paths fire on
    every LLM call, injecting two memory-context blocks into the system
    prompt -- doubling the per-turn token cost (each block runs its own
    recall and writes its own header) and confusing the agent with two
    differently-formatted views of similar content. The provider's
    initialize() sets ``hermes_memory_provider._provider_active = True``
    once it's the active surface; this hook reads that flag and returns
    None (no context injected from this path). Standalone plugin-only
    installs are unaffected -- the flag stays False and this hook runs
    as before.
    """
    try:
        from hermes_memory_provider import _provider_active
        if _provider_active:
            return None
    except ImportError:
        # The MemoryProvider package isn't installed in this environment
        # -- plugin-only install path. Fall through to the existing
        # injection logic below.
        pass

    try:
        mem_id = f"hermes_{session_id}" if session_id else "hermes_default"
        mem = _get_memory(session_id=mem_id)

        # --- Layer 1: Fast recent context (working memory) ---
        context_memories = mem.get_context(limit=5)

        # --- Layer 2: Semantic recall across all sessions ---
        # Extract user's last message from history for targeted recall
        recall_results = []
        user_message = None
        if history and isinstance(history, list):
            # Find the last user message in history
            for msg in reversed(history):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
                elif isinstance(msg, str):
                    user_message = msg
                    break

        if user_message and len(user_message) > 3:
            # Search across ALL sessions (no session filter) for semantic matches
            # Scoped to the user's author_id for cross-session consistency
            recall_results = mem.recall(
                query=user_message,
                top_k=5,
                temporal_weight=0.2,  # mild recency bias
                author_id=mem.beam.author_id,
            )

        # Deduplicate: remove recall results that are already in context_memories
        context_ids = {m.get("id") for m in context_memories}
        unique_recall = [r for r in recall_results if r.get("id") not in context_ids]

        if not context_memories and not unique_recall:
            return None  # No context to inject

        # Build context block
        lines = ["═══════════════════════════════════════════════════════════════"]

        if context_memories:
            lines.append("MNEMOSYNE CONTEXT (recent, current session)")
            lines.append("")
            for m in context_memories:
                imp = m.get('importance', 0)
                raw = m['content'][:300] if len(m['content']) > 300 else m['content']
                content = _compress_memory(raw)
                ts = m['timestamp'][:16] if len(m['timestamp']) > 16 else m['timestamp']
                scope = m.get('scope', 'session')
                lines.append(f"[{ts}] imp={imp:.1f} scope={scope} {content}")
            lines.append("")

        if unique_recall:
            lines.append("MNEMOSYNE RECALL (semantic search, all sessions)")
            lines.append("")
            for r in unique_recall:
                imp = r.get('importance', 0)
                raw = r['content'][:300] if len(r['content']) > 300 else r['content']
                content = _compress_memory(raw)
                ts = r.get('timestamp', '')[:16]
                score = r.get('score', 0)
                scope = r.get('scope', 'session')
                lines.append(f"[{ts}] imp={imp:.1f} score={score:.2f} scope={scope} {content}")
            lines.append("")

        lines.append("═══════════════════════════════════════════════════════════════")
        context_block = "\n".join(lines)
        full_context = f"\n\n{context_block}\n"

        return {"context": full_context}

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "Mnemosyne _on_pre_llm_call hook failed (session=%s): %s",
            session_id, e
        )
        return None


def _on_post_tool_call(tool_name, args, result, **kwargs):
    """
    Hook for post-tool-call processing.

    Auto-logging of tool calls is disabled by default because it quickly
    floods working_memory with low-signal operational noise (every terminal
    command, file write, etc.). Users can opt-in via MNEMOSYNE_LOG_TOOLS=1.

    If you want to remember the outcome of a tool call, use mnemosyne_remember
    explicitly from the conversation instead.
    """
    try:
        if not os.environ.get("MNEMOSYNE_LOG_TOOLS"):
            return

        mem = _get_memory()

        # Only log if explicitly opted in; keep importance low so these
        # don't pollute prompt context injection.
        if tool_name in ['terminal', 'execute_code', 'write_file', 'patch']:
            summary = f"Tool {tool_name} executed"
            if args:
                summary += f" with args: {str(args)[:100]}"

            mem.remember(
                content=f"[TOOL] {summary}",
                source="tool_execution",
                importance=0.1
            )
    except Exception as e:
        # C27: log at DEBUG rather than swallow. This hook is opt-in via
        # MNEMOSYNE_LOG_TOOLS=1, so the operator already wants the data --
        # if it's failing silently the opt-in is broken without their
        # knowledge.
        import logging
        logging.getLogger(__name__).debug(
            "Mnemosyne post-tool-call hook failed for tool=%s: %s",
            tool_name, e,
        )
