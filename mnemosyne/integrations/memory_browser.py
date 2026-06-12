"""
Mnemosyne — Memory Browser Dashboard
=======================================
A web dashboard for browsing, searching, and managing Mnemosyne memories.

Requires: pip install starlette uvicorn
(same optional deps as MCP SSE transport)

Usage:
    mnemosyne-browser  --port 8081
    # Or
    python -m mnemosyne.integrations.memory_browser --port 8081

Then open http://localhost:8081 in your browser.
"""

import argparse
import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mnemosyne-browser")


def _resolve_db_path(bank: str = "default") -> str:
    """Resolve the Mnemosyne database path for a given bank."""
    data_dir = Path(
        os.environ.get("MNEMOSYNE_DATA_DIR")
        or os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")) + "/mnemosyne/data"
    )
    return str(data_dir / f"{bank}.db")


# ── Database Queries ──────────────────────────────────────────────────


def _get_connection(db_path: str) -> sqlite3.Connection:
    """Get a read-only connection to the Mnemosyne database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = 1")
    return conn


def get_memory_stats(db_path: str) -> Dict[str, Any]:
    """Get memory statistics across all tiers."""
    stats: Dict[str, Any] = {"tiers": {}, "total": 0}
    try:
        conn = _get_connection(db_path)
        cursor = conn.cursor()

        # Check which tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        # Working memory
        for table in ["working_memory", "memories", "conversations"]:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats["tiers"][table] = count
                stats["total"] += count

        # Episodic memory
        for table in ["episodic_memory", "episodes"]:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats["tiers"][table] = count
                stats["total"] += count

        # Triples
        if "triples" in tables:
            cursor.execute("SELECT COUNT(*) FROM triples")
            stats["triples"] = cursor.fetchone()[0]

        # FTS content
        for fts_table in ["fts_working", "fts_episodes"]:
            if fts_table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {fts_table}")
                stats[f"fts_{fts_table}"] = cursor.fetchone()[0]

        conn.close()
    except Exception as e:
        stats["error"] = str(e)
    return stats


def search_memories(
    db_path: str,
    query: str = "",
    limit: int = 50,
    offset: int = 0,
    source: str = "",
    tier: str = "",
    sort: str = "recent",
) -> List[Dict[str, Any]]:
    """Search memories with optional filters."""
    results: List[Dict[str, Any]] = []
    try:
        conn = _get_connection(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        target_tables = []
        if tier in ("", "working") and "working_memory" in tables:
            target_tables.append("working_memory")
        if tier in ("", "episodic") and "episodic_memory" in tables:
            target_tables.append("episodic_memory")
        if not target_tables and "memories" in tables:
            target_tables.append("memories")

        for table in target_tables:
            where_clauses = []
            params: List[Any] = []

            if query:
                # Try FTS search first
                fts_table = {"working_memory": "fts_working", "episodic_memory": "fts_episodes"}.get(table)
                if fts_table and fts_table in tables:
                    where_clauses.append(
                        f"rowid IN (SELECT rowid FROM {fts_table} WHERE content MATCH ?)"
                    )
                    params.append(query)
                else:
                    where_clauses.append("content LIKE ?")
                    params.append(f"%{query}%")

            if source:
                where_clauses.append("source = ?")
                params.append(source)

            order = "timestamp DESC" if sort == "recent" else "importance DESC"
            sql = f"SELECT * FROM {table}"
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
            sql += f" ORDER BY {order} LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(sql, params)
            for row in cursor.fetchall():
                results.append({
                    "id": row["id"] if "id" in row.keys() else row[0],
                    "content": row["content"] if "content" in row.keys() else "",
                    "source": row["source"] if "source" in row.keys() else "",
                    "timestamp": str(row["timestamp"]) if "timestamp" in row.keys() else "",
                    "importance": row["importance"] if "importance" in row.keys() else 0,
                    "tier": table,
                })

        conn.close()
    except Exception as e:
        logger.warning("Search error: %s", e)
    return results


def get_memory_detail(db_path: str, memory_id: str) -> Optional[Dict[str, Any]]:
    """Get a single memory by ID."""
    try:
        conn = _get_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        for table in ["working_memory", "episodic_memory", "memories"]:
            if table not in tables:
                continue
            cursor.execute(f"SELECT * FROM {table} WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result["tier"] = table
                conn.close()
                return result

        conn.close()
    except Exception as e:
        logger.warning("Detail error: %s", e)
    return None


# ── HTML Template ────────────────────────────────────────────────────


PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mnemosyne Memory Browser</title>
<style>
  :root {
    --bg: #faf9f6;
    --surface: #ffffff;
    --border: #e2e0dc;
    --text: #2c2c2c;
    --text-secondary: #6b6b6b;
    --accent: #c7673d;
    --accent-light: #f0e6e0;
    --success: #5a8a6a;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    --font-mono: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.6; }
  .header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 1rem 2rem; display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap; }
  .header h1 { font-size: 1.25rem; font-weight: 600; white-space: nowrap; }
  .header h1 span { color: var(--accent); }
  .stats { display: flex; gap: 1.5rem; font-size: 0.85rem; }
  .stat { color: var(--text-secondary); }
  .stat strong { color: var(--text); }
  .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
  .filters { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; }
  .filters input, .filters select { padding: 0.5rem 0.75rem; border: 1px solid var(--border); border-radius: 6px; font-size: 0.875rem; background: var(--surface); color: var(--text); }
  .filters input[type="text"] { flex: 1; min-width: 200px; }
  .filters button { padding: 0.5rem 1.25rem; background: var(--accent); color: white; border: none; border-radius: 6px; font-size: 0.875rem; cursor: pointer; }
  .filters button:hover { opacity: 0.9; }
  .memory-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 0.75rem; }
  .memory-card:hover { border-color: var(--accent); }
  .memory-meta { display: flex; gap: 1rem; font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.5rem; flex-wrap: wrap; }
  .memory-meta .tag { background: var(--accent-light); color: var(--accent); padding: 0.125rem 0.5rem; border-radius: 4px; font-weight: 500; }
  .memory-content { font-size: 0.9rem; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
  .empty { text-align: center; padding: 3rem; color: var(--text-secondary); }
  .pagination { display: flex; justify-content: center; gap: 0.5rem; margin-top: 1.5rem; }
  .pagination a { padding: 0.5rem 0.75rem; border: 1px solid var(--border); border-radius: 6px; text-decoration: none; color: var(--text); font-size: 0.875rem; }
  .pagination a:hover { background: var(--accent-light); border-color: var(--accent); }
  .pagination a.active { background: var(--accent); color: white; border-color: var(--accent); }
  .detail { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; }
  .detail h2 { font-size: 1rem; margin-bottom: 1rem; }
  .detail .field { margin-bottom: 0.75rem; }
  .detail .field-label { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }
  .detail .field-value { font-size: 0.9rem; }
  .back { display: inline-block; margin-bottom: 1rem; color: var(--accent); text-decoration: none; font-size: 0.875rem; }
  .back:hover { text-decoration: underline; }
  @media (max-width: 768px) {
    .header { padding: 1rem; flex-direction: column; align-items: flex-start; }
    .container { padding: 1rem; }
    .filters { flex-direction: column; }
    .filters input[type="text"] { min-width: auto; width: 100%; }
  }
</style>
</head>
<body>
<div class="header">
  <h1><a href="/" style="text-decoration:none;color:inherit;">Mnemosyne <span>Browse</span></a></h1>
  <div class="stats" id="stats">Loading...</div>
</div>
<div class="container" id="app">
  <div class="filters">
    <input type="text" id="search" placeholder="Search memories..." value="{query}">
    <select id="source">
      <option value="">All sources</option>
      {source_options}
    </select>
    <select id="tier">
      <option value="">All tiers</option>
      <option value="working" {sel_working}>Working</option>
      <option value="episodic" {sel_episodic}>Episodic</option>
    </select>
    <select id="sort">
      <option value="recent" {sel_recent}>Most recent</option>
      <option value="importance" {sel_importance}>Most important</option>
    </select>
    <button onclick="search()">Search</button>
  </div>
  <div id="results">
    {results_html}
  </div>
</div>
<script>
async function fetchStats() {{
  const resp = await fetch('/api/stats');
  const data = await resp.json();
  const total = data.total || 0;
  let html = '';
  for (const [tier, count] of Object.entries(data.tiers || {{}})) {{
    html += '<span class="stat"><strong>' + count + '</strong> ' + tier.replace('_', ' ') + '</span>';
  }}
  if (data.triples) html += '<span class="stat"><strong>' + data.triples + '</strong> triples</span>';
  document.getElementById('stats').innerHTML = html;
}}
function search() {{
  const q = document.getElementById('search').value;
  const src = document.getElementById('source').value;
  const tier = document.getElementById('tier').value;
  const sort = document.getElementById('sort').value;
  const params = new URLSearchParams({{q, src, tier, sort}});
  window.location.href = '/?' + params.toString();
}}
document.getElementById('search').addEventListener('keydown', e => {{ if (e.key === 'Enter') search(); }});
fetchStats();
</script>
</body>
</html>"""


def _build_html(
    memories: List[Dict[str, Any]],
    query: str = "",
    source: str = "",
    tier: str = "",
    sort: str = "recent",
) -> str:
    """Render the memory list page."""

    # Source options
    sources = set(m.get("source", "") for m in memories)
    source_opts = "".join(
        f'<option value="{s}" {"selected" if s == source else ""}>{s}</option>'
        for s in sorted(sources)
    )

    # Results
    if memories:
        results = "".join(
            f"""<div class="memory-card">
          <div class="memory-meta">
            <span class="tag">{m.get('tier', 'memory').replace('_', ' ')}</span>
            <span>source: {m.get('source', 'unknown')}</span>
            <span>{str(m.get('timestamp', ''))[:19]}</span>
            <span>importance: {m.get('importance', 0)}</span>
            <span><a href="/detail/{m.get('id', '')}" style="color:var(--accent);text-decoration:none;">&#8599;</a></span>
          </div>
          <div class="memory-content">{str(m.get('content', ''))[:500]}</div>
        </div>"""
            for m in memories
        )
    else:
        results = '<div class="empty">No memories found.</div>'

    return PAGE_HTML.format(
        query=query,
        source_options=source_opts,
        sel_working='selected' if tier == 'working' else '',
        sel_episodic='selected' if tier == 'episodic' else '',
        sel_recent='selected' if sort == 'recent' else '',
        sel_importance='selected' if sort == 'importance' else '',
        results_html=results,
    )


# ── FastAPI App ───────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app):
    logger.info("Mnemosyne Browser starting up")
    yield
    logger.info("Mnemosyne Browser shutting down")


def create_app(banks: List[str], default_bank: str):
    """Create the FastAPI application."""
    try:
        from fastapi import FastAPI, Request, Query
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError:
        raise RuntimeError("Memory browser requires fastapi and uvicorn. Install: pip install fastapi uvicorn")

    app = FastAPI(title="Mnemosyne Browser", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        q: str = "",
        src: str = "",
        tier: str = "",
        sort: str = "recent",
        bank: str = default_bank,
        page: int = 1,
    ):
        db_path = _resolve_db_path(bank)
        limit = 50
        offset = (page - 1) * limit
        memories = search_memories(db_path, query=q, limit=limit, offset=offset, source=src, tier=tier, sort=sort)
        return _build_html(memories, query=q, source=src, tier=tier, sort=sort)

    @app.get("/api/stats")
    async def api_stats(bank: str = default_bank):
        db_path = _resolve_db_path(bank)
        stats = get_memory_stats(db_path)
        return JSONResponse(stats)

    @app.get("/api/search")
    async def api_search(
        q: str = "",
        src: str = "",
        tier: str = "",
        sort: str = "recent",
        bank: str = default_bank,
        limit: int = 50,
        offset: int = 0,
    ):
        db_path = _resolve_db_path(bank)
        results = search_memories(db_path, query=q, limit=limit, offset=offset, source=src, tier=tier, sort=sort)
        return JSONResponse({"results": results, "count": len(results)})

    @app.get("/detail/{memory_id}", response_class=HTMLResponse)
    async def detail(memory_id: str, bank: str = default_bank):
        db_path = _resolve_db_path(bank)
        mem = get_memory_detail(db_path, memory_id)
        if not mem:
            return HTMLResponse("<h1>Memory not found</h1><a href='/'>Back</a>", status_code=404)

        fields = "".join(
            f'<div class="field"><div class="field-label">{k}</div><div class="field-value">{v}</div></div>'
            for k, v in mem.items()
            if k not in ("embedding", "vector")
        )
        html = f"""<!DOCTYPE html><html><head><title>Memory Detail</title><link rel="stylesheet" href="/"></head><body>
        <div class="header"><h1><a href="/" style="text-decoration:none;color:inherit;">Mnemosyne <span>Detail</span></a></h1></div>
        <div class="container"><a href="/" class="back">&larr; Back</a><div class="detail"><h2>Memory {memory_id}</h2>{fields}</div></div></body></html>"""
        return HTMLResponse(html)

    return app


def main():
    parser = argparse.ArgumentParser(description="Mnemosyne Memory Browser")
    parser.add_argument("--port", type=int, default=8081, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--bank", default="default", help="Default memory bank")
    parser.add_argument("--banks", nargs="*", default=["default"], help="Available memory banks")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    try:
        import uvicorn
    except ImportError:
        print("Memory browser requires uvicorn. Install: pip install uvicorn")
        return

    app = create_app(banks=args.banks, default_bank=args.bank)
    logger.info("Memory browser starting on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info" if args.verbose else "warning")


if __name__ == "__main__":
    main()
