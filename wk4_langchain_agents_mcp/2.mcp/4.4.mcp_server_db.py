"""An MCP server backed by the Chinook SQLite database.

Chinook is a sample "digital music store" database (artists, albums, tracks,
genres, customers, invoices, ...). This server exposes three MCP primitives:

  * tool     -> search_tracks : run a safe, parameterized search over the catalog
  * resource -> schema://chinook : expose the database schema for an LLM to read
  * prompt   -> music_analyst : a ready-made prompt that ties the two together

It supports two transports:

  * stdio (default) -- the client launches this file as a child process and
    exchanges JSON-RPC over stdin/stdout. Used for local tools like the
    Inspector or 4.5.mcp_client_db.py.
  * streamable-http (pass --http, or set MCP_TRANSPORT=http) -- the server
    listens on a TCP port so any networked client can connect. This is what the
    Docker image uses so anyone can reach the server at http://<host>:<port>/mcp.

Host and port can be overridden with the MCP_HOST and MCP_PORT env vars (the
Docker image sets MCP_HOST=0.0.0.0 so the port is reachable from outside the
container). Because stdout is reserved for the MCP protocol over stdio, all
logging goes to stderr.
"""

import os
import sys
import asyncio
import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,  # stdout is reserved for the MCP protocol over stdio
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("mcp_server_db")

# Resolve the DB path relative to THIS file so it works no matter what working
# directory the MCP client launches the server from. DB_PATH can also be
# overridden with an env var (handy inside containers).
DB_PATH = Path(os.environ.get("CHINOOK_DB_PATH", Path(__file__).parent / "resources" / "Chinook.db"))

# langchain_common.py lives at the repo root (one level up). Add it to the
# import path so we can reuse its preconfigured Databricks LLMs.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Transport + network binding. Default is stdio; --http (or MCP_TRANSPORT=http)
# switches to streamable-http for network access.
if "--http" in sys.argv or os.environ.get("MCP_TRANSPORT", "").lower() in {"http", "streamable-http"}:
    MCP_TRANSPORT = "streamable-http"
else:
    MCP_TRANSPORT = "stdio"

MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))

mcp = FastMCP("chinook_db_server", host=MCP_HOST, port=MCP_PORT)

# The LLM is created lazily on first use (search_tracks_v2) so the server still
# starts over stdio even when the Databricks env vars are absent.
_llm_noreason = None


def _get_llm():
    """Return a cached `llm_noreason` (ChatOpenAI, reasoning_effort="none") from
    langchain_common.bootstrap_notebook(), creating it on first call."""
    global _llm_noreason
    if _llm_noreason is None:
        from langchain_common import bootstrap_notebook

        # bootstrap returns: token, host, endpoint, (llm, llm_noreason), embeddings
        _, _, _, (_, _llm_noreason), _ = bootstrap_notebook()
    return _llm_noreason


def _query(sql: str, params: tuple | dict[str, Any] = ()) -> list[dict[str, Any]]:

    uri = DB_PATH.as_uri() + "?mode=ro"  # e.g. file:///C:/.../Chinook.db?mode=ro
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def _format_duration(milliseconds: int | None) -> str:
    """Turn a track length in ms into mm:ss."""
    if not milliseconds:
        return "0:00"
    seconds = milliseconds // 1000
    return f"{seconds // 60}:{seconds % 60:02d}"


# --- Tool ---------------------------------------------------------------------
@mcp.tool()
def search_tracks(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search the music catalog for tracks whose name, album, or artist matches
    `query` (case-insensitive). Returns up to `limit` matches with the artist,
    album, genre, duration, and price for each."""

    logger.info("search_tracks(query=%r, limit=%d)", query, limit)

    rows = _query(
        """
        SELECT t.Name        AS track,
               ar.Name       AS artist,
               al.Title      AS album,
               g.Name        AS genre,
               t.Milliseconds AS ms,
               t.UnitPrice   AS price
        FROM Track t
        JOIN Album  al ON al.AlbumId  = t.AlbumId
        JOIN Artist ar ON ar.ArtistId = al.ArtistId
        LEFT JOIN Genre g ON g.GenreId = t.GenreId
        WHERE t.Name LIKE :q OR al.Title LIKE :q OR ar.Name LIKE :q
        ORDER BY ar.Name, al.Title, t.Name
        LIMIT :limit
        """,
        {"q": f"%{query}%", "limit": limit},  # parameterized -> no SQL injection
    )

    results = [
        {
            "track": r["track"],
            "artist": r["artist"],
            "album": r["album"],
            "genre": r["genre"],
            "duration": _format_duration(r["ms"]),
            "price": r["price"],
        }
        for r in rows
    ]
    return {"query": query, "count": len(results), "results": results}


# --- Tool (LLM-augmented) -----------------------------------------------------
@mcp.tool()
def search_tracks_v2(query: str, limit: int = 10) -> Dict[str, Any]:

    logger.info("search_tracks_v2(query=%r, limit=%d)", query, limit)

    # 1) Retrieve candidate tracks exactly like search_tracks does.
    found = search_tracks(query, limit)
    results = found["results"]

    if not results:
        return {**found, "summary": f"No tracks matched {query!r}."}

    # 2) Build a compact, plain-text view of the rows for the LLM to read.
    lines = [
        f"- {r['track']} by {r['artist']} (album: {r['album']}, "
        f"genre: {r['genre']}, {r['duration']}, ${r['price']})"
        for r in results
    ]
    catalog_text = "\n".join(lines)

    # 3) Ask the LLM to summarize. We only pass the rows we already retrieved,
    #    so the model can't hallucinate tracks that aren't in the database.
    prompt = (
        "You are a music store assistant. A customer searched for "
        f"\"{query}\". Using ONLY the tracks listed below, write 2-3 sentences "
        "summarizing what was found (mention notable artists, genres, and the "
        "price range). Do not invent tracks that are not listed.\n\n"
        f"Tracks:\n{catalog_text}"
    )

    try:
        summary = _get_llm().invoke(prompt).content
    except Exception as exc:  # the DB results are still useful without the LLM
        logger.warning("LLM summary failed: %s", exc)
        summary = f"(LLM summary unavailable: {exc})"

    return {**found, "summary": summary}


# --- Resource -----------------------------------------------------------------
@mcp.resource("schema://chinook")
def database_schema() -> str:
    """The full SQL schema (CREATE statements) of the Chinook database, so a
    client/LLM can understand the available tables and columns."""

    logger.info("Reading resource schema://chinook")
    rows = _query(
        """
        SELECT sql FROM sqlite_master
        WHERE type = 'table' AND sql IS NOT NULL
        ORDER BY name
        LIMIT 5
        """
    )
    return "\n\n".join(r["sql"] for r in rows)


@mcp.resource("chinook://artists")
def artist_names() -> str:
    """A sample of 5 artist names from the catalog, one per line."""

    logger.info("Reading resource chinook://artists")
    rows = _query("SELECT Name FROM Artist ORDER BY Name LIMIT 5")
    return "\n".join(r["Name"] for r in rows)



# --- Prompt -------------------------------------------------------------------
@mcp.prompt()
def music_analyst(question: str) -> str:
    """A prompt that frames the assistant as a Chinook music-store data analyst."""

    return f"""You are a data analyst for a digital music store backed by the Chinook database.

To answer the user, you may:
  - read the `schema://chinook` resource to learn the available tables and columns, and
  - call the `search_tracks` tool to look up tracks by name, album, or artist.

Only answer questions about the music catalog (artists, albums, tracks, genres).
If the question is unrelated, say you can only help with the music store data.

User question: {question}
"""


def _serve_http() -> None:
    """Run the streamable-http app under uvicorn, binding MCP_HOST:MCP_PORT."""
    import uvicorn

    config = uvicorn.Config(
        mcp.streamable_http_app(),
        host=MCP_HOST,
        port=MCP_PORT,
        log_level="info",
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)

    if sys.platform == "win32":
        # SelectorEventLoop lets Ctrl+C interrupt uvicorn promptly on Windows.
        loop = asyncio.SelectorEventLoop()
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()
    else:
        server.run()


if __name__ == "__main__":
    if not DB_PATH.exists():
        logger.error("Database not found at %s", DB_PATH)
        sys.exit(1)

    try:
        if MCP_TRANSPORT == "stdio":
            logger.info("Launching Chinook MCP server over stdio (db: %s).", DB_PATH)
            mcp.run(transport="stdio")
        else:
            logger.info(
                "Launching Chinook MCP server over %s at http://%s:%d/mcp (db: %s). "
                "Press Ctrl+C to stop.",
                MCP_TRANSPORT, MCP_HOST, MCP_PORT, DB_PATH,
            )
            _serve_http()
    except KeyboardInterrupt:
        logger.info("Received interrupt (Ctrl+C) -- stopping server.")
    finally:
        logger.info("Chinook MCP server process exited.")
