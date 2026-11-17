"""
Rammy HR Chatbot — Dedicated FastMCP Server (rammy_mcp_server.py)
"""

import os
import threading
from typing import List, Optional
from urllib.parse import quote

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

try:
    from mcp.server.transport_security import TransportSecuritySettings
    _SECURITY = TransportSecuritySettings(enable_dns_rebinding_protection=False)
except ImportError:
    _SECURITY = None


def _create_mcp_server():
    try:
        from mcp.server.fastmcp import FastMCP
        try:
            return FastMCP(
                "Rammy-HR-MCP",
                host="0.0.0.0",
                port=8000,
                stateless_http=True,
                transport_security=_SECURITY,
            )
        except TypeError:
            return FastMCP("Rammy-HR-MCP", host="0.0.0.0", port=8000)
    except ImportError:
        from mcp.server.mcpserver import MCPServer
        try:
            return MCPServer(
                "Rammy-HR-MCP",
                stateless_http=True,
                transport_security=_SECURITY,
            )
        except TypeError:
            return MCPServer("Rammy-HR-MCP")


mcp = _create_mcp_server()

QDRANT_HOST       = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "rammy_hr"
EMBED_MODEL       = "all-MiniLM-L6-v2"
QDRANT_TOP_K      = 5
PDF_PROXY_BASE    = os.getenv("PDF_PROXY_BASE", "http://localhost:3000/api/pdf")

_init_lock = threading.Lock()
_qdrant_client: Optional[QdrantClient] = None
_embed_model: Optional[SentenceTransformer] = None


class _TriggerClientFallback(Exception):
    """
    Escapes FastMCP's internal tool-error wrapper so the MCP server returns a
    JSON-RPC error when Qdrant is down. This causes unmodified chatbot_api.py
    to catch McpError in build_context() and fall back to its local Qdrant client.
    """
    def __init__(self, msg: str):
        self.msg = msg
        self._trips = 0

    def __str__(self) -> str:
        self._trips += 1
        if self._trips <= 2:
            raise self
        return self.msg


def _get_backends(force_reconnect: bool = False):
    global _qdrant_client, _embed_model
    with _init_lock:
        if _qdrant_client is None or force_reconnect:
            _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        if _embed_model is None:
            _embed_model = SentenceTransformer(EMBED_MODEL)
    return _qdrant_client, _embed_model


def pdf_label_to_url(source_label: str) -> str:
    filename = source_label[len("pdf:"):]
    return f"{PDF_PROXY_BASE.rstrip('/')}/{quote(filename)}"


def _query_qdrant(query: str, top_k: int):
    """Queries Qdrant and retries once with a fresh connection on transient failure."""
    for attempt in range(2):
        try:
            qdrant, embedder = _get_backends(force_reconnect=(attempt == 1))
            query_vector = embedder.encode(query).tolist()
            response = qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            return response.points
        except Exception as e:
            if attempt == 1:
                raise e


@mcp.tool()
def search_wcu_hr_knowledge(query: str, top_k: int = QDRANT_TOP_K) -> str:
    """
    Queries Qdrant for the top-K semantically similar chunks and formats them
    into the exact context string that unmodified chatbot_api.py expects.
    """
    try:
        results = _query_qdrant(query, top_k)
    except Exception as e:
        print(f"[mcp-retrieval] Qdrant search error: {e}")
        raise _TriggerClientFallback(f"MCP Qdrant error: {e}")

    if not results:
        return "NO_MATCHING_HR_SOURCES"

    selected = []
    for r in results:
        if not r.payload or not r.payload.get("text"):
            continue
        raw_url = r.payload.get("url", "")
        is_pdf = raw_url.startswith("pdf:")
        resolved_url = pdf_label_to_url(raw_url) if is_pdf else raw_url
        selected.append({
            "url": resolved_url,
            "text": r.payload.get("text", ""),
            "raw_url": raw_url,
            "is_pdf": is_pdf,
        })

    if not selected:
        return "NO_MATCHING_HR_SOURCES"

    has_web = any(not c["is_pdf"] for c in selected)
    if has_web:
        selected.sort(key=lambda c: (1 if c["is_pdf"] else 0))

    seen_urls: List[str] = []
    for c in selected:
        if c["url"] and c["url"] not in seen_urls:
            seen_urls.append(c["url"])

    # Append ", pdf:<filename>" on PDF source headers so chatbot_api.py's existing
    # `has_pdf_source = "pdf:" in context` check and filename regex work untouched.
    parts = [
        (
            f"Source {i}: {c['url']}, {c['raw_url']}\nContent:\n{c['text']}"
            if c["is_pdf"]
            else f"Source {i}: {c['url']}\nContent:\n{c['text']}"
        )
        for i, c in enumerate(selected, 1)
    ]

    source_list = "\n".join(f"- {url}" for url in seen_urls)
    return "\n\n".join(parts) + f"\n\nAvailable source URLs:\n{source_list}"


if __name__ == "__main__":
    # Warm up Qdrant & SentenceTransformer in a background thread so port 8000 opens immediately
    threading.Thread(target=_get_backends, daemon=True).start()
    mcp.run(transport="streamable-http")
