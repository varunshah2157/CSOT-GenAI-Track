"""
Paper search & read — Hugging Face Papers API (an ML/CS subset of arXiv).

Public API
----------
- paper_search(query, limit=8) -> {"papers": [{arxiv_id, title, abstract, url}, ...]} | {"error"}
- read_paper(arxiv_id)         -> {title, authors, abstract, content, url} | {"error"}

Endpoints (base https://huggingface.co):
- GET /api/papers/search?q=...   (search list; each item wraps data under a "paper" key)
- GET /api/papers/{id}           (structured metadata)
- GET /papers/{id}.md            (arXiv-HTML-derived markdown; not every paper has it)

This index is NOT all of arXiv. On a 404 the paper exists on arXiv but isn't
indexed here — the agent should fall back to web_fetch on arxiv.org/abs/{id}.
"""

import os
import re

import requests

HF_BASE = "https://huggingface.co"
MAX_PAPER_CHARS = 12_000          # papers can be 50k+ tokens; always cap
TIMEOUT = 20


def _headers() -> dict:
    """Auth is optional; an HF_TOKEN only helps with rate limits."""
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Strip URLs/prefixes/whitespace down to a bare arXiv id (version kept if present).

    Accepts e.g. 'https://arxiv.org/abs/2205.14135v2', 'arxiv:2205.14135',
    'abs/2205.14135' -> '2205.14135(v2)'.
    """
    s = str(arxiv_id).strip()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"(arxiv\.org/(abs|pdf)/|arxiv:)", "", s, flags=re.IGNORECASE)
    s = s.strip("/")
    s = s.split("/")[-1]            # drop any trailing path
    if s.endswith(".pdf"):
        s = s[:-4]
    return s


def _unwrap(item: dict) -> dict:
    """Search items wrap fields under a 'paper' key; detail items don't. Handle both."""
    if isinstance(item, dict) and isinstance(item.get("paper"), dict):
        # merge so top-level keys (thumbnail etc.) don't shadow the inner paper fields
        merged = dict(item)
        merged.update(item["paper"])
        return merged
    return item


def paper_search(query: str, limit: int = 8) -> dict:
    """Search HF Papers by keyword. Returns small dicts; call read_paper for full text."""
    try:
        resp = requests.get(
            f"{HF_BASE}/api/papers/search",
            params={"q": query},
            headers=_headers(),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            data = data.get("papers", []) if isinstance(data, dict) else []

        papers = []
        for raw in data[:limit]:
            p = _unwrap(raw)
            pid = p.get("id") or p.get("arxiv_id")
            if not pid:
                continue
            title = " ".join((p.get("title") or "").split())
            abstract = " ".join(
                (p.get("summary") or p.get("abstract") or "").split())
            papers.append({
                "arxiv_id": pid,
                "title": title,
                "abstract": abstract[:400] + ("..." if len(abstract) > 400 else ""),
                "url": f"https://arxiv.org/abs/{pid}",
            })

        if not papers:
            return {"error": f"No papers found for {query!r}. Try web_search as a fallback."}
        return {"papers": papers}
    except Exception as e:
        return {"error": f"paper_search failed: {e}"}


def read_paper(arxiv_id: str) -> dict:
    """Fetch metadata + markdown content for a paper. Falls back to abstract if no markdown."""
    pid = normalize_arxiv_id(arxiv_id)
    url = f"https://arxiv.org/abs/{pid}"
    try:
        meta_resp = requests.get(
            f"{HF_BASE}/api/papers/{pid}", headers=_headers(), timeout=TIMEOUT
        )
        if meta_resp.status_code == 404:
            return {
                "error": f"{pid} not indexed on Hugging Face (404). "
                f"Fall back to web_fetch('{url}').",
                "url": url,
            }
        meta_resp.raise_for_status()
        meta = _unwrap(meta_resp.json())

        title = " ".join((meta.get("title") or "").split())
        abstract = " ".join(
            (meta.get("summary") or meta.get("abstract") or "").split())
        authors = [a.get("name")
                   for a in meta.get("authors", []) if a.get("name")]

        # Markdown body (arXiv-HTML-derived); not always present.
        content = ""
        try:
            md_resp = requests.get(
                f"{HF_BASE}/papers/{pid}.md", headers=_headers(), timeout=TIMEOUT
            )
            if md_resp.ok and md_resp.text.strip():
                content = md_resp.text.strip()
        except Exception:
            pass

        if not content:
            content = abstract or "(no content available; abstract empty)"

        if len(content) > MAX_PAPER_CHARS:
            content = content[:MAX_PAPER_CHARS] + \
                "\n\n[...truncated — large paper]"

        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "content": content,
            "url": url,
        }
    except Exception as e:
        return {"error": f"read_paper failed for {pid}: {e}. Try web_fetch('{url}')."}


# ---------------------------------------------------------------------------
# Schemas + handler map
# ---------------------------------------------------------------------------

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "paper_search",
            "description": (
                "Search academic ML/CS papers via Hugging Face Papers (an arXiv subset). "
                "Use this for research/literature questions. Returns title, arxiv_id, and a "
                "short abstract snippet — then call read_paper with the arxiv_id for full text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_paper",
            "description": (
                "Read a paper's metadata + markdown content by arxiv_id (from paper_search — "
                "do not guess IDs). On a 404 the paper isn't on HF; fall back to "
                "web_fetch('https://arxiv.org/abs/{id}')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "e.g. '2205.14135' (URLs are accepted and normalised)."},
                },
                "required": ["arxiv_id"],
            },
        },
    },
]

HANDLERS = {
    "paper_search": paper_search,
    "read_paper": read_paper,
}
