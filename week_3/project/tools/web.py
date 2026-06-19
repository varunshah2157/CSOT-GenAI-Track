"""
Web search & fetch — carried forward from Week 2, normalised to {content}/{error}.

Public API
----------
- web_search(query, num_results=5) -> {"content": "<title/link/snippet blocks>"} | {"error"}
- web_fetch(url)                   -> {"content": "<clean page text>"} | {"error"}

web_search uses Serper (Google). web_fetch pulls a page and extracts readable
text with trafilatura, truncating long pages.
"""

import os

import requests
import trafilatura

MAX_FETCH_CHARS = 8_000
TIMEOUT = 10


def web_search(query: str, num_results: int = 5) -> dict:
    """Search the web for general topics, news, blogs, and docs (not papers)."""
    api_key = os.environ.get(
        "SERPER_API_KEY")  # read lazily — robust to import order
    if not api_key:
        return {"error": "SERPER_API_KEY is missing in .env"}
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        organic = resp.json().get("organic", [])
        blocks = [
            f"Title: {it.get('title')}\nLink: {it.get('link')}\nSnippet: {it.get('snippet')}"
            for it in organic
        ]
        return {"content": "\n\n".join(blocks) if blocks else "No results found."}
    except Exception as e:
        return {"error": f"web_search failed: {e}"}


def web_fetch(url: str) -> dict:
    """Fetch a URL and return its readable text (truncated). Search before fetching."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchDesk/1.0)"}
        resp = requests.get(url, headers=headers,
                            allow_redirects=True, timeout=TIMEOUT)
        resp.raise_for_status()
        text = trafilatura.extract(
            resp.text, include_comments=False, include_tables=True
        )
        if not text:
            return {"error": "Could not extract clean text from the page."}
        if len(text) > MAX_FETCH_CHARS:
            text = text[:MAX_FETCH_CHARS] + "\n\n[...truncated]"
        return {"content": text}
    except Exception as e:
        return {"error": f"web_fetch failed: {e}"}


# ---------------------------------------------------------------------------
# Schemas + handler map
# ---------------------------------------------------------------------------

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for general topics, current events, blogs, and docs (NOT academic papers — use paper_search for those). Returns snippets + URLs.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch and read the readable text of a web page. Search first; don't fetch URLs blindly.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]

HANDLERS = {
    "web_search": web_search,
    "web_fetch": web_fetch,
}
