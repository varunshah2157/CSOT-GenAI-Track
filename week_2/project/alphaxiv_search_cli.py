import os
import json
import asyncio
import webbrowser
import requests
import trafilatura
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
from openai import OpenAI
from dotenv import load_dotenv

from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "deepseek/deepseek-v4-flash:free"
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

ALPHAXIV_MCP_URL = "https://api.alphaxiv.org/mcp/v1"
REDIRECT_URI = "http://localhost:8765/callback"
TOKEN_FILE = ".alphaxiv_tokens.json"

# ---------------------------------------------------------------------------
# Local Web Tools
# ---------------------------------------------------------------------------


def web_search(query: str, num_results: int = 5) -> str:
    if not SERPER_API_KEY:
        return "Error: SERPER_API_KEY is missing."
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY,
                     "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        results = [f"Title: {item.get('title')}\nLink: {item.get('link')}\nSnippet: {item.get('snippet')}\n" for item in data.get(
            "organic", [])]
        return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search failed: {e}"


def web_fetch(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        response = requests.get(url, headers=headers,
                                allow_redirects=True, timeout=10)
        response.raise_for_status()
        text = trafilatura.extract(
            response.text, include_comments=False, include_tables=True)
        if not text:
            return "Could not extract clean text from the page."
        MAX_CHARS = 8000
        return text[:MAX_CHARS] + "\n\n[...truncated]" if len(text) > MAX_CHARS else text
    except Exception as e:
        return f"Fetch failed: {e}"


LOCAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for general topics or recent facts.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch and read the raw text of a web page.",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        },
    }
]

# ---------------------------------------------------------------------------
# OAuth & Token Storage
# ---------------------------------------------------------------------------


class FileTokenStorage(TokenStorage):
    def __init__(self):
        self.tokens: OAuthToken | None = None
        self.client_info: OAuthClientInformationFull | None = None
        if os.path.exists(TOKEN_FILE):
            try:
                data = json.loads(open(TOKEN_FILE).read())
                if data.get("tokens"):
                    self.tokens = OAuthToken(**data["tokens"])
                if data.get("client_info"):
                    self.client_info = OAuthClientInformationFull(
                        **data["client_info"])
            except Exception:
                pass

    def _save(self):
        data = {}
        if self.tokens:
            data["tokens"] = self.tokens.model_dump(mode="json")
        if self.client_info:
            data["client_info"] = self.client_info.model_dump(mode="json")
        open(TOKEN_FILE, "w").write(json.dumps(data, indent=2))

    async def get_tokens(self) -> OAuthToken | None: return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.tokens = tokens
        self._save()

    async def get_client_info(
        self) -> OAuthClientInformationFull | None: return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.client_info = client_info
        self._save()


async def open_browser(auth_url: str) -> None:
    print(
        f"\n[System] Opening browser for AlphaXiv login...\nIf it doesn't open manually navigate to: {auth_url}\n")
    webbrowser.open(auth_url)


async def wait_for_callback() -> tuple[str, str | None]:
    code = state = None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal code, state
            params = parse_qs(urlparse(self.path).query)
            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h1>Authorized. You can close this tab and return to the terminal.</h1>")

        def log_message(self, *args): pass

    print(f"[System] Waiting for callback on {REDIRECT_URI} ...")
    server = HTTPServer(("localhost", 8765), Handler)
    server.timeout = 120
    server.handle_request()
    server.server_close()

    if not code:
        raise RuntimeError("OAuth callback received no authorization code.")
    return code, state

# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------


async def call_model(messages: list[dict]) -> str:
    storage = FileTokenStorage()

    auth = OAuthClientProvider(
        server_url=ALPHAXIV_MCP_URL,
        client_metadata=OAuthClientMetadata(
            client_name="ResearchBot Agent",
            redirect_uris=[REDIRECT_URI],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope="read",
        ),
        storage=storage,
        redirect_handler=open_browser,
        callback_handler=wait_for_callback,
    )

    try:
        async with httpx.AsyncClient(auth=auth, follow_redirects=True, timeout=60) as http:
            async with streamable_http_client(ALPHAXIV_MCP_URL, http_client=http) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    print("[System] AlphaXiv MCP connection established.\n")

                    # 1. Discover Tools
                    mcp_tools_response = await session.list_tools()
                    mcp_openai_tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            },
                        }
                        for tool in mcp_tools_response.tools
                    ]

                    all_tools = LOCAL_TOOLS + mcp_openai_tools

                    # 2. Agent Loop
                    for _ in range(10):
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=messages,
                            tools=all_tools,
                        )

                        message = response.choices[0].message
                        messages.append(message.model_dump(exclude_none=True))

                        if not message.tool_calls:
                            return message.content

                        # 3. Execute Tools
                        for tool_call in message.tool_calls:
                            func_name = tool_call.function.name
                            args = json.loads(tool_call.function.arguments)
                            print(f"[Agent] Called tool: {func_name}")

                            if func_name == "web_search":
                                result = web_search(query=args.get("query"))
                            elif func_name == "web_fetch":
                                result = web_fetch(url=args.get("url"))
                            else:
                                # Delegate to AlphaXiv
                                try:
                                    mcp_result = await session.call_tool(func_name, args)
                                    result = mcp_result.content[0].text if mcp_result.content else "Success."
                                except Exception as e:
                                    result = f"MCP Tool Error: {e}"

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": str(result)
                            })

                    return "Error: Agent hit the maximum iteration limit."
    except Exception as e:
        return f"System Error during execution: {e}"

if __name__ == "__main__":
    test_messages = [
        {"role": "system", "content": "You are a helpful research agent. Use your tools to find accurate information."},
        {"role": "user", "content": "Search AlphaXiv for the original Guéant-Lehalle-Tapia paper on inventory risk."}
    ]
    print("Agent is starting up...")
    final_answer = asyncio.run(call_model(test_messages))
    print("\nFINAL ANSWER:\n", final_answer)
