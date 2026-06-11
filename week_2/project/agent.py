"""
Build 3: Terminal Perplexity - The Research Agent TUI
=====================================================
A full-screen terminal UI using Textual, chaining web search, 
page fetching, and academic paper search (AlphaXiv MCP).
"""

import os
import json
import asyncio
import webbrowser
import requests
import trafilatura
from urllib.parse import urlparse

import httpx
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Header, Footer, Input, RichLog, Static
from rich.text import Text
from rich.markup import escape

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

sync_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)

async_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)


def get_active_model() -> str:
    priority_models = [
        "deepseek/deepseek-v4-flash:free",
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "openai/gpt-oss-20b:free",
        "openrouter/free"
    ]
    print("Checking for an available model endpoint...")
    for name in priority_models:
        try:
            sync_client.chat.completions.create(
                model=name,
                messages=[{"role": "user", "content": "ping"}],
            )
            print(f"Connected successfully to: {name}\n")
            return name
        except Exception:
            pass
    raise RuntimeError(
        "Critical Error: All priority models failed to respond.")


MODEL = get_active_model()

MAX_HISTORY_TURNS = 20

SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
ALPHAXIV_MCP_URL = "https://api.alphaxiv.org/mcp/v1"
REDIRECT_URI = "http://localhost:8765/callback"
TOKEN_FILE = ".alphaxiv_tokens.json"


# ---------------------------------------------------------------------------
# Local Web Tools
# ---------------------------------------------------------------------------

def web_search(query: str, num_results: int = 5) -> str:
    if not SERPER_API_KEY:
        return "Error: SERPER_API_KEY is missing in .env"
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


def save_research_note(filename: str, content: str) -> str:
    """Save findings to a markdown file in a notes/ directory."""
    try:
        os.makedirs("notes", exist_ok=True)
        # Force a .md extension if the model forgets
        if not filename.endswith(".md"):
            filename += ".md"

        filepath = os.path.join("notes", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully saved research note to {filepath}"
    except Exception as e:
        return f"Failed to save note: {e}"


LOCAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for general topics or recent facts. Returns snippets and URLs.",
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
    },
    {
        "type": "function",
        "function": {
            "name": "save_research_note",
            "description": "Save important research findings, summaries, or synthesized answers to a markdown file. Use this for multi-session research.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file (e.g., 'market_making_models.md')"},
                    "content": {"type": "string", "description": "The full markdown content to save."}
                },
                "required": ["filename", "content"],
            },
        },
    }
]


def trim_history(messages: list[dict], max_turns: int) -> list[dict]:
    """Keep the system message and only the last `max_turns` user/assistant pairs."""
    if len(messages) <= 1:
        return messages
    system_message = messages[0]
    chat_history = messages[1:]
    max_elements = max_turns * 2
    if len(chat_history) > max_elements:
        chat_history = chat_history[-max_elements:]
    return [system_message] + chat_history


# ---------------------------------------------------------------------------
# OAuth & Token Storage (Fully Async)
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
    webbrowser.open(auth_url)


async def wait_for_callback() -> tuple[str, str | None]:
    """A non-blocking, async localhost server to catch the OAuth redirect."""
    code = state = None
    stop_event = asyncio.Event()

    async def handle_client(reader, writer):
        nonlocal code, state
        request = (await reader.read(1024)).decode()
        first_line = request.split('\n')[0]
        path = first_line.split(' ')[1]

        query_string = urlparse(path).query
        params = {k: v for k, v in [item.split(
            '=') for item in query_string.split('&') if '=' in item]}

        code = params.get("code")
        state = params.get("state")

        response = b"HTTP/1.0 200 OK\r\nContent-Type: text/html\r\n\r\n<h1>Authorized. You can close this tab.</h1>"
        writer.write(response)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        stop_event.set()

    server = await asyncio.start_server(handle_client, 'localhost', 8765)
    await stop_event.wait()
    server.close()
    await server.wait_closed()

    if not code:
        raise RuntimeError("OAuth callback received no authorization code.")
    return code, state


# ---------------------------------------------------------------------------
# Textual TUI Application
# ---------------------------------------------------------------------------

class ResearchAgentApp(App):
    TITLE = "ResearchBot: TUI Edition"
    CSS = """
    Screen { layout: vertical; }
    Horizontal { height: 1fr; }
    #chat-panel {
        width: 65%;
        border: solid $primary;
        padding: 0 1;
    }
    #tool-panel {
        width: 35%;
        border: solid $warning;
        padding: 0 1;
    }
    Input {
        dock: bottom;
        height: 3;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_display", "Clear Display"),
        Binding("ctrl+k", "clear_history", "Clear History (Reset)"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.messages = [
            {"role": "system", "content": (
                "You are an elite academic research agent. Use your tools to find and verify information. "
                "If a tool fails, returns an error, or yields empty results, do not panic. "
                "Use that error information to adjust your strategy, change your search query, or try a different approach."
            )}
        ]

        self.chat_history_text = "[bold green]Agent Initialized.[/bold green] Ready for research.\n"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            # 1. The new layout structure here
            with VerticalScroll(id="chat-panel"):
                yield Static(self.chat_history_text, id="chat-text", markup=True)
            yield RichLog(id="tool-panel", wrap=True, markup=True)
        yield Input(placeholder="Ask a research question and press Enter...")
        yield Footer()

    def on_mount(self) -> None:
        # 2. Grab the specific widgets and save them as attributes here!
        self.chat_view = self.query_one("#chat-panel", VerticalScroll)
        self.chat_static = self.query_one("#chat-text", Static)
        self.tool_log = self.query_one("#tool-panel", RichLog)

        self.tool_log.write("[bold yellow]Tool Activity Log[/bold yellow]\n")
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return

        event.input.clear()
        self.chat_history_text += f"\n[bold cyan][You][/bold cyan] {escape(user_text)}\n"
        self.chat_static.update(Text.from_markup(self.chat_history_text))
        self.chat_view.scroll_end(animate=False)

        self.messages.append({"role": "user", "content": user_text})
        self.messages = trim_history(self.messages, MAX_HISTORY_TURNS)

        self.title = "ResearchBot (Thinking...)"

        # Fire off the async worker
        self.run_worker(self._get_response(), exclusive=True)

    async def _get_response(self) -> None:
        storage = FileTokenStorage()
        auth = OAuthClientProvider(
            server_url=ALPHAXIV_MCP_URL,
            client_metadata=OAuthClientMetadata(
                client_name="ResearchBot TUI",
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

                        mcp_tools = await session.list_tools()
                        mcp_openai_tools = [{"type": "function", "function": {
                            "name": t.name, "description": t.description, "parameters": t.inputSchema}} for t in mcp_tools.tools]
                        all_tools = LOCAL_TOOLS + mcp_openai_tools

                        for iteration in range(10):
                            # 1. Use async_client, await the call, and enable stream
                            response = await async_client.chat.completions.create(
                                model=MODEL,
                                messages=self.messages,
                                tools=all_tools,
                                stream=True,
                            )

                            collected_content = ""
                            tool_calls_dict = {}

                            # 1. Add a flag to track if we've printed the nametag yet
                            started_typing = False

                            # 2. Async iterate through the stream chunks
                            async for chunk in response:
                                delta = chunk.choices[0].delta

                                # Handle Text Streaming
                                if delta.content:
                                    # 3. Only print the tag if this is the first text chunk!
                                    if not started_typing:
                                        self.chat_history_text += "\n[bold green][Agent][/bold green] "
                                        started_typing = True

                                    collected_content += delta.content
                                    # Update UI instantly and scroll down (escaping the chunks!)
                                    self.chat_static.update(Text.from_markup(
                                        self.chat_history_text + escape(collected_content)))
                                    self.chat_view.scroll_end(animate=False)

                                # Handle Tool Calling (OpenAI sends args in chunks)
                                if delta.tool_calls:
                                    for tc_chunk in delta.tool_calls:
                                        idx = tc_chunk.index
                                        if idx not in tool_calls_dict:
                                            tool_calls_dict[idx] = {
                                                "id": tc_chunk.id,
                                                "type": "function",
                                                "function": {"name": tc_chunk.function.name or "", "arguments": ""}
                                            }
                                        if tc_chunk.function.arguments:
                                            tool_calls_dict[idx]["function"]["arguments"] += tc_chunk.function.arguments

                            # Commit the fully streamed text to our history state
                            if collected_content:
                                self.chat_history_text += escape(
                                    collected_content) + "\n"

                            # 4. Construct the final message dictionary
                            message_dict = {"role": "assistant"}
                            if collected_content:
                                message_dict["content"] = collected_content
                            if tool_calls_dict:
                                message_dict["tool_calls"] = list(
                                    tool_calls_dict.values())

                            self.messages.append(message_dict)

                            # If no tools were called, the turn is over.
                            if "tool_calls" not in message_dict:
                                self.title = "ResearchBot: TUI Edition"
                                return

                            # ... (Keep your existing tool execution logic directly below here) ...

                            # 5. Process accumulated tool calls
                            for tool_call in message_dict["tool_calls"]:
                                func_name = tool_call["function"]["name"]
                                args_string = tool_call["function"]["arguments"]

                                try:
                                    args = json.loads(args_string)
                                except json.JSONDecodeError:
                                    args = {}  # Failsafe for truncated streams

                                self.tool_log.write(
                                    f"→ [magenta]Invoking: {func_name}[/magenta]")

                                # --- Your existing tool execution logic remains exactly the same! ---
                                if func_name == "web_search":
                                    result = web_search(
                                        query=args.get("query"))
                                elif func_name == "web_fetch":
                                    result = web_fetch(url=args.get("url"))
                                elif func_name == "save_research_note":
                                    result = save_research_note(
                                        filename=args.get(
                                            "filename", "note.md"),
                                        content=args.get("content", "")
                                    )
                                else:
                                    try:
                                        mcp_result = await session.call_tool(func_name, args)
                                        result = mcp_result.content[0].text if mcp_result.content else "Success."
                                    except Exception as e:
                                        result = f"MCP Error: {e}"

                                self.tool_log.write(
                                    f"  [dim]↳ Completed ({len(str(result))} chars)[/dim]")

                                self.messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": str(result)
                                })

                        self.chat_log.write(
                            "\n[bold red][System][/bold red] Agent hit the maximum iteration limit.")
                        self.title = "ResearchBot: TUI Edition"

        except Exception as e:
            self.chat_log.write(
                f"\n[bold red][Error][/bold red] Connection or Execution failed: {str(e)}")
            self.title = "ResearchBot: TUI Edition"

    # -----------------------------------------------------------------------
    # Actions (Keyboard Shortcuts)
    # -----------------------------------------------------------------------

    def action_clear_display(self) -> None:
        """Clear the visible logs without touching conversation history (Ctrl+L)."""
        # Reset the string state and update the Static widget
        self.chat_history_text = "[dim]Display cleared. Conversation history remains intact.[/dim]\n"
        self.chat_static.update(Text.from_markup(self.chat_history_text))

        # Tool log is still a RichLog, so this stays the same
        self.tool_log.clear()
        self.tool_log.write("[bold yellow]Tool Activity Log[/bold yellow]\n")

    def action_clear_history(self) -> None:
        """Reset conversation history and clear the display (Ctrl+K)."""
        self.messages = [self.messages[0]]  # Keep only the system prompt

        # Reset the string state and update the Static widget
        self.chat_history_text = "[bold yellow]History and Display cleared. Fresh start.[/bold yellow]\n"
        self.chat_static.update(Text.from_markup(self.chat_history_text))

        # Tool log stays the same
        self.tool_log.clear()
        self.tool_log.write("[bold yellow]Tool Activity Log[/bold yellow]\n")


if __name__ == "__main__":
    ResearchAgentApp().run()
