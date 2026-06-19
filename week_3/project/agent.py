"""
Research Desk — Week 3 Project
===============================
Class hierarchy:
  Agent       — brain: chat(), _run_loop(), dispatch(), session I/O   (this file)
  REPLAgent   — terminal REPL + one-shot CLI                          (this file)
  TUIAgent    — Textual UI                                            (tui.py)

`agent.py` has NO Textual imports. The brain is UI-agnostic; subclasses only
add how the user talks to it.

Usage:
  python agent.py                                # interactive REPL
  python agent.py "What is quantum computing?"   # one-shot, print answer, exit
  python agent.py --tui                          # Textual UI
  python agent.py --session abc123               # resume a session in the REPL
  python agent.py --session abc123 "continue ..."# resume + one-shot

REPL commands: /help  /sessions  /resume <id>  /new  /quit
"""

from tools import web, papers, files
from openai import OpenAI
import os
import sys
import json
import argparse
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load .env BEFORE importing tools — some tool modules read env vars, and a late
# load_dotenv() would leave them frozen as None (this bit SERPER_API_KEY).
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = os.path.abspath(os.environ.get("WORKSPACE_ROOT", "."))
SESSIONS_DIR = os.path.join(".agent", "sessions")
NOTES_DIR = "notes"
MAX_ITERATIONS = 10
REQUEST_TIMEOUT = 90    # seconds per model call; a stalled provider then fails over
MAX_TOKENS = 2000       # bounds answer length + the credit a provider reserves up-front
# stream tokens via _emit('token'); set False for a blocking loop
STREAM = True

# Placeholder keeps import + tui-import safe when the key is unset; a real call
# then fails with a clear auth error instead of crashing at import time.
# timeout + max_retries make a stalled/rate-limited free provider fail fast so
# _complete() can move to the next model instead of hanging on the SDK's 10-min default.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get(
        "OPENROUTER_API_KEY") or "MISSING_OPENROUTER_API_KEY",
    timeout=REQUEST_TIMEOUT,
    max_retries=1,
)

# Merge each tool module's self-described schema + handler map. Adding a tool
# means adding a module — dispatch() never grows an if/elif chain.
TOOLS = files.SCHEMAS + papers.SCHEMAS + web.SCHEMAS
HANDLERS = {**files.HANDLERS, **papers.HANDLERS, **web.HANDLERS}


# Free OpenRouter models, best-first. `openrouter/free` auto-selects an available
# free model that supports the features a request needs (incl. tool calling), so
# it's a resilient default. Free-tier slugs change often — edit this list if one
# stops working; verify current slugs at https://openrouter.ai/models.
PRIORITY_MODELS = [
    "openrouter/free",
    "qwen/qwen3-235b-a22b:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-20b:free",
]

_MODEL_CACHE: str | None = None


def get_active_model() -> str:
    """Model to display/attempt first. No network call (saves free-tier quota);
    real selection + failover happens per-request in Agent._complete()."""
    return _MODEL_CACHE or PRIORITY_MODELS[0]


# ---------------------------------------------------------------------------
# Session ID
# ---------------------------------------------------------------------------

def _gen_id() -> str:
    """8-char id. Prefer nanoid (per the lesson); fall back to uuid if absent."""
    try:
        from nanoid import generate
        return generate(size=8)
    except Exception:
        import uuid
        return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Procedural memory: base prompt + AGENTS.md
# ---------------------------------------------------------------------------

BASE_PROMPT = (
    "You are Research Desk, a careful research agent with web tools, academic "
    "paper tools, and sandboxed file tools.\n\n"
    "Tool routing:\n"
    "- ML / academic / literature questions: paper_search -> read_paper.\n"
    "- Current events, blogs, docs: web_search -> web_fetch.\n"
    "- Save new findings: write_file('notes/<topic>.md', ...).\n"
    "- Update existing notes: read_file -> edit_file (don't rewrite whole files).\n"
    "- Recall past work: list_files('notes/') -> read_file.\n"
    "On a tool 'error', adjust (different query, tool, or path) and retry — don't give up.\n"
    "Search before fetching. Cite sources inline as [title](url). Be concise in chat; "
    "put long detail in note files."
)

AGENTS_PATHS = ("AGENTS.md", os.path.join(".agent", "AGENTS.md"))


def build_system_prompt() -> str:
    """Base developer prompt + the user-owned rules from AGENTS.md (if present)."""
    parts = [BASE_PROMPT]
    for path in AGENTS_PATHS:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                parts.append(
                    f"## Project rules (AGENTS.md)\n{f.read().strip()}")
            break
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Session listing (module-level: used by CLI and the /sessions command)
# ---------------------------------------------------------------------------

def list_sessions() -> list[dict]:
    """Return [{id, title, updated_at}, ...] sorted newest-first."""
    if not os.path.isdir(SESSIONS_DIR):
        return []
    rows = []
    for name in os.listdir(SESSIONS_DIR):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(SESSIONS_DIR, name), "r", encoding="utf-8") as f:
                d = json.load(f)
            rows.append({
                "id": d.get("id", name[:-5]),
                "title": d.get("title", "Untitled"),
                "updated_at": d.get("updated_at", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(rows, key=lambda r: r["updated_at"], reverse=True)


# ---------------------------------------------------------------------------
# Agent — the brain. No UI.
# ---------------------------------------------------------------------------

class Agent:
    """Core agent: conversation loop, tool dispatch, and session persistence."""

    def __init__(self, workspace: str = ".", session_id: str | None = None):
        self.workspace = os.path.abspath(workspace)
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        os.makedirs(NOTES_DIR, exist_ok=True)
        if session_id:
            self.load_session(session_id)
        else:
            self._start_new()

    # --- session lifecycle -------------------------------------------------

    def _start_new(self) -> None:
        self.session_id = _gen_id()
        self.title = "Untitled"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.messages = [{"role": "system", "content": build_system_prompt()}]
        self._save_session()

    def load_session(self, session_id: str) -> None:
        """Resume an existing session; create a fresh one if the id is unknown."""
        path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if not os.path.isfile(path):
            self._start_new()
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.session_id = data.get("id", session_id)
        self.title = data.get("title", "Untitled")
        self.created_at = data.get(
            "created_at", datetime.now(timezone.utc).isoformat())
        self.messages = data.get("messages", [])
        # Refresh procedural memory: re-inject the *current* AGENTS.md so edits
        # to project rules take effect on resume (Cursor/Claude Code behaviour).
        sys_prompt = build_system_prompt()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = sys_prompt
        else:
            self.messages.insert(0, {"role": "system", "content": sys_prompt})

    def _save_session(self) -> None:
        """Persist the full message list (incl. tool calls/results) after every turn."""
        path = os.path.join(SESSIONS_DIR, f"{self.session_id}.json")
        data = {
            "id": self.session_id,
            "title": self.title,
            "created_at": getattr(self, "created_at", datetime.now(timezone.utc).isoformat()),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": self.messages,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # --- chat entry points -------------------------------------------------

    def chat(self, user_message: str, autotitle: bool = True) -> str:
        """One full turn: append user msg, run the tool loop, save. Optionally title.

        autotitle is skipped for one-shot CLI runs so the answer prints immediately
        instead of waiting on a second (cosmetic) model round-trip.
        """
        self.messages.append({"role": "user", "content": user_message})
        answer = self._run_loop()
        if autotitle:
            self._maybe_autotitle()
        self._save_session()
        return answer

    def run_once(self, prompt: str) -> str:
        return self.chat(prompt, autotitle=False)

    # --- core loop ---------------------------------------------------------

    def _complete(self, messages, use_tools: bool = True,
                  max_tokens: int = MAX_TOKENS, timeout: float | None = None):
        """Call the model with automatic failover across PRIORITY_MODELS.

        A 402/429/timeout from one free provider shouldn't kill the run — try the
        next model. The first model that succeeds is cached for the session.
        `timeout` overrides the client default for cheap/cosmetic calls.
        """
        global _MODEL_CACHE
        ordered = ([_MODEL_CACHE] if _MODEL_CACHE else []) + \
            [m for m in PRIORITY_MODELS if m != _MODEL_CACHE]
        last_err = None
        for name in ordered:
            try:
                kwargs = {"model": name, "messages": messages,
                          "max_tokens": max_tokens}
                if use_tools:
                    kwargs["tools"] = TOOLS
                if timeout is not None:
                    kwargs["timeout"] = timeout
                resp = client.chat.completions.create(**kwargs)
                if name != _MODEL_CACHE:
                    self._emit("model", name=name)
                _MODEL_CACHE = name
                return resp
            except Exception as e:  # noqa: BLE001
                last_err = e
                self._emit("model_error", name=name, error=str(e)[:160])
                continue
        raise RuntimeError(
            f"All free models failed (last: {last_err}). The OpenRouter free tier is "
            f"likely rate-limited or out of daily quota — retry later, or add a small "
            f"credit top-up / your own provider key on OpenRouter to raise limits."
        )

    def _stream(self, messages):
        """Stream one model turn with failover. Emits 'token' for content deltas.

        Returns (content, tool_calls) where tool_calls is a list of dicts in OpenAI
        tool-call shape (assembled from streamed fragments), or []. Failover only
        happens before the first chunk — once tokens have streamed, a mid-stream
        error is raised rather than silently restarting on another model.
        """
        global _MODEL_CACHE
        ordered = ([_MODEL_CACHE] if _MODEL_CACHE else []) + \
            [m for m in PRIORITY_MODELS if m != _MODEL_CACHE]
        last_err = None
        for name in ordered:
            started = False
            try:
                stream = client.chat.completions.create(
                    model=name, messages=messages, tools=TOOLS,
                    max_tokens=MAX_TOKENS, stream=True,
                )
                parts, frags = [], {}
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if not started:
                        started = True
                        if name != _MODEL_CACHE:
                            self._emit("model", name=name)
                        _MODEL_CACHE = name
                    if delta.content:
                        parts.append(delta.content)
                        self._emit("token", text=delta.content)
                    for tc in (delta.tool_calls or []):
                        slot = frags.setdefault(
                            tc.index, {"id": None, "name": None, "args": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["args"] += tc.function.arguments
                tool_calls = [
                    {"id": f["id"] or f"call_{i}", "type": "function",
                     "function": {"name": f["name"], "arguments": f["args"]}}
                    for i, (_, f) in enumerate(sorted(frags.items()))
                ]
                return "".join(parts), tool_calls
            except Exception as e:  # noqa: BLE001
                last_err = e
                self._emit("model_error", name=name, error=str(e)[:160])
                if started:
                    raise  # don't fail over mid-stream — would double-emit tokens
                continue
        raise RuntimeError(
            f"All free models failed (last: {last_err}). The OpenRouter free tier is "
            f"likely rate-limited or out of daily quota — retry later, or add a small "
            f"credit top-up / your own provider key on OpenRouter to raise limits."
        )

    def _run_loop(self) -> str:
        for _ in range(MAX_ITERATIONS):
            if STREAM:
                content, tool_calls = self._stream(self.messages)
            else:
                msg = self._complete(self.messages).choices[0].message
                content = msg.content or ""
                tool_calls = [tc.model_dump() for tc in (msg.tool_calls or [])]

            if not tool_calls:
                self.messages.append({"role": "assistant", "content": content})
                return content

            # Record the assistant's tool-call intent (dict shape for portable replay).
            self.messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                fname = tc["function"]["name"]
                self._emit("tool_call", name=fname)
                result = self.dispatch(tc)
                self._emit("tool_result", name=fname, chars=len(result))
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        return "Stopped: reached the maximum tool-iteration limit."

    def dispatch(self, tool_call) -> str:
        """Route a tool call to its handler. Accepts a dict (streamed) or an
        SDK object (non-stream). Always returns a JSON string."""
        if isinstance(tool_call, dict):
            fn = tool_call.get("function", {})
            name, raw_args = fn.get("name"), fn.get("arguments") or "{}"
        else:
            name, raw_args = tool_call.function.name, tool_call.function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON arguments from model."})

        handler = HANDLERS.get(name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool {name!r}"})
        try:
            return json.dumps(handler(**args))
        except TypeError as e:
            return json.dumps({"error": f"Bad arguments for {name}: {e}"})

    # --- presentation hook (overridden by REPL/TUI) ------------------------

    def _emit(self, event: str, **data) -> None:
        """No-op in the base brain. Subclasses log tool activity here."""

    # --- bonus: auto-title -------------------------------------------------

    def _maybe_autotitle(self) -> None:
        """After the first exchange, ask the model for a short session title."""
        if self.title != "Untitled":
            return
        first_user = next((m["content"]
                          for m in self.messages if m["role"] == "user"), None)
        if not first_user:
            return
        try:
            resp = self._complete(
                [{"role": "user",
                  "content": f"Give a 3-5 word title (no quotes, no punctuation) for a "
                  f"research session that starts with: {first_user}"}],
                use_tools=False,
                max_tokens=24,   # a title is a few tokens — keep it cheap
                timeout=20,      # cosmetic: never let it hang the turn
            )
            title = (resp.choices[0].message.content or "").strip().strip(
                '"').splitlines()[0]
            if title:
                self.title = title[:60]
        except Exception:
            pass  # title is cosmetic — never block a turn on it


# ---------------------------------------------------------------------------
# REPLAgent — terminal REPL + one-shot CLI
# ---------------------------------------------------------------------------

class REPLAgent(Agent):
    """Interactive terminal loop and one-shot runner. Inherits the whole brain."""

    COMMANDS = "/help  /sessions  /resume <id>  /new  /quit"

    def run(self) -> None:
        print(
            f"Research Desk [{self.session_id}: {self.title}] — model: auto (OpenRouter free)")
        print(f"Commands: {self.COMMANDS}\n")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("/quit", "/exit"):
                break
            if self._handle_command(line):
                continue
            self.chat(line)      # answer streams live via _emit('token')
            print("\n")          # newline + spacing after the streamed answer

    def _handle_command(self, line: str) -> bool:
        """Return True if `line` was a REPL command (handled here, not sent to the model)."""
        if not line.startswith("/"):
            return False
        parts = line.split()
        cmd, args = parts[0], parts[1:]

        if cmd == "/help":
            print(f"Commands: {self.COMMANDS}\n")
        elif cmd == "/sessions":
            rows = list_sessions()
            if not rows:
                print("No saved sessions.\n")
            else:
                for r in rows:
                    marker = "*" if r["id"] == self.session_id else " "
                    print(
                        f" {marker} {r['id']}  {r['title']}  ({r['updated_at'][:19]})")
                print()
        elif cmd == "/resume":
            if not args:
                print("Usage: /resume <id>\n")
            else:
                self.load_session(args[0])
                print(f"Resumed [{self.session_id}: {self.title}]\n")
        elif cmd == "/new":
            self._start_new()
            print(f"New session [{self.session_id}]\n")
        else:
            print(f"Unknown command. {self.COMMANDS}\n")
        return True

    def _emit(self, event: str, **data) -> None:
        if event == "token":
            print(data.get("text", ""), end="", flush=True)
        elif event == "tool_call":
            print(f"  [tool] {data.get('name')}", file=sys.stderr)
        elif event == "tool_result":
            print(
                f"  [done] {data.get('name')} ({data.get('chars')} chars)", file=sys.stderr)
        elif event == "model":
            print(f"  [model] using {data.get('name')}", file=sys.stderr)
        elif event == "model_error":
            print(
                f"  [model] {data.get('name')} failed, trying next… ({data.get('error')})", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Research Desk agent")
    parser.add_argument("question", nargs="*",
                        help="One-shot question (omit for REPL).")
    parser.add_argument("--tui", action="store_true",
                        help="Launch the Textual UI.")
    parser.add_argument("--session", metavar="ID",
                        help="Resume an existing session.")
    args = parser.parse_args()

    if args.tui:
        # Imported lazily so agent.py stays Textual-free.
        from tui import TUIAgent
        TUIAgent(session_id=args.session).run()
        return

    agent = REPLAgent(session_id=args.session)

    if args.question:
        # answer streams via _emit('token')
        agent.run_once(" ".join(args.question))
        print()                                   # trailing newline
        return

    agent.run()


if __name__ == "__main__":
    main()
