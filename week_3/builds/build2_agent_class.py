"""
Build 2: Agent + REPLAgent
===========================
Agent = brain (loop, tools, sessions). REPLAgent = terminal UI.

Before running:
  mkdir -p notes

Tasks:
  1. Agent — chat(), run_once(), _run_loop(), dispatch(), _emit(), session I/O
  2. REPLAgent(Agent) — run() interactive loop
  3. resolve_path, read_file, write_file, list_files, edit_file
  4. main() — one-shot: python build2_agent_class.py "hello"

TUIAgent comes in the project (tui.py). No Textual imports here.
"""

import os
import sys
import json
import glob as glob_module
import uuid
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

WORKSPACE_ROOT = os.path.abspath(os.environ.get("WORKSPACE_ROOT", "."))
MAX_ITERATIONS = 10
MAX_READ_CHARS = 12_000
SESSIONS_DIR = ".agent/sessions"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
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
            client.chat.completions.create(
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


# --- File tools ---

def resolve_path(path: str) -> str:
    # Ensure path stays within WORKSPACE_ROOT (Sandboxing)
    full_path = os.path.abspath(os.path.join(WORKSPACE_ROOT, path))
    if not full_path.startswith(WORKSPACE_ROOT):
        raise ValueError(f"Path {path} escapes workspace {WORKSPACE_ROOT}")
    return full_path


def read_file(path: str, start_line: int = 1, read_lines: int = 200) -> dict:
    try:
        full_path = resolve_path(path)
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total = len(lines)
        start_idx = max(0, start_line - 1)
        end_idx = min(total, start_idx + read_lines)

        # Prepend line numbers for targeted editing
        chunk = [
            f"{i + start_idx + 1:4d}| {lines[i].rstrip()}" for i in range(start_idx, end_idx)]
        content = "\n".join(chunk)

        if len(content) > MAX_READ_CHARS:
            content = content[:MAX_READ_CHARS] + "\n...[TRUNCATED]"

        return {"content": content, "total_lines": total, "has_more": end_idx < total}
    except Exception as e:
        return {"error": str(e)}


def write_file(path: str, content: str) -> dict:
    try:
        full_path = resolve_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"content": f"Successfully wrote to {path}"}
    except Exception as e:
        return {"error": str(e)}


def edit_file(
    path: str,
    operation: str,
    start_line: int,
    end_line: int | None = None,
    content: str | None = None,
) -> dict:
    try:
        full_path = resolve_path(path)
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        start_idx = max(0, start_line - 1)
        new_lines = [l + "\n" for l in content.split("\n")] if content else []

        if operation == "replace":
            end_idx = min(len(lines), end_line) if end_line else start_idx + 1
            lines[start_idx:end_idx] = new_lines
        elif operation == "delete":
            end_idx = min(len(lines), end_line) if end_line else start_idx + 1
            del lines[start_idx:end_idx]
        elif operation == "append":
            # 0 = before line 1. start_line is exactly the index to insert at
            insert_idx = max(0, start_line)
            # Ensure the line right before our insertion point has a newline
            if 0 < insert_idx <= len(lines):
                if not lines[insert_idx - 1].endswith('\n'):
                    lines[insert_idx - 1] += '\n'
            lines[insert_idx:insert_idx] = new_lines
        else:
            return {"error": f"Unknown operation: {operation}"}

        with open(full_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Return a quick diff/status preview
        preview = "".join(new_lines[:5]) + \
            ("..." if len(new_lines) > 5 else "")
        return {"content": f"Operation '{operation}' successful. Preview of changes:\n{preview}"}
    except Exception as e:
        return {"error": str(e)}


def list_files(path: str = ".", pattern: str = "*") -> dict:
    try:
        full_path = resolve_path(path)
        search_pattern = os.path.join(full_path, pattern)
        # Use recursive glob to find files
        files = glob_module.glob(search_pattern, recursive=True)
        # Return relative paths to keep context clean
        rel_files = [os.path.relpath(f, WORKSPACE_ROOT)
                     for f in files if os.path.isfile(f)]
        return {"content": "\n".join(rel_files) if rel_files else "No files found."}
    except Exception as e:
        return {"error": str(e)}


# Schema definitions for OpenRouter/OpenAI tool routing
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file with line numbers. Use start_line and read_lines for pagination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "default": 1},
                    "read_lines": {"type": "integer", "default": 200}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or completely overwrite a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit an existing file using line numbers. Operations: replace, delete, append.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "operation": {"type": "string", "enum": ["replace", "delete", "append"]},
                    "start_line": {
                        "type": "integer",
                        "description": "Line number. For 'append', it inserts AFTER this line (e.g., set to 0 to prepend at the top, or set to the total number of lines to append to the very end)."
                    },
                    "end_line": {"type": "integer"},
                    "content": {"type": "string"}
                },
                "required": ["path", "operation", "start_line"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory using a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "pattern": {"type": "string", "default": "*"}
                }
            }
        }
    }
]


class Agent:
    """Core agent: loop, tools, sessions. No UI."""

    def __init__(self, workspace: str = ".", session_id: str | None = None):
        self.workspace = os.path.abspath(workspace)
        # TODO: session_id, load messages
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        if session_id:
            self.session_id = session_id
            file_path = os.path.join(SESSIONS_DIR, f"{self.session_id}.json")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    self.messages = json.load(f).get("messages", [])
            else:
                self.messages = [
                    {"role": "system", "content": build_system_prompt()}]
        else:
            self.session_id = uuid.uuid4().hex[:8]
            self.messages = [
                {"role": "system", "content": build_system_prompt()}]
            self._save_session()

    def _save_session(self):
        """Internal helper to persist session to disk."""
        file_path = os.path.join(SESSIONS_DIR, f"{self.session_id}.json")
        now_str = datetime.now(timezone.utc).isoformat()
        session_data = {
            "id": self.session_id,
            "updated_at": now_str,
            "messages": self.messages
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

    def chat(self, user_message: str) -> str:
        # TODO: append user msg, _run_loop(), save session, return answer
        self.messages.append({"role": "user", "content": user_message})
        response = self._run_loop()
        self._save_session()
        return response

    def run_once(self, prompt: str) -> str:
        return self.chat(prompt)

    def _run_loop(self) -> str:
        # TODO: agent loop — call self.dispatch(), self._emit() on tool calls
        for _ in range(MAX_ITERATIONS):
            response = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=TOOLS
            )
            msg = response.choices[0].message

            # If standard text response, we are done
            if not msg.tool_calls:
                self.messages.append(
                    {"role": "assistant", "content": msg.content})
                return msg.content or ""

            # Store the assistant's intent to call tools
            self.messages.append(msg.model_dump())

            # Handle tool calls
            for tool_call in msg.tool_calls:
                self._emit("tool_call", name=tool_call.function.name)
                result_str = self.dispatch(tool_call)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str
                })

        return "Error: Maximum iterations reached."

    def dispatch(self, tool_call) -> str:
        # TODO: route to file tools, return JSON string
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON arguments generated."})

        result = {"error": f"Tool {name} not found"}
        if name == "read_file":
            result = read_file(**args)
        elif name == "write_file":
            result = write_file(**args)
        elif name == "edit_file":
            result = edit_file(**args)
        elif name == "list_files":
            result = list_files(**args)

        return json.dumps(result)

    def _emit(self, event: str, **data) -> None:
        """Override in REPLAgent/TUIAgent for tool logging."""
        pass


class REPLAgent(Agent):
    """Terminal REPL + one-shot CLI."""

    def run(self) -> None:
        print(f"Research Desk [{self.session_id}] — /quit to exit")
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input or user_input in ("/quit", "/exit"):
                break
            print(self.chat(user_input))
            print()

    def _emit(self, event: str, **data) -> None:
        if event == "tool_call":
            print(f"  [tool] {data.get('name')}", file=sys.stderr)


def build_system_prompt() -> str:
    base = "You are Research Desk, a helpful research assistant."
    parts = [base]
    for path in ("AGENTS.md", ".agent/AGENTS.md"):
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f"## Project rules\n{f.read()}")
            break
    return "\n\n".join(parts)


def main():
    # If a session ID is provided via CLI flag (e.g. --session id), we would parse it here.
    # For now, default to creating a new REPLAgent.
    agent = REPLAgent()
    if len(sys.argv) > 1:
        print(agent.run_once(" ".join(sys.argv[1:])))
        return
    agent.run()


if __name__ == "__main__":
    main()
