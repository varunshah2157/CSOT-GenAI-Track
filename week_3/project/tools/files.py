"""
Sandboxed file tools — OpenCode-style (see week_3/2_agent_class.md).

Public API
----------
- resolve_path(path)                                  -> absolute path, sandboxed to WORKSPACE_ROOT
- read_file(path, start_line=1, read_lines=200)       -> {content, start_line, end_line, total_lines, has_more} | {error}
- write_file(path, content)                           -> {content} | {error}
- edit_file(path, operation, start_line, end_line?, content?)
                                                      -> {content (incl. diff preview)} | {error}
- list_files(path=".", pattern="*")                   -> {content} | {error}

Every tool returns a dict with either a "content" key (success) or an "error"
key (failure). The agent's dispatch() json.dumps() the dict into the tool
message, so the model always sees one of two predictable shapes and can retry
on "error" instead of crashing on a traceback.

Each module owns its own schema + handler map so agent.py can merge tools
without a growing if/elif chain (OpenCode-inspired: a tool is a self-contained
unit of schema + implementation).
"""

import os
import glob as glob_module

# Sandbox root. Every path the model touches must resolve inside this folder.
WORKSPACE_ROOT = os.path.abspath(os.environ.get("WORKSPACE_ROOT", "."))

MAX_READ_CHARS = 12_000  # safety cap even when read_lines is generous


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

def resolve_path(path: str) -> str:
    """Resolve `path` under WORKSPACE_ROOT, refusing anything that escapes it.

    Uses a separator-aware prefix check so that a sibling directory like
    ``<root>-evil`` cannot masquerade as being inside ``<root>``.
    """
    full_path = os.path.abspath(os.path.join(WORKSPACE_ROOT, path))
    if full_path != WORKSPACE_ROOT and not full_path.startswith(WORKSPACE_ROOT + os.sep):
        raise ValueError(f"Path {path!r} escapes workspace {WORKSPACE_ROOT}")
    return full_path


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_file(path: str, start_line: int = 1, read_lines: int = 200) -> dict:
    """Read a window of a file, prefixed with 1-indexed line numbers.

    Returns metadata (total_lines, has_more, the range actually returned) so the
    model knows whether to page again and which line numbers to pass to edit_file.
    """
    try:
        full_path = resolve_path(path)
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total = len(lines)
        start_idx = max(0, start_line - 1)
        end_idx = min(total, start_idx + max(1, read_lines))

        numbered = [
            f"{i + 1:4d}| {lines[i].rstrip(chr(10))}" for i in range(start_idx, end_idx)
        ]
        content = "\n".join(numbered)

        truncated = False
        if len(content) > MAX_READ_CHARS:
            content = content[:MAX_READ_CHARS] + \
                "\n...[TRUNCATED — narrow read_lines or page with start_line]"
            truncated = True

        return {
            "content": content,
            "start_line": start_idx + 1,
            "end_line": end_idx,
            "total_lines": total,
            "has_more": end_idx < total or truncated,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_file(path: str, content: str) -> dict:
    """Create or completely overwrite a file. Use edit_file to amend existing files."""
    try:
        full_path = resolve_path(path)
        parent = os.path.dirname(full_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        n = content.count("\n") + \
            (1 if content and not content.endswith("\n") else 0)
        return {"content": f"Wrote {n} line(s) to {path}"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Edit (line-based: replace | delete | append) with diff preview
# ---------------------------------------------------------------------------

def _diff_preview(removed: list[str], added: list[str], limit: int = 8) -> str:
    """Render a compact unified-style -/+ preview of an edit."""
    out = []
    for ln in removed[:limit]:
        out.append(f"- {ln.rstrip(chr(10))}")
    if len(removed) > limit:
        out.append(f"- ...(+{len(removed) - limit} more removed)")
    for ln in added[:limit]:
        out.append(f"+ {ln.rstrip(chr(10))}")
    if len(added) > limit:
        out.append(f"+ ...(+{len(added) - limit} more added)")
    return "\n".join(out) if out else "(no line changes)"


def edit_file(
    path: str,
    operation: str,
    start_line: int,
    end_line: int | None = None,
    content: str | None = None,
) -> dict:
    """Line-level edit. Operations:

    - replace : lines [start_line..end_line] (inclusive, 1-indexed) -> content
    - delete  : remove lines [start_line..end_line] (inclusive)
    - append  : insert content AFTER start_line (0 = before line 1)

    Returns a diff preview so the edit is verifiable in the tool log.
    """
    try:
        full_path = resolve_path(path)
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Normalise new content into newline-terminated lines.
        new_lines = [l + "\n" for l in content.splitlines()] if content else []

        if operation == "replace":
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line if end_line else start_line)
            removed = lines[start_idx:end_idx]
            lines[start_idx:end_idx] = new_lines
            added = new_lines

        elif operation == "delete":
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line if end_line else start_line)
            removed = lines[start_idx:end_idx]
            del lines[start_idx:end_idx]
            added = []

        elif operation == "append":
            insert_idx = max(0, min(start_line, len(lines)))
            # ensure the preceding line is newline-terminated so we don't fuse lines
            if 0 < insert_idx <= len(lines) and not lines[insert_idx - 1].endswith("\n"):
                lines[insert_idx - 1] += "\n"
            lines[insert_idx:insert_idx] = new_lines
            removed = []
            added = new_lines

        else:
            return {"error": f"Unknown operation {operation!r} (use replace|delete|append)"}

        with open(full_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        preview = _diff_preview(removed, added)
        return {
            "content": f"{operation} on {path} OK. New length: {len(lines)} lines.\n--- diff ---\n{preview}"
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# List (glob)
# ---------------------------------------------------------------------------

def list_files(path: str = ".", pattern: str = "*") -> dict:
    """List files under `path` matching a glob `pattern` (supports ** recursion)."""
    try:
        full_path = resolve_path(path)
        search = os.path.join(full_path, pattern)
        matches = glob_module.glob(search, recursive=True)
        rel = sorted(
            os.path.relpath(m, WORKSPACE_ROOT) for m in matches if os.path.isfile(m)
        )
        return {"content": "\n".join(rel) if rel else "No files found."}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Schemas + handler map (consumed by agent.py)
# ---------------------------------------------------------------------------

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file with 1-indexed line numbers. Use start_line + read_lines to "
                "page large files; check 'has_more' and read the next window if true. "
                "Read a file immediately before editing it so line numbers are fresh."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "start_line": {"type": "integer", "description": "First line to read (1-indexed).", "default": 1},
                    "read_lines": {"type": "integer", "description": "How many lines to return.", "default": 200},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or completely overwrite a file. Use for NEW notes; use edit_file to amend existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Edit an existing file by line number. Returns a diff preview. "
                "replace/delete use start_line..end_line (inclusive). "
                "append inserts AFTER start_line (use 0 to prepend, or total_lines to add at the end)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "operation": {"type": "string", "enum": ["replace", "delete", "append"]},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer", "description": "Inclusive end line for replace/delete."},
                    "content": {"type": "string", "description": "New text for replace/append (split on newlines)."},
                },
                "required": ["path", "operation", "start_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory using a glob pattern (e.g. list_files('notes', '*.md')). Use to recall past notes before reading them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "pattern": {"type": "string", "default": "*"},
                },
            },
        },
    },
]

HANDLERS = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "list_files": list_files,
}
