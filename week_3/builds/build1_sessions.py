"""
Build 1: Session Store
========================
Save and resume conversations on disk. Load AGENTS.md into the system prompt.

Tasks:
  1. create_session() -> session_id
  2. save_session(session_id, messages, title?)
  3. load_session(session_id) -> {id, title, messages, ...}
  4. list_sessions() -> [{id, title, updated_at}, ...]
  5. build_system_prompt() -> base + AGENTS.md contents

Run twice: save a session in run 1, load it in run 2 and confirm messages restored.
"""

import json
import os
import uuid
from datetime import datetime, timezone

SESSIONS_DIR = ".agent/sessions"
AGENTS_PATHS = ("AGENTS.md", ".agent/AGENTS.md")

BASE_PROMPT = "You are Research Desk, a helpful research assistant."


def create_session() -> str:
    """Return a new 8-char hex session ID."""
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    # TODO: initiate a new, empty session with a unique ID
    session_id = uuid.uuid4().hex[:8]  # Generate 8-char hex ID
    save_session(session_id, [], title="Untitled")  # Initialize file
    return session_id


def save_session(session_id: str, messages: list, title: str = "Untitled") -> None:
    """Write session JSON to .agent/sessions/{id}.json"""
    # TODO: implement
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    now_str = datetime.now(timezone.utc).isoformat()

    # Preserve original creation time if file already exists
    created_at = now_str
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                created_at = json.load(f).get("created_at", now_str)
        except json.JSONDecodeError:
            pass  # Fallback to current time if file is corrupted

    session_data = {
        "id": session_id,
        "title": title,
        "created_at": created_at,
        "updated_at": now_str,
        "messages": messages
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)


def load_session(session_id: str) -> dict:
    """Load and return session dict including messages list."""
    # TODO: implement
    file_path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Session {session_id} not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_sessions() -> list[dict]:
    """Return sessions sorted by updated_at descending."""
    # TODO: implement
    if not os.path.exists(SESSIONS_DIR):
        return []

    sessions = []
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(SESSIONS_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Extract only the metadata needed for listing
                    sessions.append({
                        "id": data.get("id"),
                        "title": data.get("title", "Untitled"),
                        "updated_at": data.get("updated_at")
                    })
            except (json.JSONDecodeError, IOError):
                continue  # Skip corrupted or unreadable files

    # Sort descending by timestamp (ISO format strings sort lexicographically)
    return sorted(sessions, key=lambda s: s.get("updated_at", ""), reverse=True)


def build_system_prompt() -> str:
    """Base prompt + AGENTS.md if it exists."""
    # TODO: implement
    system_prompt = BASE_PROMPT
    for path in AGENTS_PATHS:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                # Append the contents of the file
                system_prompt += f"\n\n{f.read().strip()}"
            break  # Stop searching once we find the first valid config

    return system_prompt


if __name__ == "__main__":
    # sid = create_session()
    # messages = [
    #     {"role": "system", "content": build_system_prompt()},
    #     {"role": "user", "content": "What is a surface code?"},
    #     {"role": "assistant",
    #         "content": "A surface code is a type of quantum error correcting code."},
    # ]
    # save_session(sid, messages, title="Quantum error correction")
    # print(f"Saved session: {sid}")
    # print(f"All sessions: {list_sessions()}")
    # print(f"Loaded: {load_session(sid)['title']}")

    # 1. Hardcode the ID from your previous run
    past_session_id = "8dbe5215"

    # 2. Load it directly from the hard drive
    restored_session = load_session(past_session_id)

    # 3. Print the results to prove it remembered
    print(f"Resumed Chat: {restored_session['title']}\n")
    print("--- Conversation History ---")

    for msg in restored_session['messages']:
        # Truncating the content just to keep the terminal output clean
        print(f"{msg['role'].upper()}: {msg['content']}...")
