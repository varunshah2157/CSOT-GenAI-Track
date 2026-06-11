"""
Build 3: Extend Your Week 1 Chatbot into a TUI
===============================================
Take the multi-turn chatbot you built in Week 1 and give it a full-screen terminal UI
using Textual. The chat logic stays the same; you're just changing the interface.

Requirements:
  - A scrollable chat log that shows conversation history
  - An input box at the bottom for the user to type
  - Keyboard shortcuts:
      Ctrl+L  →  clear the chat display (not the conversation history)
      Ctrl+K  →  compact: clear conversation history too (fresh start)
      Ctrl+Q  →  quit the application
  - Messages displayed with clear role labels: [You] and [Agent]
  - The UI must not freeze while waiting for an API response

Stretch goals:
  - Show the model name and token count in the Header or Footer
  - Add a Ctrl+S binding to save the conversation to a text file
  - Display a "thinking..." indicator while the API call is in progress

Important: API calls are blocking. Use run_worker(thread=True) to keep the UI alive
while waiting for responses. See Lesson 4 for the pattern.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, Input, RichLog
from textual.containers import Vertical

load_dotenv()

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
MAX_HISTORY_TURNS = 20   # keep last N user+assistant pairs

# ---------------------------------------------------------------------------
# Chat logic (reuse / adapt from your Week 1 submission)
# ---------------------------------------------------------------------------


def call_model(messages: list[dict]) -> str:
    """
    Send the full messages list to the model and return the assistant's reply text.
    This is a blocking call. It must run in a worker thread in the TUI.
    """
    # TODO: implement using client.chat.completions.create()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


def trim_history(messages: list[dict], max_turns: int) -> list[dict]:
    """
    Keep the system message and only the last `max_turns` user/assistant pairs.

    messages[0] is assumed to be the system message.
    Drop oldest pairs from messages[1:] when over the limit.
    A 'pair' is one user message + one assistant message = 2 entries.
    """
    # TODO: implement
    if len(messages) <= 1:
        return messages
    system_message = messages[0]
    chat_history = messages[1:]
    max_elements = max_turns * 2
    if len(chat_history) > max_elements:
        chat_history = chat_history[-max_elements:]
    return [system_message] + chat_history


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

class ChatApp(App):
    """A full-screen terminal chatbot."""

    TITLE = "Week 2 Chatbot TUI"
    CSS = """
    Screen {
        layout: vertical;
    }

    RichLog {
        height: 1fr;
        border: solid $primary;
        padding: 0 1;
    }

    Input {
        dock: bottom;
        height: 3;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_display", "Clear display"),
        Binding("ctrl+k", "clear_history", "Clear history"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.messages: list[dict] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="log", wrap=True, markup=True, highlight=True)
        yield Input(placeholder="Type a message and press Enter...")
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#log", RichLog)
        log.write(
            "[bold green]Chat started.[/bold green] Ctrl+Q to quit, Ctrl+L to clear.\n")
        self.query_one(Input).focus()

    # -----------------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Called when the user presses Enter."""
        user_text = event.value.strip()
        if not user_text:
            return

        event.input.clear()

        log = self.query_one("#log", RichLog)
        log.write(f"[bold cyan][You][/bold cyan] {user_text}\n")

        # Append user message to history
        self.messages.append({"role": "user", "content": user_text})
        self.messages = trim_history(self.messages, MAX_HISTORY_TURNS)

        # Run the API call in a background thread so the UI stays responsive
        self.run_worker(self._get_response(), thread=True)

    async def _get_response(self) -> None:
        """
        Fetch the model response and update the UI.
        This runs in a background thread (called via run_worker).

        Steps:
          1. Call call_model(self.messages)  [blocking, OK in a thread]
          2. Append the assistant reply to self.messages
          3. Use self.call_from_thread(log.write, ...) to update the UI safely

        Handle exceptions: if call_model raises, display an error in the log.
        """
        log = self.query_one("#log", RichLog)
        try:
            reply = call_model(self.messages)
            self.messages.append({"role": "assistant", "content": reply})
            self.call_from_thread(
                log.write, f"[bold green][Agent][/bold green] {reply}\n")
        except Exception as e:
            self.call_from_thread(
                log.write, f"[bold red][Error][/bold red] Failed to fetch response: {e}\n")

    # -----------------------------------------------------------------------
    # Actions (bound to keyboard shortcuts)
    # -----------------------------------------------------------------------

    def action_clear_display(self) -> None:
        """Clear the visible log without touching conversation history."""
        self.query_one("#log", RichLog).clear()

    def action_clear_history(self) -> None:
        """Reset conversation history and clear the display."""
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        log = self.query_one("#log", RichLog)
        log.clear()
        log.write("[bold yellow]History cleared.[/bold yellow]\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ChatApp().run()
