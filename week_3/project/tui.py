"""
TUIAgent — full-screen Textual UI inheriting the Agent brain.

Two classes:
  TUIAgent(Agent)        — overrides _emit() to stream tool activity into the UI;
                           run() boots the app passing itself in.
  ResearchDeskApp(App)   — layout, input, key bindings. on_input_submitted runs
                           the (blocking) inherited chat() in a thread worker.

The brain is unchanged: TUIAgent.chat() *is* Agent.chat(). Only presentation
differs. Launch with:  python agent.py --tui
"""

import os
import sys

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Header, Footer, Input, RichLog, Static
from rich.text import Text
from rich.markup import escape

from agent import Agent


class TUIAgent(Agent):
    """Agent whose tool activity is mirrored into a Textual log panel."""

    def __init__(self, workspace: str = ".", session_id: str | None = None):
        super().__init__(workspace=workspace, session_id=session_id)
        self._app: "ResearchDeskApp | None" = None

    def run(self) -> None:
        self._app = ResearchDeskApp(self)
        self._app.run()

    def _emit(self, event: str, **data) -> None:
        # Called from the worker thread → marshal onto the UI thread.
        if self._app is None:
            return
        if event == "token":
            self._app.call_from_thread(
                self._app.stream_token, data.get("text", ""))
        elif event == "tool_call":
            self._app.call_from_thread(
                self._app.write_tool, f"→ [magenta]{data.get('name')}[/magenta]")
        elif event == "tool_result":
            self._app.call_from_thread(
                self._app.write_tool,
                f"  [dim]↳ {data.get('name')} ({data.get('chars')} chars)[/dim]",
            )
        elif event == "model":
            self._app.call_from_thread(
                self._app.write_tool, f"[green]model: {data.get('name')}[/green]")
        elif event == "model_error":
            self._app.call_from_thread(
                self._app.write_tool, f"[red]model {data.get('name')} failed → next[/red]")


class ResearchDeskApp(App):
    TITLE = "Research Desk"

    CSS = """
    Screen { layout: vertical; }
    Horizontal { height: 1fr; }
    #chat-panel { width: 65%; border: solid $primary; padding: 0 1; }
    #tool-panel { width: 35%; border: solid $warning; padding: 0 1; }
    Input { dock: bottom; height: 3; }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_display", "Clear Display"),
        Binding("ctrl+k", "clear_history", "Reset History"),
        Binding("ctrl+n", "new_session", "New Session"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, agent: TUIAgent):
        super().__init__()
        self.agent = agent
        self._chat_text = "[bold green]Research Desk ready.[/bold green]\n"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with VerticalScroll(id="chat-panel"):
                yield Static(self._chat_text, id="chat-text", markup=True)
            yield RichLog(id="tool-panel", wrap=True, markup=True)
        yield Input(placeholder="Ask a research question and press Enter…")
        yield Footer()

    def on_mount(self) -> None:
        self.chat_view = self.query_one("#chat-panel", VerticalScroll)
        self.chat_static = self.query_one("#chat-text", Static)
        self.tool_log = self.query_one("#tool-panel", RichLog)
        self.tool_log.write("[bold yellow]Tool Activity[/bold yellow]\n")
        self.sub_title = f"{self.agent.session_id}: {self.agent.title}"
        self.query_one(Input).focus()

    # --- UI-thread helpers (safe to call via call_from_thread) -------------

    def _append_chat(self, markup: str) -> None:
        self._chat_text += markup
        self.chat_static.update(Text.from_markup(self._chat_text))
        self.chat_view.scroll_end(animate=False)

    def write_tool(self, markup: str) -> None:
        self.tool_log.write(markup)

    def stream_token(self, text: str) -> None:
        # tokens are plain text — escape so stray brackets don't break markup
        self._chat_text += escape(text)
        self.chat_static.update(Text.from_markup(self._chat_text))
        self.chat_view.scroll_end(animate=False)

    # --- input → worker → inherited chat() ---------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()
        self._append_chat(f"\n[bold cyan][You][/bold cyan] {escape(text)}\n")
        self.sub_title = f"{self.agent.session_id} (thinking…)"
        self.run_worker(lambda: self._respond(text),
                        thread=True, exclusive=True)

    def _respond(self, text: str) -> None:
        # Runs in a worker thread. chat() is the *inherited* Agent.chat();
        # its answer streams in token-by-token via TUIAgent._emit('token').
        self.call_from_thread(
            self._append_chat, "\n[bold green][Agent][/bold green] ")
        try:
            self.agent.chat(text)
        except Exception as e:
            self.call_from_thread(
                self._append_chat, f"[red][error] {escape(str(e))}[/red]")
        self.call_from_thread(self._append_chat, "\n")
        self.call_from_thread(setattr, self, "sub_title",
                              f"{self.agent.session_id}: {self.agent.title}")

    # --- key bindings ------------------------------------------------------

    def action_clear_display(self) -> None:
        """Ctrl+L — clear the visible panels; keep conversation history intact."""
        self._chat_text = "[dim]Display cleared. History intact.[/dim]\n"
        self.chat_static.update(Text.from_markup(self._chat_text))
        self.tool_log.clear()
        self.tool_log.write("[bold yellow]Tool Activity[/bold yellow]\n")

    def action_clear_history(self) -> None:
        """Ctrl+K — reset the conversation (keep the system prompt) and persist."""
        self.agent.messages = [self.agent.messages[0]]
        self.agent._save_session()
        self._chat_text = "[bold yellow]History reset.[/bold yellow]\n"
        self.chat_static.update(Text.from_markup(self._chat_text))
        self.tool_log.clear()
        self.tool_log.write("[bold yellow]Tool Activity[/bold yellow]\n")

    def action_new_session(self) -> None:
        """Ctrl+N — start a fresh session on disk."""
        self.agent._start_new()
        self._chat_text = f"[bold green]New session {self.agent.session_id}.[/bold green]\n"
        self.chat_static.update(Text.from_markup(self._chat_text))
        self.tool_log.clear()
        self.tool_log.write("[bold yellow]Tool Activity[/bold yellow]\n")
        self.sub_title = f"{self.agent.session_id}: {self.agent.title}"


if __name__ == "__main__":
    TUIAgent(session_id=(sys.argv[1] if len(sys.argv) > 1 else None)).run()
