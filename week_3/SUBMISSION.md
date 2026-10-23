# Week 3 Submission: Multi-Mode Agent Architecture & Session Persistence

**Name:** Varun Shah  
**Entry Number:** 2025MT11346  

---

## 1. Build Architectures

### Build 1: Persistent Session Engine
- Designed UUID-based session identification, serialization, and disk persistence to JSON.
- Engineered state restoration allowing resuming previous conversational contexts with role ordering preserved.

### Build 2: File System Tooling & Interaction Modes
- Implemented core file-manipulation primitives (`read_file`, `write_file`, `edit_file`, `list_files`).
- Supported dual invocation modalities: single-turn execution (`run_once`) and interactive Read-Eval-Print Loop (REPL).

### Project: Production Multi-Mode Research Agent
- **Modular Object-Oriented Architecture:** Abstracted agent capabilities into structured classes encapsulating state, memory, and tool dispatching.
- **Multi-Modal User Interface:** Engineered support for Textual TUI (`tui.py`), CLI arguments, and an interactive REPL shell.
- **Storage Layout:** Structured filesystem layout separating operational sessions (`.agent/sessions`), generated research artifacts (`notes/`), and dynamic tools (`tools/`).
- **Comprehensive Verification:** Validated all REPL commands, session persistence, auto-titling, and tool execution across edge cases.
