# Week 2 Submission: Agentic Research Engine & Tool Integration

**Name:** Varun Shah  
**Entry Number:** 2025MT11346  

---

## 1. Implemented Features
- **Web Search Tool:** Integrated real-time web querying via Serper API.
- **Web Extraction Engine:** Developed direct URL fetching and HTML-to-text sanitization.
- **Persistent Notes Tooling:** Implemented structured file writing for automated research artifact generation in `/notes`.
- **Academic Paper Integration (alphaXiv MCP):** Connected to academic repositories via Model Context Protocol (MCP) with OAuth authentication to retrieve scholarly papers.
- **Split-Panel Textual TUI:** Dual-pane interface with responsive keyboard shortcuts, status monitors, and real-time tool execution tracking.
- **Real-Time Streaming:** Buffered chunk-by-chunk token output integrated with TUI layout stability.
- **Fault-Tolerant Model Fallback:** Multi-tier model priority hierarchy automatically redirecting failed requests without breaking conversational context.

---

## 2. Agent Execution Mechanics
1. **Dynamic Model Routing:** Automatically evaluates endpoint availability against a priority list, failing over gracefully if primary endpoints throttle or error.
2. **ReAct Feedback Loop:** Bounded iterative loop (maximum 10 iterations) evaluating tool invocation requirements against conversational state.
3. **Tool Execution & Context Re-injection:** Dispatches tool calls, captures outputs, appends results to conversation state, and generates the final synthesized response.
4. **State Streaming & Monitoring:** Renders output tokens in real-time on the main panel while displaying tool status and telemetry on the side panel.

---

## 3. Engineering Assessment & Takeaways
- Implemented protocol design for Model Context Protocol (MCP) clients and OAuth token handshakes for remote tools.
- Refactored asynchronous streaming within a stateful TUI to prevent render tearing and layout jitter across tool invocations.
- Engineered resilient API fallback mechanisms to maintain session continuity during upstream rate limits or outages.
