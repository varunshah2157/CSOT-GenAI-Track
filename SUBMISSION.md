# Week 1 Submission: GenAI/Agentic Track

**Name:** Varun Shah
**Entry Number:** 2025MT11346

---

## Project Overview

This repository contains the foundational components for a multi-turn LLM agent, built from scratch without high-level wrappers like LangChain. The project strictly adheres to API security best practices and handles the stateless nature of LLM endpoints through manual conversation buffering and context compaction.

### Core Requirements Met
- **API Key Hygiene:** Keys are strictly managed via `python-dotenv` and isolated in a `.env` file (which is included in `.gitignore`). No keys are hardcoded.
- **Model Agnosticism:** Configured to dynamically connect to OpenRouter's free-tier models, iterating through a priority list to ensure maximum uptime.
- **Manual State Management:** The chat loop maintains a strictly formatted array of `system`, `user`, and `assistant` dictionaries to persist context across the stateless API.

---

## Build Features

### Build 1: Single-Turn Architecture
- **Pre-Flight Endpoint Verification:** Implemented a ping loop to verify model availability before attempting the primary generation.
- **Real-Time Streaming:** Utilized `stream=True` and I/O buffer flushing (`flush=True`) to deliver tokens to the console instantaneously.
- **Metadata Extraction:** Leveraged `stream_options={"include_usage": True}` to capture and display precise token metrics (Prompt, Completion, Total) from the final chunk of the stream.

### Build 2: Multi-Turn Memory & Compaction (FINAL SUBMISSION)
- **Context Preservation:** Successfully engineered a loop that appends and sends the full conversational history on every interaction.
- **Dynamic Compaction:** Implemented a rolling buffer safeguard. When the conversation exceeds the `MAX_TURNS` threshold, a background API call automatically summarizes the middle turns into a condensed context block to prevent token overflow while preserving the system prompt and recent history.
- **Interactive CLI Commands:**
  - `/reset`: Wipes the memory buffer to seamlessly demonstrate context loss and the stateless API problem.
  - `/tokens`: Retrieves and prints the token usage of the most recent data stream.
  - `/compact`: Allows the user to manually trigger the background summarization logic.
- **Graceful Error Handling:** If a stream fails mid-generation, the agent automatically drops the most recent `user` turn from the buffer to strictly preserve role alternation rules.
