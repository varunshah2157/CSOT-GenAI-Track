"""
Build 1: Custom Tool Call Parser
=================================
Before modern SDKs handled tool calls natively, developers used custom text formats
that the model was prompted to emit. This build has you implement that pattern from
scratch: prompt the model to emit tool calls in a structured format, parse them, run
the corresponding Python function, and feed the result back.

This is NOT the production way to do it (Build 2 is). But doing it manually first
makes the mechanics obvious. The SDK is doing exactly this, just more robustly.

The format we'll use:
    The model emits tool calls wrapped in <tool_call> tags, like:

        I need to read the file first.

        <tool_call>
        {"name": "read_file", "arguments": {"path": "notes.txt"}}
        </tool_call>

    Your code finds the tag, parses the JSON, runs the function, and injects
    the result back as a <tool_response> in the next message.

Tasks:
  1. Complete `parse_tool_call` to extract name + arguments from a model response
  2. Complete `dispatch` to route a tool call to the right Python function
  3. Complete `run_agent` to implement the back-and-forth loop

Tools to implement:
  - read_file(path: str) -> dict    reads a file from disk and returns its content
  - write_file(path: str, content: str) -> dict    writes content to a file on disk

Before running, create a file called `sample.txt` with some text in it.
"""

import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

SYSTEM_PROMPT = """You are a helpful file assistant with access to the following tools:

- read_file(path: str): reads a file from disk and returns its content
- write_file(path: str, content: str): writes content to a file on disk

When you need to use a tool, emit EXACTLY this format and nothing else after it:

<tool_call>
{"name": "TOOL_NAME", "arguments": {"arg1": "value1"}}
</tool_call>

After you receive the tool result in a <tool_response> block, continue your response
normally. Do not emit a tool_call and prose in the same turn. Pick one or the other.
"""


def get_active_model() -> str:
    priority_models = [
        "deepseek/deepseek-v4-flash:free",
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        # "openai/gpt-oss-20b:free",
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

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def read_file(path: str) -> dict:
    """
    Read a file from disk and return its content.
    Return {"content": ..., "path": ...} on success.
    Return {"error": ...} if the file doesn't exist or can't be read.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"content": content, "path": path}
    except Exception as e:
        return {"error": str(e)}


def write_file(path: str, content: str) -> dict:
    """
    Write content to a file on disk.
    Return {"success": True, "path": ..., "bytes_written": ...} on success.
    Return {"error": ...} on failure.

    Hint: open(path, 'w') and then f.write(content).
    """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"success": True, "path": path, "bytes_written": len(content)}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_tool_call(response_text: str) -> dict | None:
    """
    Extract a tool call from the model's response text.

    Returns a dict {"name": str, "arguments": dict} if a <tool_call> block is found,
    or None if there is no tool call in the response.

    The format to parse:
        <tool_call>
        {"name": "...", "arguments": {...}}
        </tool_call>

    Hint: use re.search() with re.DOTALL to find the block, then json.loads() the body.
    """
    if not response_text:
        return None
    match = re.search(r"<tool_call>(.*?)</tool_call>",
                      response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
    return None


def strip_tool_call(response_text: str) -> str:
    """
    Return the response text with any <tool_call>...</tool_call> block removed.
    Useful for printing the model's prose without the raw tag.
    """
    return re.sub(r"<tool_call>.*?</tool_call>", "", response_text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "read_file": read_file,
    "write_file": write_file,
}


def dispatch(name: str, arguments: dict) -> str:
    """
    Look up the tool by name, call it with the given arguments, and return a
    JSON string of the result.

    If the tool is not found, return: {"error": "Unknown tool: <name>"}
    If the call raises an exception, return: {"error": "<exception message>"}

    Always return a string (json.dumps the result dict).
    """
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        func = TOOL_REGISTRY[name]
        result = func(**arguments)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 6


def run_agent(user_message: str) -> str:
    """
    Run the tool-calling agent loop for a single user message.

    Steps:
      1. Build the initial messages list with SYSTEM_PROMPT + user message.
      2. Call the model.
      3. Parse the response for a <tool_call>.
      4. If found: run the tool, inject a <tool_response> block into messages, go to 2.
      5. If not found: return the model's text (the final answer).
      6. If MAX_ITERATIONS reached: return an error string.

    The <tool_response> you inject back should look like:
        <tool_response>
        {"content": "Hello, world!", "path": "sample.txt"}
        </tool_response>

    Wrap it in a user message so the model sees it as a continuation:
        {"role": "user", "content": "<tool_response>\n...\n</tool_response>"}

    Print a line to stderr each time a tool is called so you can follow the loop.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        response_text = response.choices[0].message.content or ""
        tool_call = parse_tool_call(response_text)
        if tool_call:
            messages.append({"role": "assistant", "content": response_text})
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
            print(f"  [Agent invoking tool: {name}({arguments})]")
            tool_result_json = dispatch(name, arguments)
            tool_response_msg = f"<tool_response>\n{tool_result_json}\n</tool_response>"
            messages.append({"role": "user", "content": tool_response_msg})
        else:
            return response_text
    return f"[Agent stopped after {MAX_ITERATIONS} iterations]"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a sample file for the agent to work with
    with open("sample.txt", "w") as f:
        f.write(
            "IIT Delhi was established in 1961. It is one of the premier engineering institutions in India.\n")
        f.write("The campus spans 325 acres in Hauz Khas, New Delhi.\n")

    try:
        MODEL = get_active_model()
    except RuntimeError as e:
        print(e)
        exit(1)

    test_queries = [
        "Read sample.txt and summarise what it says.",
        "Read sample.txt and write a one-sentence version of its content to summary.txt.",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        result = run_agent(query)
        print(f"Answer: {result}")
