"""
Build 2: Tool Calling with the OpenAI SDK
==========================================
Build 1 had you implement the tool-call round-trip by hand using a custom text format.
This build does the same thing the production way: using the OpenAI SDK's native
`tools` parameter, `tool_calls` response field, and `"role": "tool"` messages.

The mechanics are identical. You're still parsing a tool name, running a function,
and sending the result back. The difference is that the SDK handles the encoding
and the model is trained to produce structured JSON tool calls rather than freeform XML.

Implement the same two tools as Build 1:
  - get_weather(city: str) -> dict
  - calculate(expression: str) -> dict

Then complete the agent loop and watch the pattern become clean.

Stretch goals (not required):
  - Add a third tool: get_time(timezone: str) -> dict
  - Handle multiple tool_calls in a single response (the model can call several at once)
  - Add a token counter that prints total tokens used after the loop ends
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import sys

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

# ---------------------------------------------------------------------------
# Tool schemas (the contract between you and the model)
# ---------------------------------------------------------------------------


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Returns the current weather for a given city. "
                "Call this whenever the user asks about weather, temperature, or climate. "
                "Do not guess weather. Always call this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Delhi' or 'San Francisco'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Default to celsius.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluates a mathematical expression and returns the result. "
                "Use this for any arithmetic the user asks about. "
                "Pass the expression as a string, e.g. '1337 * 42 + 7'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python arithmetic expression, e.g. '100 / 4 + 3'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def get_weather(city: str, unit: str = "celsius") -> dict:
    """
    Return realistic-looking fake weather data for the city.
    In production this would call a weather API.

    Return a dict like:
        {"city": city, "temperature": 28, "unit": unit, "condition": "partly cloudy"}
    """
    city_lower = city.lower()
    if "tokyo" in city_lower:
        temp = 14 if unit == "celsius" else 57
        condition = "Rainy"
    elif "delhi" in city_lower:
        temp = 34 if unit == "celsius" else 93
        condition = "Sunny"
    elif "london" in city_lower:
        temp = 11 if unit == "celsius" else 52
        condition = "Overcast"
    else:
        temp = 21 if unit == "celsius" else 70
        condition = "Partly Cloudy"
    return {
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": condition
    }


def calculate(expression: str) -> dict:
    """
    Safely evaluate a math expression.
    Use eval() with restricted globals so imports and builtins are blocked.
    Return {"result": value} or {"error": message}.
    """
    try:
        allowed_globals = {"__builtins__": None}
        allowed_locals = {}
        result = eval(expression, allowed_globals, allowed_locals)
        return {"result": result}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
}


def dispatch(tool_call) -> str:
    """
    Execute a single tool_call object from the API response.

    tool_call has:
        tool_call.function.name       (the tool name)
        tool_call.function.arguments  (a JSON string of arguments)

    Return a JSON string of the result dict.
    On unknown tool or exception, return a JSON error dict.

    Note: tool_call.function.arguments is a *string*, not a dict. Parse it first.
    """
    name = tool_call.function.name
    arguments_str = tool_call.function.arguments
    try:
        args = json.loads(arguments_str)
    except Exception as e:
        return json.dumps({"error": f"Failed to parse arguments JSON: {str(e)}"})
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        func = TOOL_REGISTRY[name]
        result_dict = func(**args)
        return json.dumps(result_dict)
    except Exception as e:
        return json.dumps({"error": f"Execution error in tool '{name}': {str(e)}"})


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 8


def run_agent(user_message: str) -> str:
    """
    Run the agent loop using native SDK tool calling.

    Steps:
      1. Append the user message to history.
      2. Call client.chat.completions.create() with tools=TOOLS.
      3. If response.choices[0].finish_reason == "tool_calls":
           a. Append the assistant message (it contains .tool_calls) to history.
           b. For each tool_call in message.tool_calls:
                - dispatch it
                - append a {"role": "tool", "tool_call_id": ..., "content": ...} message
           c. Go to 2.
      4. If finish_reason == "stop": return message.content.
      5. If MAX_ITERATIONS reached: return an error string.

    Print to stderr whenever a tool executes so you can follow the loop.

    Hint: the assistant message you append in step 3a must be the raw message object,
    not a dict. The SDK accepts both, but keep it consistent with what the API returned.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when appropriate."},
        {"role": "user", "content": user_message},
    ]

    for _ in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        if getattr(response, 'choices', None) is None:
            print(
                f"\n[API Error] Invalid response from OpenRouter: {response}", file=sys.stderr)
            return "Error: The model provider returned an invalid response (likely a rate limit or timeout)."

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "tool_calls":
            messages.append(message.model_dump(exclude_none=True))
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    print(
                        f"[Tool Call] Executing '{tool_call.function.name}' with args: {tool_call.function.arguments}",
                        file=sys.stderr
                    )
                    result_str = dispatch(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": result_str,
                    })
            continue
        elif finish_reason == "stop":
            return message.content
    return f"[Agent stopped after {MAX_ITERATIONS} iterations without a final answer]"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate: (2**10) - 1",
        "Compare the weather in London and Delhi, and tell me what 451 * 3 is.",
    ]

    MODEL = get_active_model()

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        result = run_agent(query)
        print(f"\nFinal answer:\n{result}")
