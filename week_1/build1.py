import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def call_model(prompt: str) -> str:
    priority_models = [
        "deepseek/deepseek-v4-flash:free",
        "openai/gpt-oss-20b:free",
        "openrouter/free"
    ]
    active_model = None
    print("Checking for an available model endpoint...")
    for name in priority_models:
        try:
            client.chat.completions.create(
                model=name,
                messages=[{"role": "user", "content": "ping"}],
            )
            active_model = name
            print(f"Connected successfully to: {active_model}\n")
            break
        except Exception:
            pass
    if not active_model:
        return "Critical Error: All priority models failed to respond."
    try:
        print("--- MODEL RESPONSE ---")
        print("[MODEL]: ", end="", flush=True)
        response_stream = client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": "You have to answer any question asked to you concisely and to the point."},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            stream_options={"include_usage": True}
        )
        full_reply = ""
        last_usage = None
        for chunk in response_stream:
            if chunk.choices:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    print(delta_content, end="", flush=True)
                    full_reply += delta_content

            if hasattr(chunk, 'usage') and chunk.usage is not None:
                last_usage = chunk.usage
        print("\n" + "-" * 50)
        print(f"Model Used        : {active_model}")
        if last_usage:
            print(f"Prompt Tokens     : {last_usage.prompt_tokens}")
            print(f"Completion Tokens : {last_usage.completion_tokens}")
            print(f"Total Tokens      : {last_usage.total_tokens}")
        else:
            print("Token Usage       : Metadata unavailable")
        print("-" * 50)
        return full_reply
    except Exception as e:
        return f"Error executing single-turn call: {e}"


if __name__ == "__main__":
    user_prompt = input("ASK ME ANYTHING: ").strip()
    if user_prompt:
        call_model(user_prompt)
