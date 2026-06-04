import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def run_chatbot():
    priority_models = [
        "deepseek/deepseek-v4-flash:free",
        "openai/gpt-oss-20b:free",
        "openrouter/free"
    ]
    active_model = None
    print("Checking for an available model endpoint...")
    for model_name in priority_models:
        try:
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "ping"}],
            )
            active_model = model_name
            print(f"Connected successfully to: {active_model}\n")
            break
        except Exception:
            pass
    if not active_model:
        print("Critical Error: All priority models failed to respond.")
        return
    system_prompt = {"role": "system",
                     "content": "You are a helpful assistant."}
    messages = [system_prompt]
    MAX_TURNS = 10
    last_usage = None
    print("Chat started. Type 'exit' or 'quit' to quit.")
    print("Commands: /reset (clear history), /compact (force condense history), /tokens (show usage)\n")
    while True:
        if len(messages) > MAX_TURNS:
            print(
                "\n[SYSTEM]: History threshold reached. Compacting old turns to save context space...")
            try:
                turns_to_compress = messages[1:-2]
                saved_turns = messages[-2:]
                compaction_prompt = (
                    "Summarize the core facts, context, and user details from this chat history "
                    "into a single, concise paragraph. Focus only on persistent context:\n\n"
                    f"{turns_to_compress}"
                )
                summary_response = client.chat.completions.create(
                    model=active_model,
                    messages=[
                        {"role": "system", "content": compaction_prompt}],
                )
                summary_text = summary_response.choices[0].message.content
                messages = [
                    system_prompt,
                    {"role": "system",
                        "content": f"Summary of previous conversation: {summary_text}"}
                ] + saved_turns
                print("[SYSTEM]: Context successfully compacted.\n")
            except Exception as e:
                print(
                    f"[SYSTEM WARNING]: Automatic compaction failed: {e}. Continuing anyway...\n")
        user_input = input("[YOU]: ").strip()
        if not user_input:
            continue
        lower_input = user_input.lower()
        if lower_input in ["exit", "quit"]:
            print("Goodbye!")
            break
        elif lower_input == "/reset":
            messages = [system_prompt]
            print("[SYSTEM]: Conversation history cleared.\n")
            continue
        elif lower_input == "/tokens":
            if last_usage:
                print(
                    f"[SYSTEM]: Last call used {last_usage.total_tokens} tokens (Prompt: {last_usage.prompt_tokens}, Comp: {last_usage.completion_tokens})\n")
            else:
                print("[SYSTEM]: No data streams processed yet.\n")
            continue
        elif lower_input == "/compact":
            if len(messages) <= 3:
                print("[SYSTEM]: Not enough history to compact yet.\n")
            else:
                print("[SYSTEM]: Manually compressing context...")
                turns_to_compress = messages[1:]
                try:
                    summary_response = client.chat.completions.create(
                        model=active_model,
                        messages=[
                            {"role": "system", "content": f"Summarize this context into one concise paragraph:\n{turns_to_compress}"}]
                    )
                    summary_text = summary_response.choices[0].message.content
                    messages = [system_prompt, {
                        "role": "system", "content": f"Summary of previous conversation: {summary_text}"}]
                    print("[SYSTEM]: Context successfully compressed.\n")
                except Exception as e:
                    print(f"[SYSTEM ERROR]: Compaction failed: {e}\n")
            continue
        messages.append({"role": "user", "content": user_input})
        try:
            print("[CHATBOT]: ", end="", flush=True)
            response_stream = client.chat.completions.create(
                model=active_model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True}
            )
            full_reply = ""
            for chunk in response_stream:
                if chunk.choices:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        print(delta_content, end="", flush=True)
                        full_reply += delta_content
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    last_usage = chunk.usage
            print("\n")
            messages.append({"role": "assistant", "content": full_reply})
        except Exception as e:
            print(f"\n[ERROR]: Failed to get stream response: {e}")
            print("Please retry your message.\n")
            messages.pop()


if __name__ == "__main__":
    run_chatbot()
