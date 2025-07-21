import time
import requests, json, gradio as gr

BASE_URL = "http://0.0.0.0:8000/v1"
VLLM_URL = f"{BASE_URL}/chat/completions"
HEADERS = {"Content-Type": "application/json"}
RETRY_INTERVAL = 5          # seconds between retries
RETRY_TIMEOUT = 300         # total retry duration in seconds

def get_vllm_model_id_with_retry():
    """
    Try retrieving a model ID by polling /models endpoint,
    retrying every RETRY_INTERVAL seconds for up to RETRY_TIMEOUT.
    """
    start_time = time.time()
    while time.time() - start_time < RETRY_TIMEOUT:
        try:
            resp = requests.get(f"{BASE_URL}/models")
            resp.raise_for_status()
            data = resp.json()
            if data.get("data"):
                model_id = data["data"][0]["id"]
                print(f"Model ID found: {model_id}")
                return model_id
            print("No model available yet. Retrying...")
        except Exception as e:
            print(f"Request failed: {e}. Retrying...")
        time.sleep(RETRY_INTERVAL)
    raise TimeoutError("Timed out waiting for model.")

# Fetch and fix the model ID at startup
MODEL = get_vllm_model_id_with_retry()

def vllm_chat(message, history, temperature, top_p):
    """
    Send user+history to vLLM and stream tokens back to Gradio.
    History/items use {'role': ..., 'content': ...} format (type='messages').
    """
    # ① include the new user turn right away
    chat_history = history + [{"role": "user", "content": message}]

    payload = {
        "model": MODEL,
        "messages": chat_history,
        "stream": True,
        "temperature": temperature,
        "top_p": top_p,
    }
    resp = requests.post(VLLM_URL, headers=HEADERS, json=payload, stream=True)

    assistant_text = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        chunk = line[6:].strip()
        if chunk == "[DONE]":
            break
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue

        delta = data["choices"][0]["delta"].get("content", "")
        assistant_text += delta

        # ② yield BOTH the user turn and the growing assistant reply
        yield chat_history + [{"role": "assistant", "content": assistant_text}]

# Define the Gradio app at module level as `demo`
with gr.Blocks(theme=gr.themes.Glass(), css="""
    .gradio-container { background: #f5f7fa; }
    .chatbot .message-row { margin-bottom: 0.6rem; }
""") as demo:
    gr.Markdown("# 🚀 vLLM Chatbot")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(elem_id="chatbot", height=600,
                                  bubble_full_width=False,
                                  show_copy_button=True,
                                  show_label=False,
                                 type="messages")
            msg = gr.Textbox(placeholder="Type your message…", show_label=False, lines=1)
            send = gr.Button("Send", variant="primary")
        with gr.Column(scale=1, min_width=180):
            with gr.Sidebar():
                with gr.Accordion("⚙️ Settings", open=False):
                    temp = gr.Slider(0, 1, 0.7, label="Temperature")
                    top_p = gr.Slider(0, 1, 1.0, label="Top‑p")
                clear = gr.Button("🗑️ Clear chat", variant="secondary")

    send.click(vllm_chat, [msg, chatbot, temp, top_p], chatbot)
    msg.submit(vllm_chat, [msg, chatbot, temp, top_p], chatbot)
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
