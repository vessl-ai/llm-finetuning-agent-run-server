import gradio as gr
from openai import OpenAI
import requests
import time


BASE_URL = "http://0.0.0.0:8000/v1"
RETRY_INTERVAL = 5          # 5 seconds
RETRY_TIMEOUT = 300         # 5 minutes = 300 seconds

def get_vllm_model_id_with_retry():
    """Retry getting model ID every 5 seconds for up to 5 minutes."""
    start_time = time.time()

    while time.time() - start_time < RETRY_TIMEOUT:
        try:
            response = requests.get(f"{BASE_URL}/models")
            response.raise_for_status()

            data = response.json()
            if data.get("data"):
                model_id = data["data"][0]["id"]
                print(f"Model ID found: {model_id}")
                return model_id
            else:
                print("No model available yet. Retrying...")

        except Exception as e:
            print(f"Request failed: {e}. Retrying...")

        time.sleep(RETRY_INTERVAL)

    raise TimeoutError("Timed out waiting for model to become available.")
    
client = OpenAI(
    base_url=BASE_URL  # Point to local vLLM instance
)
model_id = get_vllm_model_id_with_retry()

def respond(message, history):
    try:
        # Send user message to vLLM for inference
        completion = client.completions.create(
            model=model_id,
            prompt=message,
        )
        return completion.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"
# Define the chatbot interface
demo = gr.ChatInterface(
    fn=respond,
    title="AI Chatbot",
    description="Chat with a locally hosted LLM using vLLM and Gradio.",
    examples=["What’s the weather like?", "Tell me a joke.", "Write a short story."]
)
if __name__ == "__main__":
    demo.launch(
        share=False,  # Disable public access
        server_name="0.0.0.0",
        server_port=7860  # Default Gradio port
    )