#!/usr/bin/env python3
"""
Inference script to generate predictions for JSONL data and save as _inferenced.jsonl.
Supports conversation, instruction, and generic_text formats.
Works with both local Hugging Face models and vLLM endpoints.
"""

import argparse
import glob
import json
import logging
import os
import re
from typing import Dict, List, Tuple, Any

import requests

# Lazy imports for local Transformers path
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None

from tqdm import tqdm


# -----------------------
# Utilities
# -----------------------

def setup_logging(level: str = "INFO") -> None:
    """Initialize logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def is_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))


def find_single_jsonl(root: str = "/root/data") -> str:
    files = glob.glob(os.path.join(root, "*.jsonl"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one .jsonl in {root}, found {len(files)}")
    return files[0]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    logging.info(f"Loading JSONL file: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    logging.info(f"Loaded {len(rows)} data rows.")
    return rows


def normalize_text(s: str) -> str:
    # Simple normalization for exact-match accuracy
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


# -----------------------
# Prompt builders
# -----------------------

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def prepare_conversation(row: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    """
    Expect: {"messages": [{"role": "...", "content": "..."}, ...]}
    The last message must be assistant (reference). We feed all previous messages to the model.
    """
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Invalid 'messages' format.")
    if messages[-1].get("role") != "assistant":
        raise ValueError("For 'conversation' format, the last message must be from 'assistant'.")

    reference = messages[-1].get("content", "")
    input_messages = messages[:-1]
    # Ensure there is at least a system message; if not, prepend one.
    if not input_messages or input_messages[0].get("role") != "system":
        input_messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}] + input_messages
    return input_messages, reference


def prepare_instruction(row: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    """
    Expect: {"prompt": "...", "completion": "..."}
    """
    prompt = row.get("prompt")
    completion = row.get("completion")
    if prompt is None or completion is None:
        raise ValueError("Instruction row must contain 'prompt' and 'completion'.")
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return messages, completion


def prepare_generic_text(row: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    """
    Expect: {"text": "..."}
    We instruct the model to repeat the text EXACTLY so comparisons are meaningful.
    """
    text = row.get("text")
    if text is None:
        raise ValueError("Generic_text row must contain 'text'.")
    user_msg = f"Repeat exactly the following text between triple backticks:\n```{text}```"
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return messages, text


def prepare_row(row: Dict[str, Any], data_format: str) -> Tuple[List[Dict[str, str]], str]:
    if data_format == "conversation":
        return prepare_conversation(row)
    elif data_format == "instruction":
        return prepare_instruction(row)
    elif data_format == "generic_text":
        return prepare_generic_text(row)
    else:
        raise ValueError(f"Unknown data_format: {data_format}")


# -----------------------
# Inference backends
# -----------------------

class LocalHFGenerator:
    def __init__(self, model_name: str, device: str = "cuda"):
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            raise RuntimeError("Transformers/torch not available, cannot run local inference.")
        
        logging.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding=True)
        
        logging.info(f"Loading model: {model_name} -> {device}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

        # Default safe pad token id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logging.info("Local model loading complete")

    def generate_from_messages(self, messages: List[Dict[str, str]],
                               max_new_tokens: int = 512,
                               temperature: float = 0.0) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        # Slice off the prompt tokens to get only the generated continuation
        gen_ids = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()


class VLLMClient:
    """
    Minimal OpenAI-compatible client for vLLM /v1/chat/completions.
    Pass base_url like http://host:8000 or http://host:8000/v1 (both supported).
    """
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY"):
        base = base_url.rstrip("/")
        self.base_url = base if base.endswith("/v1") else base + "/v1"
        self.model = model
        self.api_key = api_key

        logging.info(f"Initializing vLLM client: {self.base_url}, model: {model}")

        self.session = requests.Session()
        self.headers = {"Content-Type": "application/json"}
        # Many vLLM deployments accept "EMPTY" or no auth; add header only if provided.
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        logging.info("vLLM client initialization complete")

    def generate_from_messages(self, messages: List[Dict[str, str]],
                               max_new_tokens: int = 512,
                               temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": False,
        }
        url = f"{self.base_url}/chat/completions"
        resp = self.session.post(url, headers=self.headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Standard OpenAI-like response
        return data["choices"][0]["message"]["content"].strip()


# -----------------------
# Main inference function
# -----------------------

def run_inference(
    data_format: str,
    model_name: str,
    remote_model: str = None,
    api_key: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    jsonl_path = find_single_jsonl("/root/data")
    rows = load_jsonl(jsonl_path)

    # Build backend
    if is_url(model_name):
        if not remote_model:
            raise ValueError("When --model_name is a URL, you must also pass --remote_model (the served model id).")
        logging.info(f"Using vLLM backend: {model_name}, model: {remote_model}")
        generator = VLLMClient(base_url=model_name, model=remote_model, api_key=api_key)
        backend = "vllm_openai_compatible"
    else:
        logging.info(f"Using local Transformers backend: {model_name}")
        generator = LocalHFGenerator(model_name=model_name, device="cuda")
        backend = "transformers_local"

    # Prepare messages and references
    logging.info("Preparing data...")
    prepared: List[Tuple[List[Dict[str, str]], str]] = []
    for i, row in tqdm(enumerate(rows), total=len(rows), desc="Preparing data"):
        try:
            prepared.append(prepare_row(row, data_format))
        except Exception as e:
            logging.error(f"Error preparing row {i}: {e}")
            raise RuntimeError(f"Error preparing row {i}: {e}")
    
    logging.info(f"Data preparation complete: {len(prepared)} rows")

    # Inference
    logging.info("Starting model inference...")
    predictions: List[str] = [""] * len(prepared)

    if backend == "transformers_local":
        # Local: iterate sequentially to avoid GPU OOM
        logging.info("Starting local sequential inference")
        for idx, (messages, _) in tqdm(enumerate(prepared), total=len(prepared), desc="Local inference"):
            try:
                predictions[idx] = generator.generate_from_messages(messages, max_new_tokens, temperature)
            except Exception as e:
                logging.error(f"Error during inference for row {idx}: {e}")
                predictions[idx] = "[ERROR]"
    else:
        # vLLM: single request for now (can be extended to multi-threaded if needed)
        logging.info("Starting vLLM inference")
        for idx, (messages, _) in tqdm(enumerate(prepared), total=len(prepared), desc="vLLM inference"):
            try:
                predictions[idx] = generator.generate_from_messages(messages, max_new_tokens, temperature)
            except Exception as e:
                logging.error(f"Error during inference for row {idx}: {e}")
                predictions[idx] = "[ERROR]"

    logging.info("Inference complete")

    # Save inference results to new JSONL file
    output_path = save_inference_results(rows, predictions, jsonl_path)
    
    return output_path


def save_inference_results(original_rows: List[Dict[str, Any]], 
                           predictions: List[str], 
                           original_path: str) -> str:
    """Save original data with inference results to a new JSONL file."""
    # Create output filename
    base_name = os.path.splitext(original_path)[0]
    output_path = f"{base_name}_inferenced.jsonl"
    
    logging.info(f"Saving inference results to: {output_path}")
    
    # Add inference results to original rows
    inferenced_rows = []
    error_count = 0
    success_count = 0
    
    for i, (row, prediction) in enumerate(zip(original_rows, predictions)):
        new_row = row.copy()
        new_row["inferenced"] = prediction
        
        # Track statistics
        if prediction == "[ERROR]":
            error_count += 1
        else:
            success_count += 1
            
        inferenced_rows.append(new_row)
    
    # Save to new file
    with open(output_path, "w", encoding="utf-8") as f:
        for row in inferenced_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    logging.info(f"Saved {len(inferenced_rows)} rows with inference results")
    logging.info(f"Success: {success_count}, Errors: {error_count}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run inference on JSONL data and save results.")
    parser.add_argument("--data_format", required=True, choices=["conversation", "instruction", "generic_text"],
                        help="Input row format.")
    parser.add_argument("--model_name", required=True,
                        help="Local HF path/repo OR base URL for vLLM (e.g., http://host:8000 or http://host:8000/v1).")
    parser.add_argument("--remote_model", default=None,
                        help="Model name served by vLLM (required if model_name is a URL).")
    parser.add_argument("--api_key", default="",
                        help="Optional API key for vLLM/OpenAI-compatible server (many use 'EMPTY').")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set logging level")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    output_path = run_inference(
        data_format=args.data_format,
        model_name=args.model_name,
        remote_model=args.remote_model,
        api_key=args.api_key,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(f"\n✅ Inference completed successfully!")
    print(f"📁 Results saved to: {output_path}")
    print(f"🔧 Backend used: {'vLLM' if is_url(args.model_name) else 'Local Transformers'}")


if __name__ == "__main__":
    main()
