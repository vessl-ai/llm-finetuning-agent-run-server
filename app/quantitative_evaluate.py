#!/usr/bin/env python3
"""
Evaluate a single JSONL file in /root/data using either a local Hugging Face transformers model
or a remote vLLM OpenAI-compatible server. Supports:
- Data formats: conversation | instruction | generic_text
- Metrics: accuracy_precision_recall_f1 | bleu_rouge

Usage examples:
  python eval_runner.py --data_format conversation --metrics accuracy_precision_recall_f1 bleu_rouge \
      --model_name /root/output/finetuning-output-2/merged

  python eval_runner.py --data_format instruction --metrics bleu_rouge \
      --model_name http://localhost:8000 --remote_model Qwen/Qwen3-8B --api_key EMPTY
"""

import argparse
import glob
import json
import logging
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

import requests
from tqdm import tqdm

# Lazy imports for local Transformers path (so remote-only runs don't require these)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None

# Hugging Face evaluate (for BLEU/ROUGE)
import evaluate  # pip install evaluate


def save_inferenced_results(original_rows: List[Dict[str, Any]], 
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


def save_evaluation_results(original_rows: List[Dict[str, Any]], 
                           expected_outputs: List[str],
                           actual_outputs: List[str],
                           data_format: str,
                           results: Dict[str, Any],
                           original_path: str) -> str:
    """Save original data with evaluation results to a new JSONL file."""
    # Create output filename
    base_name = os.path.splitext(original_path)[0]
    output_path = f"{base_name}_quantitative_evaluated.jsonl"
    
    logging.info(f"Saving quantitative evaluation results to: {output_path}")
    
    # Add evaluation results to original rows
    evaluated_rows = []
    
    for i, (row, expected, actual) in enumerate(zip(original_rows, expected_outputs, actual_outputs)):
        new_row = row.copy()
        
        # Add evaluation data
        new_row["evaluation"] = {
            "data_format": data_format,
            "expected_output": expected,
            "actual_output": actual,
            "row_index": i
        }
        
        # Add quantitative metrics for this row
        if "accuracy_precision_recall_f1" in results:
            # Calculate individual row metrics
            row_metrics = compute_individual_accuracy_prf1(actual, expected)
            new_row["evaluation"]["accuracy_precision_recall_f1"] = row_metrics
        
        if "bleu_rouge" in results:
            # Calculate individual row metrics
            row_metrics = compute_individual_bleu_rouge(actual, expected)
            new_row["evaluation"]["bleu_rouge"] = row_metrics
        
        evaluated_rows.append(new_row)
    
    # Save to new file
    with open(output_path, "w", encoding="utf-8") as f:
        for row in evaluated_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    logging.info(f"Saved {len(evaluated_rows)} rows with evaluation results")
    return output_path


def compute_individual_accuracy_prf1(actual: str, expected: str) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, F1 for a single output pair."""
    try:
        # Normalize texts
        actual_norm = normalize_text(actual)
        expected_norm = normalize_text(expected)
        
        # Exact match accuracy
        accuracy = 1.0 if actual_norm == expected_norm else 0.0
        
        # Word-level precision/recall/F1
        actual_tokens = set(actual.split())
        expected_tokens = set(expected.split())
        
        if not expected_tokens:
            precision = recall = f1 = 0.0
        else:
            common_tokens = actual_tokens & expected_tokens
            precision = len(common_tokens) / len(actual_tokens) if actual_tokens else 0.0
            recall = len(common_tokens) / len(expected_tokens)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "actual_tokens": len(actual_tokens),
            "expected_tokens": len(expected_tokens),
            "common_tokens": len(common_tokens)
        }
    except Exception as e:
        return {
            "error": str(e),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }


def compute_individual_bleu_rouge(actual: str, expected: str) -> Dict[str, Any]:
    """Compute BLEU and ROUGE for a single output pair."""
    try:
        # Simple BLEU calculation (word-based)
        actual_words = actual.split()
        expected_words = expected.split()
        
        if not expected_words:
            return {"bleu": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        # BLEU-1 (unigram precision)
        if not actual_words:
            bleu = 0.0
        else:
            common_words = set(actual_words) & set(expected_words)
            bleu = len(common_words) / len(actual_words) if actual_words else 0.0
        
        # ROUGE-1 (unigram recall)
        rouge1 = len(common_words) / len(expected_words) if expected_words else 0.0
        
        # ROUGE-2 (bigram recall) - simplified
        if len(expected_words) >= 2:
            expected_bigrams = set(zip(expected_words[:-1], expected_words[1:]))
            actual_bigrams = set(zip(actual_words[:-1], actual_words[1:])) if len(actual_words) >= 2 else set()
            common_bigrams = expected_bigrams & actual_bigrams
            rouge2 = len(common_bigrams) / len(expected_bigrams) if expected_bigrams else 0.0
        else:
            rouge2 = 0.0
        
        # ROUGE-L (longest common subsequence) - simplified as word overlap
        rougeL = rouge1
        
        return {
            "bleu": bleu,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "actual_words": len(actual_words),
            "expected_words": len(expected_words),
            "common_words": len(common_words)
        }
    except Exception as e:
        return {
            "error": str(e),
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0
        }


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
# Metrics
# -----------------------

def compute_accuracy_prf1(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1 with error handling."""
    try:
        assert len(preds) == len(refs)

        # Exact match accuracy on normalized strings
        acc = sum(1 for p, r in zip(preds, refs) if normalize_text(p) == normalize_text(r)) / max(len(preds), 1)

        # Micro-averaged word-level precision/recall/F1 using token counts
        tp = fp = fn = 0
        for p, r in zip(preds, refs):
            p_tokens = Counter(p.split())
            r_tokens = Counter(r.split())
            # overlap per token
            common = p_tokens & r_tokens
            tp += sum(common.values())
            fp += sum((p_tokens - r_tokens).values())
            fn += sum((r_tokens - p_tokens).values())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_samples": len(preds),
        }
    except Exception as e:
        logging.error(f"Error computing accuracy/precision/recall/F1 metrics: {e}")
        # Return default values when metrics fail
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "n_samples": len(preds) if 'preds' in locals() else 0,
            "error": f"Computation failed: {str(e)}"
        }


def compute_bleu_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Compute BLEU and ROUGE metrics with proper error handling."""
    try:
        # BLEU (word-based): use evaluate BLEU with a tokenizer function = str.split
        logging.info("Loading BLEU metric...")
        bleu_metric = evaluate.load("bleu")
        bleu = bleu_metric.compute(
            predictions=preds,
            references=[[r] for r in refs],   # evaluate BLEU expects list of list(s)
            tokenizer=str.split
        )

        # ROUGE (word-based): pass tokenizer=str.split and aggregation
        logging.info("Loading ROUGE metric...")
        rouge_metric = evaluate.load("rouge")
        rouge = rouge_metric.compute(
            predictions=preds,
            references=refs,
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            use_aggregator=True,
            use_stemmer=False,
            tokenizer=str.split
        )

        out = {
            "bleu": bleu.get("bleu", 0.0),
            "bleu_precisions": bleu.get("precisions"),
            "bleu_brevity_penalty": bleu.get("brevity_penalty"),
            "bleu_length_ratio": bleu.get("length_ratio"),
            "bleu_translation_length": bleu.get("translation_length"),
            "bleu_reference_length": bleu.get("reference_length"),
        }
        out.update(rouge)
        return out
        
    except ImportError as e:
        logging.error(f"Missing dependencies for BLEU/ROUGE metrics: {e}")
        logging.error("Please install required packages: pip install rouge_score nltk")
        # Return default values when metrics cannot be computed
        return {
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "rougeLsum": 0.0,
            "error": "Dependencies missing - install rouge_score and nltk"
        }
    except Exception as e:
        logging.error(f"Error computing BLEU/ROUGE metrics: {e}")
        # Return default values when metrics fail
        return {
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "rougeLsum": 0.0,
            "error": f"Computation failed: {str(e)}"
        }


# -----------------------
# Driver
# -----------------------

def find_inferenced_jsonl(root: str = "/root/data") -> str:
    """Find the _inferenced.jsonl file in the data directory."""
    files = glob.glob(os.path.join(root, "*_inferenced.jsonl"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one *_inferenced.jsonl in {root}, found {len(files)}")
    return files[0]


def extract_expected_actual(row: Dict[str, Any], data_format: str) -> Tuple[str, str]:
    """Extract expected_output and actual_output from a row based on data_format."""
    if data_format == "conversation":
        # Last message from assistant is expected output
        messages = row.get("messages", [])
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Conversation row must have last message from assistant")
        expected_output = messages[-1].get("content", "")
    elif data_format == "instruction":
        # Completion field is expected output
        expected_output = row.get("completion", "")
    elif data_format == "generic_text":
        # Text field is expected output
        expected_output = row.get("text", "")
    else:
        raise ValueError(f"Unknown data_format: {data_format}")
    
    # Actual output is the inferenced field
    actual_output = row.get("inferenced", "")
    
    if not expected_output:
        raise ValueError(f"Expected output is empty for data_format: {data_format}")
    if not actual_output:
        raise ValueError("Actual output (inferenced) is empty")
    
    return expected_output, actual_output


def run_evaluation(
    data_format: str,
    metrics: List[str],
    model_name: str = None,  # No longer needed for evaluation
    remote_model: str = None,  # No longer needed for evaluation
    api_key: str = "",  # No longer needed for evaluation
    max_new_tokens: int = 512,  # No longer needed for evaluation
    temperature: float = 0.0,  # No longer needed for evaluation
    num_threads: int = 8,  # No longer needed for evaluation
) -> Dict[str, Any]:
    logging.info(f"Starting evaluation: data_format={data_format}, metrics={metrics}")
    
    # Load the inferenced JSONL file instead of running inference
    jsonl_path = find_inferenced_jsonl("/root/data")
    rows = load_jsonl(jsonl_path)
    
    logging.info(f"Loaded {len(rows)} rows from inferenced file: {jsonl_path}")

    # Extract expected and actual outputs
    logging.info("Extracting expected and actual outputs...")
    expected_outputs = []
    actual_outputs = []
    
    for i, row in tqdm(enumerate(rows), total=len(rows), desc="Extracting outputs"):
        try:
            expected, actual = extract_expected_actual(row, data_format)
            expected_outputs.append(expected)
            actual_outputs.append(actual)
        except Exception as e:
            logging.error(f"Error extracting outputs from row {i}: {e}")
            raise RuntimeError(f"Error extracting outputs from row {i}: {e}")
    
    logging.info(f"Successfully extracted {len(expected_outputs)} output pairs")

    # Initialize results dictionary
    results: Dict[str, Any] = {
        "meta": {
            "backend": "evaluation_only",  # No inference backend used
            "data_format": data_format,
            "num_samples": len(expected_outputs),
            "jsonl_path": jsonl_path,
        }
    }

    # Compute metrics
    logging.info("Calculating metrics...")
    
    if "accuracy_precision_recall_f1" in metrics:
        logging.info("Calculating accuracy_precision_recall_f1 metric...")
        results["accuracy_precision_recall_f1"] = compute_accuracy_prf1(actual_outputs, expected_outputs)

    if "bleu_rouge" in metrics:
        logging.info("Calculating bleu_rouge metric...")
        results["bleu_rouge"] = compute_bleu_rouge(actual_outputs, expected_outputs)

    # Save individual row evaluation results
    output_path = save_evaluation_results(rows, expected_outputs, actual_outputs, data_format, results, jsonl_path)
    results["meta"]["output_path"] = output_path

    logging.info("Evaluation complete")
    return results


def main():
    parser = argparse.ArgumentParser(description="Quantitative evaluation for inferenced JSONL data.")
    parser.add_argument("--data_format", required=True, choices=["conversation", "instruction", "generic_text"],
                        help="Input row format.")
    parser.add_argument("--metrics", required=True, nargs="+",
                        choices=["accuracy_precision_recall_f1", "bleu_rouge"],
                        help="One or more metrics to compute.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set logging level")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    results = run_evaluation(
        data_format=args.data_format,
        metrics=args.metrics,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))
    
    # Print summary information
    if "meta" in results:
        print(f"\n📁 Evaluation completed for: {results['meta']['jsonl_path']}")
        print(f"📊 Total samples processed: {results['meta']['num_samples']}")
        print(f"🔧 Evaluation mode: {results['meta']['backend']}")


if __name__ == "__main__":
    main()
