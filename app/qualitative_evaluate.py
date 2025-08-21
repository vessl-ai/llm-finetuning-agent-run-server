#!/usr/bin/env python3
"""
Qualitative evaluation script for inferenced JSONL data using deepeval (G-Eval).
Builds LLMTestCase objects and evaluates with GEval metrics.
"""

import argparse
import glob
import json
import logging
import os
from statistics import mean
from typing import Dict, List, Any, Tuple

from tqdm import tqdm

# deepeval imports
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import evaluate  # optional runner; results are computed manually via measure()

# -----------------------
# Utilities
# -----------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def find_inferenced_jsonl(root: str = "/root/data") -> str:
    files = glob.glob(os.path.join(root, "*_inferenced.jsonl"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one *_inferenced.jsonl in {root}, found {len(files)}")
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

def extract_fields(row: Dict[str, Any], data_format: str) -> Tuple[str, str, str]:
    """
    Return (input_text, expected_output, actual_output) from a row based on data_format.
    - conversation: input = compressed version of user/assistant conversation (before last assistant), expected = last assistant utterance
    - instruction:  input = row['instruction'] or row['input'] or row['prompt'] (fallback 'N/A')
    - generic_text: input = row['input'] or row['prompt'] or 'N/A'
    """
    if data_format == "conversation":
        messages = row.get("messages", [])
        if not messages:
            raise ValueError("Conversation row must have 'messages'")
        if messages[-1].get("role") != "assistant":
            raise ValueError("Conversation row must have last message from assistant")
        expected_output = messages[-1].get("content", "")
        convo_prefix = []
        for m in messages[:-1]:
            role = m.get("role", "user")
            content = m.get("content", "")
            # Simple compression: combine as role: content format
            convo_prefix.append(f"{role}: {content}")
        input_text = "\n".join(convo_prefix) or "N/A"
    elif data_format == "instruction":
        expected_output = row.get("completion", "")
        input_text = row.get("instruction") or row.get("input") or row.get("prompt") or "N/A"
    elif data_format == "generic_text":
        expected_output = row.get("text", "")
        input_text = row.get("input") or row.get("prompt") or "N/A"
    else:
        raise ValueError(f"Unknown data_format: {data_format}")

    actual_output = row.get("inferenced", "")
    if not expected_output:
        raise ValueError(f"Expected output is empty for data_format: {data_format}")
    if not actual_output:
        raise ValueError("Actual output (inferenced) is empty")
    return input_text, expected_output, actual_output

# -----------------------
# Evaluation Logic
# -----------------------

def create_evaluator_from_metadata(meta: Dict[str, Any]) -> GEval:
    rubric = []
    for r in meta.get("rubric", []):
        score_range = r["score_range"]
        if isinstance(score_range, list):
            score_range = tuple(score_range)
        rubric.append(Rubric(score_range=score_range, expected_outcome=r["expected_outcome"]))
    return GEval(
        name=meta["name"],
        criteria=meta.get("criteria"),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        rubric=rubric or None,
    )

def build_llm_test_cases(rows: List[Dict[str, Any]], data_format: str) -> List[LLMTestCase]:
    test_cases: List[LLMTestCase] = []
    for i, row in tqdm(enumerate(rows), total=len(rows), desc="Building LLMTestCase"):
        input_text, expected, actual = extract_fields(row, data_format)
        tc = LLMTestCase(
            input=input_text,              # REQUIRED by LLMTestCase
            actual_output=actual,          # REQUIRED by most metrics
            expected_output=expected,      # used by GEval when included in evaluation_params
            name=str(row.get("id", i)),    # helpful label if present
        )
        test_cases.append(tc)
        if i < 3:
            logging.info(f"Sample {i}: input_len={len(input_text)}, expected_len={len(expected)}, actual_len={len(actual)}")
    return test_cases

def run_qualitative_evaluation(
    data_format: str,
    evaluation_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    logging.info(f"Starting qualitative evaluation: data_format={data_format}")

    jsonl_path = find_inferenced_jsonl("/root/data")
    rows = load_jsonl(jsonl_path)
    logging.info(f"Loaded {len(rows)} rows from inferenced file: {jsonl_path}")

    # Build test cases
    test_cases = build_llm_test_cases(rows, data_format)
    logging.info(f"Constructed {len(test_cases)} LLMTestCase objects")

    # Create evaluators
    evaluators: List[GEval] = []
    for meta in evaluation_metadata:
        evaluator = create_evaluator_from_metadata(meta)
        evaluators.append(evaluator)
        logging.info(f"Created evaluator: {evaluator.name}")

    # Manually measure each metric to get structured results
    results = {
        "meta": {
            "data_format": data_format,
            "num_samples": len(test_cases),
            "jsonl_path": jsonl_path,
            "evaluators": [e.name for e in evaluators],
        },
        "evaluations": {},
    }

    for metric in evaluators:
        logging.info(f"Measuring metric: {metric.name}")
        scores, reasons = [], []
        for i, tc in enumerate(test_cases):
            metric.measure(tc)  # compute for this test case
            scores.append(float(metric.score))
            # Collect reasons if available (first 5 samples only)
            if getattr(metric, "reason", None) and i < 5:
                reasons.append(str(metric.reason))

        results["evaluations"][metric.name] = {
            "score_avg": round(mean(scores), 4) if scores else 0.0,
            "score_min": min(scores) if scores else 0.0,
            "score_max": max(scores) if scores else 0.0,
            "num_cases": len(scores),
            "sample_reasons": reasons,
            "evaluator_info": {
                "name": metric.name,
                "criteria": metric.criteria,
                "rubric_count": len(metric.rubric) if getattr(metric, "rubric", None) else 0,
            },
        }
        logging.info(f"{metric.name} average score: {results['evaluations'][metric.name]['score_avg']}")

    # Save individual row evaluation results
    output_path = save_qualitative_evaluation_results(rows, test_cases, data_format, results, jsonl_path)
    results["meta"]["output_path"] = output_path

    logging.info("Qualitative evaluation complete")
    return results

def save_qualitative_evaluation_results(original_rows: List[Dict[str, Any]], 
                                      test_cases: List[LLMTestCase],
                                      data_format: str,
                                      results: Dict[str, Any],
                                      original_path: str) -> str:
    """Save original data with qualitative evaluation results to a new JSONL file."""
    # Create output filename
    base_name = os.path.splitext(original_path)[0]
    output_path = f"{base_name}_qualitative_evaluated.jsonl"
    
    logging.info(f"Saving qualitative evaluation results to: {output_path}")
    
    # Add evaluation data to original rows
    evaluated_rows = []
    
    for i, (row, tc) in enumerate(zip(original_rows, test_cases)):
        new_row = row.copy()
        
        # Add evaluation data
        new_row["qualitative_evaluation"] = {
            "data_format": data_format,
            "input": tc.input,
            "expected_output": tc.expected_output,
            "actual_output": tc.actual_output,
            "row_index": i
        }
        
        # Add individual metric scores for this row
        row_metrics = {}
        for metric_name, metric_data in results["evaluations"].items():
            # Get individual score for this test case
            if hasattr(tc, 'metrics') and tc.metrics:
                for metric in tc.metrics:
                    if metric.name == metric_name:
                        row_metrics[metric_name] = {
                            "score": float(metric.score) if hasattr(metric, 'score') else 0.0,
                            "reason": str(metric.reason) if hasattr(metric, 'reason') else None
                        }
                        break
            else:
                # Fallback: use average score if individual not available
                row_metrics[metric_name] = {
                    "score": metric_data.get("score_avg", 0.0),
                    "note": "Using average score (individual scores not available)"
                }
        
        new_row["qualitative_evaluation"]["metrics"] = row_metrics
        evaluated_rows.append(new_row)
    
    # Save to new file
    with open(output_path, "w", encoding="utf-8") as f:
        for row in evaluated_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    logging.info(f"Saved {len(evaluated_rows)} rows with qualitative evaluation results")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Qualitative evaluation for inferenced JSONL data using deepeval (G-Eval).")
    parser.add_argument("--data_format", required=True, choices=["conversation", "instruction", "generic_text"])
    parser.add_argument("--evaluation_metadata", required=True,
                        help="JSON string or file path containing evaluation metadata.")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Parse evaluation metadata
    if os.path.exists(args.evaluation_metadata):
        with open(args.evaluation_metadata, "r", encoding="utf-8") as f:
            evaluation_metadata = json.load(f)
    else:
        evaluation_metadata = json.loads(args.evaluation_metadata)
    if not isinstance(evaluation_metadata, list):
        raise ValueError("evaluation_metadata must be a list of dictionaries")

    results = run_qualitative_evaluation(
        data_format=args.data_format,
        evaluation_metadata=evaluation_metadata,
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))

    # Print summary information
    if "meta" in results:
        print(f"\n📁 Qualitative evaluation completed for: {results['meta']['jsonl_path']}")
        print(f"📊 Total samples processed: {results['meta']['num_samples']}")
        print(f"🔍 Evaluators used: {', '.join(results['meta']['evaluators'])}")
        
        # Print evaluation summaries
        for eval_name, eval_data in results["evaluations"].items():
            if "score_avg" in eval_data:
                print(f"📈 {eval_name}: avg={eval_data['score_avg']} (min={eval_data['score_min']}, max={eval_data['score_max']}, n={eval_data['num_cases']})")
            elif "error" in eval_data:
                print(f"❌ {eval_name}: Error - {eval_data['error']}")
        
        print(f"\n💡 Tip: Use --log_level DEBUG for detailed logs")

if __name__ == "__main__":
    main()
