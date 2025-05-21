import json
import argparse
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    AnswerRelevancyMetric,
    GEval
)

from constants import (
    TEST_DATASET_PATH, 
    TEST_DATASET_DEEPEVAL_PATH, 
    METRIC_DEEPEVAL_PATH, 
    MODEL_PATH
)

# TODO: Using custom model for evaluation
class CustomModelLLM(DeepEvalBaseLLM):
    """Custom LLM implementation for deepeval using a Hugging Face model."""
    
    def __init__(self, model: Any, tokenizer: Any):
        """
        Initialize the custom LLM model.
        
        Args:
            model: The loaded HuggingFace model
            tokenizer: The model's tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self) -> Any:
        """Return the loaded model."""
        return self.model

    def generate(self, prompt: str) -> str:
        """
        Generate text response for the given prompt.
        
        Args:
            prompt: The input prompt text
            
        Returns:
            Generated response text
        """
        model = self.load_model()
        device = "cuda"  # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        """Async version of generate method."""
        return self.generate(prompt)

    def get_model_name(self) -> str:
        """Return the model name."""
        return "Custom LLM Model"


def set_deepeval_model(model_path: str) -> CustomModelLLM:
    """
    Load and set up the model for deepeval.
    
    Args:
        model_path: Path to the pretrained model
        
    Returns:
        Initialized CustomModelLLM instance
    """
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return CustomModelLLM(model=model, tokenizer=tokenizer)


def generate_response() -> str:
    """
    Generate responses for test dataset using the finetuned model.
    
    Returns:
        Path to the processed dataset with model responses
    """
    generation_pipeline = pipeline(
        task="text-generation", 
        model=MODEL_PATH, 
        device_map="auto"
    )

    with open(TEST_DATASET_PATH, "r") as f:
        dataset_json = json.load(f)
    
    for ds in dataset_json:
        question = ds['question']
        chat = [
            {"role": "system", "content": "Answer clearly about user input"},
            {"role": "user", "content": question}
        ]
        response = generation_pipeline(chat)
        response = response[0]["generated_text"][-1]["content"]
        ds['actual_output'] = response

    with open(TEST_DATASET_DEEPEVAL_PATH, 'w') as json_file:
        json.dump(dataset_json, json_file, indent=4)

    return TEST_DATASET_DEEPEVAL_PATH


def set_deepeval_dataset(testset_label: str) -> EvaluationDataset:
    """
    Set up deepeval dataset from JSON file.
    
    Returns:
        Initialized EvaluationDataset
    """
    deepeval_dataset = EvaluationDataset()
        
    deepeval_dataset.add_test_cases_from_json_file(
        file_path=TEST_DATASET_DEEPEVAL_PATH,
        input_key_name="question",
        actual_output_key_name="actual_output",
        expected_output_key_name=testset_label,
    )
    return deepeval_dataset


def generate_evaluate_pipeline(metric_settings: List[dict], testset_label: str) -> None:
    """
    Run the generation and evaluation pipeline with specified metrics.
    
    Args:
        metric_settings: List of metric names to use for evaluation
    """
    # Generate responses for test dataset
    generate_response()
    
    # Set up evaluation dataset
    deepeval_dataset = set_deepeval_dataset(testset_label)

    # Initialize metrics
    metrics = [AnswerRelevancyMetric(threshold=0.5)]  # Default metric
    
    # Add requested metrics
    for metric_setting in metric_settings:
        metrics.append(
            GEval(
                name=metric_setting['name'],
                criteria=metric_setting['criteria'],
                threshold=metric_setting['threshold'],
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT]
            )
        )

    # Run evaluation and save results
    evaluation_result = deepeval_dataset.evaluate(metrics)
    with open(METRIC_DEEPEVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_result.model_dump(), f, indent=4)
        
    print("Evaluation completed. Results saved to evaluation_result.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM with selected metrics.")
    parser.add_argument(
        "--testset-label",
        required=False,
        default="answer",
    )
    parser.add_argument(
        "--metrics",
        help="List of metrics to use (e.g., [{'name': 'Relevancy', 'criteria': 'Check if the actual output directly addresses the input.', 'threshold': 0.5}])",
        required=True,
    )

    args = parser.parse_args()
    metric_settings = json.loads(args.metrics)
    generate_evaluate_pipeline(metric_settings, args.testset_label)
