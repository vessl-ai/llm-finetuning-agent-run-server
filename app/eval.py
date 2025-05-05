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
    RELEVANCY, 
    CORRECTNESS, 
    CLARITY, 
    PROFESSIONALISM, 
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
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    with open(TEST_DATASET_PATH, "r") as f:
        dataset_json = json.load(f)
    
    for ds in dataset_json:
        prompt = ds['prompt']
        chat = [
            {"role": "system", "content": "Answer clearly about user input"},
            {"role": "user", "content": prompt}
        ]
        response = generation_pipeline(chat)
        response = response[0]["generated_text"][-1]["content"]
        ds['actual_output'] = response

    with open(TEST_DATASET_DEEPEVAL_PATH, 'w') as json_file:
        json.dump(dataset_json, json_file, indent=4)

    return TEST_DATASET_DEEPEVAL_PATH


def set_deepeval_dataset() -> EvaluationDataset:
    """
    Set up deepeval dataset from JSON file.
    
    Returns:
        Initialized EvaluationDataset
    """
    deepeval_dataset = EvaluationDataset()

    # Add test cases from JSON file
    deepeval_dataset.add_test_cases_from_json_file(
        file_path=TEST_DATASET_DEEPEVAL_PATH,
        input_key_name="prompt",
        actual_output_key_name="actual_output",
        expected_output_key_name="response",
    )
    return deepeval_dataset


def generate_evaluate_pipeline(metric_names: List[str]) -> None:
    """
    Run the generation and evaluation pipeline with specified metrics.
    
    Args:
        metric_names: List of metric names to use for evaluation
    """
    # Generate responses for test dataset
    generate_response()
    
    # Set up evaluation dataset
    deepeval_dataset = set_deepeval_dataset()

    # Initialize metrics
    metrics = [AnswerRelevancyMetric(threshold=0.5)]  # Default metric
    
    # Add requested metrics
    for name in metric_names:
        name = name.lower()
        
        if name == RELEVANCY:
            metrics.append(GEval(
                name="Relevancy",
                criteria="Check if the actual output directly addresses the input.",
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT]
            ))

        if name == CORRECTNESS:
            metrics.append(GEval(
                name="Correctness",
                criteria="Determine whether the actual output is factually correct based on the expected output.",
                evaluation_steps=[
                    "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                    "You should also heavily penalize omission of detail",
                    "Vague language, or contradicting OPINIONS, are OK"
                ],
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            ))

        if name == CLARITY:
            metrics.append(GEval(
                name="Clarity",
                evaluation_steps=[
                    "Evaluate whether the response uses clear and direct language.",
                    "Check if the explanation avoids jargon or explains it when used.",
                    "Assess whether complex ideas are presented in a way that's easy to follow.",
                    "Identify any vague or confusing parts that reduce understanding."
                ],
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            ))
            
        if name == PROFESSIONALISM:
            metrics.append(GEval(
                name="Professionalism",
                criteria="Assess the level of professionalism and expertise conveyed in the response.",
                evaluation_steps=[
                    "Determine whether the actual output maintains a professional tone throughout.",
                    "Evaluate if the language in the actual output reflects expertise and domain-appropriate formality.",
                    "Ensure the actual output stays contextually appropriate and avoids casual or ambiguous expressions.",
                    "Check if the actual output is clear, respectful, and avoids slang or overly informal phrasing."
                ],
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            ))

    # Run evaluation and save results
    evaluation_result = deepeval_dataset.evaluate(metrics)
    with open(METRIC_DEEPEVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_result.model_dump(), f, indent=4)
        
    print("Evaluation completed. Results saved to evaluation_result.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM with selected metrics.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="List of metrics to use (e.g., --metrics relevancy correctness hallucination)",
        required=True,
    )

    args = parser.parse_args()
    generate_evaluate_pipeline(args.metrics)
