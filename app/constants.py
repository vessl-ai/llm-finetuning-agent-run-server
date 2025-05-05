# File paths for metrics and model data
METRIC_ROOT_DIR = "/root/data/finetuning-output"
METRIC_TF_EVENT_PATH = METRIC_ROOT_DIR + "/runs"
METRIC_DEEPEVAL_PATH = METRIC_ROOT_DIR + "/evaluation_result.json"
MODEL_PATH = "/root/data/finetuning-output"

# Dataset paths
TEST_DATASET_PATH = "/root/data/llm_finetuning_dataset_test.json"
TEST_DATASET_DEEPEVAL_PATH = "/root/data/llm_finetuning_dataset_eval.json"

# Evaluation metrics
RELEVANCY = "relevancy"
CORRECTNESS = "correctness"
CLARITY = "clarity"
PROFESSIONALISM = "professionalism"
HALLUCINATION = "hallucination"
HUMAN_EVAL = "human_eval"

# Set of supported evaluation metrics
SUPPORTED_METRICS = {
    RELEVANCY,
    CORRECTNESS,
    CLARITY,
    PROFESSIONALISM,
}