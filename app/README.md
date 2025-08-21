# Model Evaluation Pipeline

This pipeline performs model inference and quantitative/qualitative evaluation on JSONL data, saving detailed evaluation results for each row.

## File Structure

- `inference.py`: Performs model inference on JSONL data and generates `_inferenced.jsonl` file
- `quantitative_evaluate.py`: Calculates quantitative metrics (accuracy, precision, recall, F1, BLEU, ROUGE) + saves individual row results
- `qualitative_evaluate.py`: Performs qualitative evaluation (using deepeval library, custom criteria and rubric-based) + saves individual row results
- `example_evaluation_metadata.json`: Example metadata for qualitative evaluation

## Usage

### Step 1: Run Inference

```bash
# Using local model
python inference.py \
    --data_format conversation \
    --model_name Qwen/Qwen3-4B

# Using vLLM endpoint
python inference.py \
    --data_format instruction \
    --model_name http://localhost:8000 \
    --remote_model Qwen/Qwen3-8B \
    --api_key EMPTY
```

### Step 2: Run Individual Evaluations

#### Quantitative Evaluation Only

```bash
python quantitative_evaluate.py \
    --data_format conversation \
    --metrics accuracy_precision_recall_f1 bleu_rouge
# Result: {original_name}_quantitative_evaluated.jsonl
```

#### Qualitative Evaluation Only

```bash
python qualitative_evaluate.py \
    --data_format conversation \
    --evaluation_metadata example_evaluation_metadata.json
# Result: {original_name}_qualitative_evaluated.jsonl
```

## Output File Structure

### `{original_name}_quantitative_evaluated.jsonl` (Quantitative Evaluation)

```json
{
  "messages": [...],
  "inferenced": "Model generated response",
  "evaluation": {
    "data_format": "conversation",
    "expected_output": "Expected response",
    "actual_output": "Actual response",
    "row_index": 0,
    "accuracy_precision_recall_f1": {
      "accuracy": 0.8,
      "precision": 0.75,
      "recall": 0.8,
      "f1": 0.77,
      "actual_tokens": 15,
      "expected_tokens": 12,
      "common_tokens": 9
    },
    "bleu_rouge": {
      "bleu": 0.6,
      "rouge1": 0.75,
      "rouge2": 0.5,
      "rougeL": 0.75
    }
  }
}
```

### `{original_name}_qualitative_evaluated.jsonl` (Qualitative Evaluation)

```json
{
  "messages": [...],
  "inferenced": "Model generated response",
  "qualitative_evaluation": {
    "data_format": "conversation",
    "input": "User input",
    "expected_output": "Expected response",
    "actual_output": "Actual response",
    "row_index": 0,
    "metrics": {
      "Correctness": {
        "score": 8.5,
        "reason": "Score 8.5/10 based on word overlap (0.85) and length similarity (0.92)"
      },
      "Completeness": {
        "score": 7.2,
        "reason": "Score 7.2/10 based on word overlap (0.72) and length similarity (0.88)"
      }
    }
  }
}
```

## Data Formats

### Conversation Format

```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is the capital of France?" },
    { "role": "assistant", "content": "The capital of France is Paris." }
  ]
}
```

### Instruction Format

```json
{
  "prompt": "What is the capital of France?",
  "completion": "The capital of France is Paris."
}
```

### Generic Text Format

```json
{
  "text": "The capital of France is Paris."
}
```

## Metrics

### Quantitative Metrics

- **accuracy_precision_recall_f1**: Accuracy, precision, recall, F1 score
- **bleu_rouge**: BLEU, ROUGE-1, ROUGE-2, ROUGE-L score

### Qualitative Metrics (using deepeval)

- Custom criteria and rubric-based evaluation
- 0-10 scoring system
- Professional evaluation using deepeval's GEval

## Dependencies

```bash
pip install tqdm requests evaluate rouge_score nltk deepeval
# or
pip install -r requirements.txt
```

## Qualitative Evaluation Metadata Format

```json
[
  {
    "name": "Correctness",
    "criteria": "Determine whether the actual output is factually correct based on the expected output.",
    "rubric": [
      { "score_range": [0, 2], "expected_outcome": "Factually incorrect." },
      { "score_range": [3, 6], "expected_outcome": "Mostly correct." },
      {
        "score_range": [7, 9],
        "expected_outcome": "Correct but missing minor details."
      },
      { "score_range": [10, 10], "expected_outcome": "100% correct." }
    ]
  }
]
```

## Workflow Summary

1. **Inference**: `inference.py` → `{original_name}_inferenced.jsonl`
2. **Individual Evaluations**:
   - `quantitative_evaluate.py` → `{original_name}_quantitative_evaluated.jsonl`
   - `qualitative_evaluate.py` → `{original_name}_qualitative_evaluated.jsonl`

Each step includes detailed evaluation results for individual rows, enabling data analysis and model performance improvement.
