# llm-finetuning-agent-run-server

This repository contains the server code required for running the llm-finetuning-agent. It includes various components for evaluation, API serving, and chatbot functionality.

## Prerequisites

- Python 3.x
- uv (Python package manager)

## Setup

1. Install uv

```bash
cd app
pip uv
```

## Running the Server

The server consists of multiple components that need to be run in sequence:

### 1. Evaluation

Run the evaluation script with specified metrics:

```bash
uv run eval.py --metrics relevancy correctness clarity professionalism --data-generation-method raft
```

### 2. Run API Server

Start the main API server:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8080 > run_server.log 2>&1 &
```

### 3. Chatbot Serving

Start the chatbot server:

```bash
uv run python chatbot.py > chatbot.log 2>&1 &
```
