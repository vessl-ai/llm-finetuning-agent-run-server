import os
import json
import logging
from typing import Dict, List

import pandas as pd
import gradio as gr
from fastapi import APIRouter, HTTPException, FastAPI
from pydantic import BaseModel
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from constants import METRIC_TF_EVENT_PATH, METRIC_DEEPEVAL_PATH

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/llm_finetuning_run", tags=["llm_finetuning_run"])


class LLMFinetuningRunMetricTFEventResponse(BaseModel):
    result: Dict


class LLMFinetuningRunMetricDeepevalResponse(BaseModel):
    result: Dict


def find_tfevent_files(root_dir: str) -> List[str]:
    """Find all tensorboard event files under the root directory."""
    tfevent_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if 'tfevents' in filename:
                full_path = os.path.join(dirpath, filename)
                tfevent_files.append(full_path)
    return tfevent_files


def tfevent_parsing(raw_data: List[Dict]) -> pd.DataFrame:
    """Parse tensorboard event data into a pandas DataFrame."""
    records = []
    for entry in raw_data:
        file = entry["tfevent_file"]
        for event in entry["events"]:
            tag = event["tag"]
            for value in event["values"]:
                records.append({
                    "tfevent_file": file,
                    "tag": tag,
                    "value": value
                })
    return pd.DataFrame(records)


def deepeval_parsing(raw_data: Dict) -> pd.DataFrame:
    """Parse deepeval test results into a pandas DataFrame."""
    test_results = raw_data["test_results"]

    rows = []
    for test in test_results:
        input_text = test.get("input")
        actual_output = test.get("actual_output")
        expected_output = test.get("expected_output")
    
        for metric in test.get("metrics_data", []):
            rows.append({
                "name": metric.get("name"),
                "score": metric.get("score"),
                "input": input_text,
                "actual_output": actual_output,
                "expected_output": expected_output
            })
    
    return pd.DataFrame(rows)


@router.get("/metric/tensorboard")
async def get_llm_finetuning_run_tensorboard_metric() -> LLMFinetuningRunMetricTFEventResponse:
    """Get the tensorboard metric for the LLM finetuning run."""
    try: 
        tfevent_files = find_tfevent_files(METRIC_TF_EVENT_PATH)
        metrics = []
        
        for tfevent_file in tfevent_files:
            event_acc = EventAccumulator(tfevent_file)
            event_acc.Reload()
            
            event_history = []
            for tag in event_acc.Tags()['scalars']:
                events = event_acc.Scalars(tag)
                values = [event.value for event in events]
                event_history.append({
                    "tag": tag,
                    "values": values
                })
                
            metrics.append({
                "tfevent_file": tfevent_file,
                "events": event_history
            })
            
        metrics_df = tfevent_parsing(metrics)
        metrics_df_dict = json.loads(metrics_df.to_json())
        
        return LLMFinetuningRunMetricTFEventResponse(result=metrics_df_dict)
    
    except Exception as e:
        logger.error(f"Error getting LLM finetuning run tensorboard metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app = FastAPI()
app.include_router(router)

with open(METRIC_DEEPEVAL_PATH, "r") as f:
    metrics = json.load(f)    
    metric_df = deepeval_parsing(metrics)

with gr.Blocks() as demo:
    gr.Dataframe(metric_df)

app = gr.mount_gradio_app(app, demo, path="/api/v1/llm_finetuning_run/metric/deepeval")
