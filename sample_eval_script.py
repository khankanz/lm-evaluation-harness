import pandas as pd
import numpy as np
import lm_eval
from lm_eval.logging_utils import WandbLogger
from fastcore.basics import patch
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from lm_eval.logging_utils import WandbLogger

@patch
def _generate_dataset(self: WandbLogger, data: List[Dict[str,Any]], config: Dict[str, Any]) -> pd.DataFrame:    
    def _obfuscate(field_value: str) -> str:
        return field_value[:15] + '*'*(len(field_value) - 10)

    ids = [x['doc_id'] for x in data]
    labels = [x['target'] for x in data]
    resps = [""] * len(ids)
    filtered_resp = [""] * len(ids)
    model_outputs = {}

    metrics_list = config["metric_list"]
    metrics = {}
    for metric in metrics_list:
        metric = metric.get("metric")
        if metric in ["word_perplexity", "byte_perplexity", "bits_per_byte"]:
            metrics[f"{metric}_loglikelihood"] = [x[metric][0] for x in data]
            if metric in ["byte_perplexity", "bits_per_byte"]:
                metrics[f"{metric}_bytes"] = [x[metric][1] for x in data]
            else:
                metrics[f"{metric}_words"] = [x[metric][1] for x in data]
        else:
            metrics[metric] = [x[metric] for x in data]

    if config["output_type"] == "loglikelihood":
        instance = [_obfuscate(x['arguments'][0][0]) for x in data]
        labels = [x["arguments"][0][1] for x in data]
        resps = [
            f'log probability of continuation is {x["resps"][0][0][0]} '
            + "\n\n"
            + "continuation will {} generated with greedy sampling".format(
                "not be" if not x["resps"][0][0][1] else "be"
            )
            for x in data
        ]
        filtered_resps = [
            f'log probability of continuation is {x["filtered_resps"][0][0]} '
            + "\n\n"
            + "continuation will {} generated with greedy sampling".format(
                "not be" if not x["filtered_resps"][0][1] else "be"
            )
            for x in data
        ]
    elif config["output_type"] == "multiple_choice":
        instance = [_obfuscate(x['arguments'][0][0]) for x in data]
        choices = [
            "\n".join([f"{idx}. {y[1]}" for idx, y in enumerate(x["arguments"])])
            for x in data
        ]
        resps = [np.argmax([n[0][0] for n in x["resps"]]) for x in data]
        filtered_resps = [
            np.argmax([n[0] for n in x["filtered_resps"]]) for x in data
        ]
    elif config["output_type"] == "loglikelihood_rolling":
        instance = [_obfuscate(x['arguments'][0][0]) for x in data]
        resps = [x["resps"][0][0] for x in data]
        filtered_resps = [x["filtered_resps"][0] for x in data]
    elif config["output_type"] == "generate_until":
        instance = [_obfuscate(x['arguments'][0][0]) for x in data]
        resps = [x["resps"][0][0] for x in data]
        filtered_resps = [x["filtered_resps"][0] for x in data]

    model_outputs["raw_predictions"] = resps
    model_outputs["filtered_predictions"] = filtered_resps

    df_data = {
        "id": ids,
        "data": instance,
    }
    if config["output_type"] == "multiple_choice":
        df_data["choices"] = choices

    tmp_data = {
        "input_len": [len(x) for x in instance],
        "labels": labels,
        "output_type": config["output_type"],
    }
    df_data.update(tmp_data)
    df_data.update(model_outputs)
    df_data.update(metrics)

    return pd.DataFrame(df_data)

results = lm_eval.simple_evaluate(
    model='hf',
    model_args='mistralai/Mistral-7B-Instruct-v0.2',
    tasks=['bpe','prev_cancer','menopausal_status','breast_density','modality','purpose'],
)

wandb_logger = WandbLogger(
    project='birads-eval-v1', job_type='eval'
)

wandb_logger.post_init(results)
wandb_logger.log_eval_result()
wandb_logger.log_eval_samples(results['samples'])

#lm-evaluation-harness$ lm_eval --model hf --model_args   
#--tasks prev_cancer --device cpu 
#--output_path output/roberta-base-squad2 --wandb_args project=birads-eval-v1
