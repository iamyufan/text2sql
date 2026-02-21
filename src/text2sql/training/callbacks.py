"""Custom W&B and eval callbacks for training."""

import time
from typing import Any

import numpy as np
import torch
import wandb
from datasets import Dataset
from transformers import TrainerCallback


class Text2SqlEvalCallback(TrainerCallback):
    """Logs avg_input_tokens and inference latency (p50/p95) to W&B after eval."""

    def __init__(
        self,
        eval_dataset: Dataset,
        tokenizer: Any,
        generation_max_length: int = 256,
        latency_sample_size: int = 100,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.generation_max_length = generation_max_length
        self.latency_sample_size = min(latency_sample_size, len(eval_dataset))

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or len(self.eval_dataset) == 0:
            return
        model.eval()
        device = next(model.parameters()).device

        # Avg input tokens from dataset
        input_token_counts = [
            sum(
                1
                for x in self.eval_dataset[i]["input_ids"]
                if x != self.tokenizer.pad_token_id
            )
            for i in range(len(self.eval_dataset))
        ]
        avg_input_tokens = float(np.mean(input_token_counts))

        # Per-example latency on a sample
        latencies_sec = []
        if self.latency_sample_size > 0:
            indices = np.linspace(
                0, len(self.eval_dataset) - 1, self.latency_sample_size, dtype=int
            )
            with torch.no_grad():
                for idx in indices:
                    row = self.eval_dataset[int(idx)]
                    input_ids = torch.tensor(
                        [row["input_ids"]], dtype=torch.long, device=device
                    )
                    attention_mask = torch.tensor(
                        [row["attention_mask"]], dtype=torch.long, device=device
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=self.generation_max_length,
                        num_beams=4,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    latencies_sec.append(time.perf_counter() - t0)
        else:
            latencies_sec = [0.0]

        metrics = {
            "eval_avg_input_tokens": avg_input_tokens,
            "eval_inference_latency_p50": float(np.percentile(latencies_sec, 50)),
            "eval_inference_latency_p95": float(np.percentile(latencies_sec, 95)),
        }
        if state.is_world_process_zero:
            wandb.log(metrics, step=state.global_step)
            control.should_log = True
