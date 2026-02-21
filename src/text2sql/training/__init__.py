"""Training: HuggingFace Trainer setup and callbacks."""

from text2sql.training.callbacks import Text2SqlEvalCallback
from text2sql.training.trainer import build_trainer, make_compute_metrics

__all__ = ["build_trainer", "make_compute_metrics", "Text2SqlEvalCallback"]
