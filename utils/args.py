import argparse

def get_t5_args():
    """
    Arguments for training. You may choose to change or extend these as you see fit.
    """
    parser = argparse.ArgumentParser(description="T5 training loop")

    # Model hyperparameters
    parser.add_argument(
        "--finetune", action="store_true", help="Whether to finetune T5 or not"
    )

    # Training hyperparameters
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="AdamW",
        choices=["AdamW"],
        help="What optimizer to use",
    )
    parser.add_argument("--learning_rate", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0)

    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["none", "cosine", "linear"],
        help="Whether to use a LR scheduler and what type to use if so",
    )
    parser.add_argument(
        "--num_warmup_epochs",
        type=int,
        default=0,
        help="How many epochs to warm up the learning rate for if using a scheduler",
    )
    parser.add_argument(
        "--max_n_epochs",
        type=int,
        default=0,
        help="How many epochs to train the model for",
    )
    parser.add_argument(
        "--patience_epochs",
        type=int,
        default=0,
        help="If validation performance stops improving, how many epochs should we wait before stopping?",
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="If set, we will use wandb to keep track of experiments",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment",
        help="How should we name this experiment?",
    )

    # Data hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)

    args = parser.parse_args()
    return args