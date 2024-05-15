import argparse


def get_prompting_args():
    """
    Arguments for prompting. You may choose to change or extend these as you see fit.
    """
    parser = argparse.ArgumentParser(
        description="Text-to-SQL experiments with prompting."
    )

    parser.add_argument(
        "-s",
        "--shot",
        type=int,
        default=0,
        help="Number of examples for k-shot learning (0 for zero-shot)",
    )
    parser.add_argument("-p", "--ptype", type=int, default=0, help="Prompt type")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gemma",
        help="Model to use for prompting: gemma (gemma-1.1-2b-it) or codegemma (codegemma-7b-it)",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        action="store_true",
        help="Use a quantized version of the model (e.g. 4bits)",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed to help reproducibility"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment",
        help="How should we name this experiment?",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate for each prompt",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for inference"
    )
    args = parser.parse_args()
    return args
