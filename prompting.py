# Packages import
import os
import torch
from torch.utils.data import DataLoader
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.evaluation import compute_metrics, save_queries_and_records
from utils.data import load_data
from utils.prompting_utils import (
    generate_sql_queries,
    extract_sql_query,
    schema_description,
)
from dataset.sql_dataset import SQLDataset
from options.prompting_options import get_prompting_args


# Set the device
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"

# Paths
# Data paths
data_path = "data"
train_text_path = os.path.join(data_path, "train.nl")
train_queries_path = os.path.join(data_path, "train.sql")
dev_text_path = os.path.join(data_path, "dev.nl")
dev_queries_path = os.path.join(data_path, "dev.sql")
test_text_path = os.path.join(data_path, "test.nl")
dev_gt_queries_path = os.path.join(data_path, "dev_gt_queries.sql")
dev_gt_records_path = os.path.join(data_path, "dev_gt_records.pkl")

# Results
results_path = "results"
queries_result_path = os.path.join(results_path, "queries")
records_result_path = os.path.join(results_path, "records")


def main():
    # Options
    args = get_prompting_args()
    k = args.shot
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size

    # Load the token from .env file
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    login(token=huggingface_token, add_to_git_credential=True)

    # Load datasets
    train_texts, train_gt_queries = load_data(train_text_path, train_queries_path)
    dev_texts, dev_gt_queries = load_data(dev_text_path, dev_queries_path)
    test_texts, _ = load_data(test_text_path)

    # Create datasets
    dev_dataset = SQLDataset(dev_texts, dev_gt_queries)
    test_dataset = SQLDataset(test_texts)

    # Create dataloaders
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-1.1-2b-it",
        device_map="cuda",
        torch_dtype=torch.float16,
        revision="float16",
    )

    # Generate results
    print("Generating results...")
    dev_generated_queries = generate_sql_queries(
        dev_loader,
        schema_description,
        model,
        tokenizer,
        train_texts,
        train_gt_queries,
        k=k,
        max_new_tokens=max_new_tokens,
    )

    # Clean the results
    print("Cleaning results...")
    dev_model_queries = [
        extract_sql_query(dev_generated_queries[i])
        for i in range(len(dev_generated_queries))
    ]

    # Save the results
    print("Saving results...")
    # Paths to save the dev model queries and records
    model_query_path = os.path.join(
        queries_result_path, f"llm_dev_model_queries_epoch_.sql"
    )
    model_record_path = os.path.join(
        records_result_path, f"llm_dev_model_records_epoch_.pkl"
    )
    save_queries_and_records(dev_model_queries, model_query_path, model_record_path)

    # Compute the metrics
    dev_sql_em, dev_record_em, dev_record_f1, _ = compute_metrics()

    print(f"> Dev SQL EM: {dev_sql_em}")
    print(f"> Dev Record EM: {dev_record_em}")
    print(f"> Dev Record F1: {dev_record_f1}")
    print()

    # Generate the SQL queries for the test texts
    test_generated_queries = generate_sql_queries(
        test_loader,
        schema_description,
        model,
        tokenizer,
        train_texts,
        train_gt_queries,
        k=k,
        max_new_tokens=max_new_tokens,
        test=True,
    )
    test_model_queries = [
        extract_sql_query(test_generated_queries[i])
        for i in range(len(test_generated_queries))
    ]
    save_queries_and_records(
        test_model_queries,
        os.path.join(queries_result_path, f"llm_model_queries.sql"),
        os.path.join(records_result_path, f"llm_model_records.pkl"),
    )


if __name__ == "__main__":
    main()
