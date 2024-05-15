import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AdamW
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

from utils.evaluation import compute_metrics, save_queries_and_records
from utils.data import load_data
from utils.t5_utils import generate_sql_predictions
from options.t5_options import get_t5_args
from dataset.sql_dataset import SQLDataset


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

# Checkpoints
checkpoint_path = "checkpoints"

# Results
results_path = "results"
queries_result_path = os.path.join(results_path, "queries")
records_result_path = os.path.join(results_path, "records")


def main():
    # Parse arguments
    args = get_t5_args()
    model_type = "ft" if args.finetune else "scr"
    num_epochs = args.max_n_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    patience = args.patience_epochs

    # Load the texts and gt data from files
    print("Loading data...")
    train_texts, train_gt_queries = load_data(train_text_path, train_queries_path)
    dev_texts, dev_gt_queries = load_data(dev_text_path, dev_queries_path)
    test_texts, _ = load_data(test_text_path)

    # Check if dev_gt_queries.sql and dev_gt_records.pkl exist in the data path
    if not os.path.exists(dev_gt_queries_path) or not os.path.exists(
        dev_gt_records_path
    ):
        print("Ground truth dev queries and records not found. Computing them...")
        save_queries_and_records(
            dev_gt_queries,
            dev_gt_queries_path,
            dev_gt_records_path,
        )

    # Create datasets
    train_dataset = SQLDataset(train_texts, train_gt_queries)
    dev_dataset = SQLDataset(dev_texts, dev_gt_queries)
    test_dataset = SQLDataset(test_texts)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"> Size of train dataset: {len(train_dataset)}")
    print(f"> Size of dev dataset: {len(dev_dataset)}")
    print(f"> Size of test dataset: {len(test_dataset)}")
    print()

    # Initialize the tokenizer and model
    if model_type == "ft":
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
        # Layer freezing for fine-tuning
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif model_type == "scr":
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        config = T5Config.from_pretrained("google-t5/t5-small")
        config.decoder_start_token_id = tokenizer.pad_token_id
        model = T5ForConditionalGeneration(config)

    # Set the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.to(DEVICE)

    # Train
    for epoch in range(num_epochs):  # number of epochs
        total_loss = 0
        for text, query in tqdm(train_loader):
            # Tokenize the texts and queries
            inputs = tokenizer(
                text, padding=True, truncation=True, return_tensors="pt"
            ).to(DEVICE)
            labels = tokenizer(
                query, padding=True, truncation=True, return_tensors="pt"
            ).input_ids.to(DEVICE)

            # The labels need to be adjusted for T5 which expects -100 for ignored indices
            labels[labels == tokenizer.pad_token_id] = -100

            # Forward pass
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Early stopping
        if (epoch + 1) % 5 == 0:
            # Paths to save the dev model queries and records
            model_query_path = os.path.join(
                queries_result_path, f"dev_model_queries_epoch_{epoch+1}.sql"
            )
            model_record_path = os.path.join(
                records_result_path, f"dev_model_records_epoch_{epoch+1}.pkl"
            )

            # Generate the SQL queries from the texts
            dev_model_queries, dev_gt_queries = generate_sql_predictions(
                model, dev_loader, tokenizer, DEVICE
            )

            # Save the queries and records
            save_queries_and_records(
                dev_model_queries,
                os.path.join(
                    queries_result_path,
                    model_query_path,
                ),
                os.path.join(
                    records_result_path,
                    model_record_path,
                ),
            )

            # Save the model
            torch.save(
                model.state_dict(),
                os.join.path(checkpoint_path, f"model_epoch_{epoch+1}.pt"),
            )

            # Calculate the metrics
            dev_sql_em, dev_record_em, dev_record_f1, _ = compute_metrics(
                dev_gt_queries_path,
                model_query_path,
                dev_gt_records_path,
                model_record_path,
            )
            print(f"> Dev SQL EM: {dev_sql_em}")
            print(f"> Dev Record EM: {dev_record_em}")
            print(f"> Dev Record F1: {dev_record_f1}\n")

            # Check if the dev loss has increased for the past 3 epochs
            if epoch >= patience + 1:
                prev_losses = [total_loss / len(train_loader)]
                for i in range(1, 4):
                    prev_losses.append(
                        torch.load(f"{checkpoint_path}model_epoch_{epoch-i+1}.pt")
                    )
                if all(prev_losses[i] < prev_losses[i + 1] for i in range(patience)):
                    print("Early stopping triggered. Stopping training.")
                    break

        print(f"\nEpoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    # Generate the SQL queries from the test texts
    print("\nGenerating SQL queries for test texts...")
    test_model_queries, _ = generate_sql_predictions(
        model, test_loader, tokenizer, DEVICE
    )
    save_queries_and_records(
        test_model_queries,
        os.path.join(queries_result_path, f"t5_{model_type}_model_queries.sql"),
        os.path.join(records_result_path, f"t5_{model_type}_model_records.pkl"),
    )


if __name__ == "__main__":
    main()
