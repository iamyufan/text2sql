import torch


# Function to generate SQL predictions
def generate_sql_predictions(model, loader, tokenizer, device, test=False):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    references = []
    with torch.no_grad():
        for texts, queries in loader:
            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            outputs = model.generate(inputs.input_ids, max_length=512)
            decoded_preds = [
                tokenizer.decode(g, skip_special_tokens=True) for g in outputs
            ]
            predictions.extend(decoded_preds)
            if not test:
                references.extend(queries)
    return predictions, references
