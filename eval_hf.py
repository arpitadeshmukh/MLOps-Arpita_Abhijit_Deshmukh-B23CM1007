import torch
import numpy as np
import json

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

from utils2 import prepare_datasets, ReviewsDataset


HF_REPO = "arpita2desh/distilbert-genres-mlops"
MAX_LENGTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    print("Loading model from Hugging Face Hub...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(HF_REPO)
    model = DistilBertForSequenceClassification.from_pretrained(HF_REPO).to(device)

    print("Preparing dataset...")
    _, _, test_texts, test_labels = prepare_datasets()

    label2id = model.config.label2id
    test_labels_encoded = [label2id[l] for l in test_labels]

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    test_dataset = ReviewsDataset(test_encodings, test_labels_encoded)

    model.eval()
    predictions = []
    total_loss = 0

    with torch.no_grad():
        for batch in test_dataset:
            inputs = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}
            labels = inputs.pop("labels")

            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.append(preds.item())

    avg_loss = total_loss / len(test_dataset)
    accuracy = accuracy_score(test_labels_encoded, predictions)
    macro_f1 = f1_score(test_labels_encoded, predictions, average="macro")

    results = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }

    print("\nEvaluation from Hugging Face repo:")
    print(results)

    with open("hf_evaluation.json", "w") as f:
        json.dump(results, f, indent=4)
