import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

from utils2 import prepare_datasets, ReviewsDataset


MODEL_PATH = "distilbert-reviews-genres"
MAX_LENGTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    print("Preparing dataset...")
    _, _, test_texts, test_labels = prepare_datasets()

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

    id2label = model.config.id2label
    label2id = model.config.label2id
    unique_labels = list(label2id.keys())

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

    print("Running inference...")

    with torch.no_grad():
        for batch in test_dataset:
            inputs = {
                k: v.unsqueeze(0).to(device)
                for k, v in batch.items()
            }

            labels = inputs.pop("labels")
            outputs = model(**inputs, labels=labels)

            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.append(preds.item())

    # ==========================
    # Metrics
    # ==========================
    avg_loss = total_loss / len(test_dataset)
    accuracy = accuracy_score(test_labels_encoded, predictions)
    macro_f1 = f1_score(test_labels_encoded, predictions, average="macro")
    weighted_f1 = f1_score(test_labels_encoded, predictions, average="weighted")

    print("\n===== Evaluation Results =====")
    print("Loss:", avg_loss)
    print("Accuracy:", accuracy)
    print("Macro F1:", macro_f1)
    print("Weighted F1:", weighted_f1)

    # Save summary metrics
    summary_metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

    with open("evaluation_summary.json", "w") as f:
        json.dump(summary_metrics, f, indent=4)

    # ==========================
    # Confusion Matrix
    # ==========================
    print("Saving confusion matrix...")

    cm = confusion_matrix(test_labels_encoded, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=unique_labels,
        yticklabels=unique_labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("eval_confusion_matrix.png")
    plt.close()

    # ==========================
    # Classification Report
    # ==========================
    report = classification_report(
        test_labels_encoded,
        predictions,
        target_names=unique_labels,
        output_dict=True
    )

    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # ==========================
    # Prediction Distribution
    # ==========================
    pred_counts = np.bincount(predictions)

    plt.figure()
    plt.bar(unique_labels, pred_counts)
    plt.xlabel("Class")
    plt.ylabel("Number of Predictions")
    plt.title("Prediction Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("prediction_distribution.png")
    plt.close()

    print("\nAll evaluation results saved successfully.")
