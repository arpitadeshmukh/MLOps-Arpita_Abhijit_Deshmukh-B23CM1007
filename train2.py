import torch
import numpy as np
import os

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import accuracy_score
from utils2 import prepare_datasets, ReviewsDataset

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
SAVE_DIR = "distilbert-reviews-genres"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    acc = accuracy_score(pred.label_ids, preds)
    return {"accuracy": acc}


if __name__ == "__main__":

    print("Preparing dataset...")
    train_texts, train_labels, test_texts, test_labels = prepare_datasets()

    unique_labels = sorted(set(train_labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_labels_encoded = [label2id[l] for l in train_labels]
    test_labels_encoded = [label2id[l] for l in test_labels]

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    train_dataset = ReviewsDataset(train_encodings, train_labels_encoded)
    test_dataset = ReviewsDataset(test_encodings, test_labels_encoded)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    training_args = TrainingArguments(
	    output_dir="./results",
	    num_train_epochs=3,
	    per_device_train_batch_size=8,
	    per_device_eval_batch_size=16,
	    logging_steps=10,
	    do_train=True,
	    do_eval=True,
	    report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    os.environ["WANDB_DISABLED"] = "true"

    # ==========================
    # Training
    # ==========================
    print("Training...")
    trainer.train()

    # ==========================
    # Save Loss Curves
    # ==========================
    print("Saving loss curves...")

    log_history = trainer.state.log_history
    train_loss = []
    eval_loss = []
    epochs = []

    for log in log_history:
        if "loss" in log and "epoch" in log:
            train_loss.append(log["loss"])
            epochs.append(log["epoch"])
        if "eval_loss" in log:
            eval_loss.append(log["eval_loss"])

    plt.figure()
    plt.plot(epochs, train_loss, label="Training Loss")

    if len(eval_loss) > 0:
        plt.plot(range(1, len(eval_loss) + 1), eval_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig("loss_curve.png")
    plt.close()

    # ==========================
    # Evaluation
    # ==========================
    print("Evaluating...")
    metrics = trainer.evaluate()
    print(metrics)

    # ==========================
    # Confusion Matrix
    # ==========================
    print("Generating confusion matrix...")

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)

    cm = confusion_matrix(test_labels_encoded, preds)

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
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ==========================
    # Save Model
    # ==========================
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("Model saved successfully.")
