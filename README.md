# DistilBERT Review Classification

## Docker Image Build Instructions

### Build Docker Image
```bash
docker build -t distilbert-review-classifier .
```

### Run Container
```bash
docker run distilbert-review-classifier
```

---

## Evaluation Results

- Accuracy: `0.625`
- Macro F1: `0.628`
- Loss Curve: `results/loss_curve.png`
- Confusion Matrix: `results/eval_confusion_matrix.png`
- Classification Report: `results/classification_report.json`

---

## Model Link (Hugging Face)

Model available at:
[Hugging Face Model](https://huggingface.co/arpita2desh/distilbert-genres-mlops)

---

Note: Other results like training loss graph and confusion matrix can also be found in `results/`
