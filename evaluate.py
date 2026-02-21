import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

import random
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from PIL import Image

from sklearn.metrics import confusion_matrix
import numpy as np



# ==========================
# Config
# ==========================
DATA_DIR = "data/test/"
MODEL_PATH = "trained_model.pth"
BATCH_SIZE = 32
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset
# ==========================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = dataset.classes
print("Classes:", class_names)

# ==========================
# Load Model
# ==========================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model = model.to(DEVICE)
model.eval()

print("Model Loaded Successfully!")

# ==========================
# Evaluation
# ==========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==========================
# Overall Accuracy
# ==========================
overall_acc = accuracy_score(all_labels, all_preds)
print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")

# ==========================
# Classwise Accuracy
# ==========================
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

num_classes = NUM_CLASSES 

classwise_acc = []

for c in range(num_classes):
    idx = (all_labels == c)         
    correct = np.sum(all_preds[idx] == c)
    total = np.sum(idx)

    acc = correct / total if total > 0 else 0
    classwise_acc.append(acc)

    print(f"Class {c} Accuracy: {acc*100:.2f}%")


cm = confusion_matrix(all_labels, all_preds)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

# Add numbers inside cells
num_classes = NUM_CLASSES
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j], 
                 ha="center", 
                 va="center")

plt.tight_layout()
plt.savefig("confusion_matrix2.png")  
plt.close()                          


# ==========================
# F1 Score
# ==========================
macro_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"F1 Score: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))


def predict_single_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    print(f"\nImage: {image_path}")
    print(f"Predicted Class: {class_names[pred.item()]}")
    print(f"Confidence: {confidence.item()*100:.2f}%")

# Pick random image from dataset
image_path = "data/test/7/6561.png"
predict_single_image(image_path)
