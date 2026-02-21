import os
import torch
import wandb
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch import nn, optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
wandb.login()

# ============================================================
# Configuration
# ============================================================
BATCH_SIZE = 64
NUM_EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_MODEL_PATH = "best_resnet18_cifar10_subset.pth"

wandb.init(
    project="resnet18-cifar10-subset",
    config={
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "model": "resnet18_pretrained",
        "dataset": "CIFAR-10 Subset",
    },
)


# ============================================================
# Load Hugging Face Dataset
# ============================================================
dataset = load_dataset("Chiranjeev007/CIFAR-10_Subset")
print(dataset)

train_ds = dataset["train"]
val_ds   = dataset["validation"]
test_ds  = dataset["test"]

# ============================================================
# Image Transforms
# ============================================================
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

# ============================================================
# Custom Dataset Wrapper
# ============================================================
class CIFAR10Subset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[int(idx)]
        image, label = item["image"], item["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

train_dataset = CIFAR10Subset(train_ds, transform=transform_train)
val_dataset   = CIFAR10Subset(val_ds,   transform=transform_eval)
test_dataset  = CIFAR10Subset(test_ds,  transform=transform_eval)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ============================================================
# Model: Pretrained ResNet18
# ============================================================
model = models.resnet18(pretrained=True)

# Change num_classes → 10
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ============================================================
# Training & Validation
# ============================================================
def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc="Train", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return running_loss / len(train_loader), correct / total


def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(val_loader, desc="Val", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return running_loss / len(val_loader), correct / total

best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()

    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/acc": train_acc,
        "val/loss": val_loss,
        "val/acc": val_acc,
    })

    print(
        f"Epoch {epoch}/{NUM_EPOCHS} | "
        f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
    )

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)

print("Training complete!")
print("Best Val Accuracy:", best_val_acc)

# ============================================================
# Test Evaluation
# ============================================================
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

test_loss = 0
test_correct = 0
test_total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        test_total += labels.size(0)
        test_correct += (preds == labels).sum().item()

print("Test Acc:", test_correct / test_total)

wandb.log({
    "test/acc": test_correct / test_total,
    "test/loss": test_loss / len(test_loader),
})

wandb.finish()


# ============================================================
# PUSH BEST MODEL TO HUGGING FACE
# ============================================================

from huggingface_hub import login, HfApi, upload_file

print("Logging into Hugging Face...")
login()  # secure prompt

HF_REPO_NAME = "arpita2desh/resnet18-cifar10-subset"

api = HfApi()
api.create_repo(repo_id=HF_REPO_NAME, exist_ok=True)

print("Uploading model...")

upload_file(
    path_or_fileobj=BEST_MODEL_PATH,
    path_in_repo="pytorch_model.bin",
    repo_id=HF_REPO_NAME,
)

print("✅ Upload successful!")
