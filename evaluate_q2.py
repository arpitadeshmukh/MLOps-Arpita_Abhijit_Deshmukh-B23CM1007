import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from huggingface_hub import HfApi, upload_file
from torchvision import models
from datasets import load_dataset

# ============================================================
# CONFIG
# ============================================================
HF_REPO_NAME = "arpita2desh/resnet18-cifar10-subset"
BEST_MODEL_PATH = "best_resnet18_cifar10_subset.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

wandb.init(project="resnet18-cifar10-subset-eval")

# ============================================================
# 1️⃣ PUSH BEST MODEL TO HUGGING FACE
# ============================================================
api = HfApi()
api.create_repo(repo_id=HF_REPO_NAME, exist_ok=True)

upload_file(
    path_or_fileobj=BEST_MODEL_PATH,
    path_in_repo="pytorch_model.bin",
    repo_id=HF_REPO_NAME,
)

print("Model pushed to Hugging Face!")

# ============================================================
# 2️⃣ LOAD MODEL BACK FROM HF FOR EVALUATION
# ============================================================
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 10)

model.load_state_dict(
    torch.hub.load_state_dict_from_url(
        f"https://huggingface.co/{HF_REPO_NAME}/resolve/main/pytorch_model.bin",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

# ============================================================
# LOAD TEST DATA
# ============================================================
dataset = load_dataset("Chiranjeev007/CIFAR-10_Subset")
test_ds = dataset["test"]

from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2470,0.2435,0.2616))
])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        img = transform(self.ds[idx]["img"])
        label = self.ds[idx]["label"]
        return img, label

test_loader = DataLoader(CustomDataset(test_ds),
                         batch_size=64,
                         shuffle=False)

# ============================================================
# 3️⃣ EVALUATION
# ============================================================
all_preds = []
all_labels = []
correct_samples = []
incorrect_samples = []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader):
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

        for img, pred, label in zip(imgs.cpu(), preds.cpu(), labels):
            if len(correct_samples) < 10 and pred == label:
                correct_samples.append((img, pred.item(), label.item()))
            elif len(incorrect_samples) < 10 and pred != label:
                incorrect_samples.append((img, pred.item(), label.item()))

# ============================================================
# 4️⃣ CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

wandb.log({"Confusion Matrix": wandb.Image(plt)})
plt.close()

# ============================================================
# 5️⃣ CLASS-WISE ACCURACY
# ============================================================
class_correct = [0]*10
class_total = [0]*10

for label, pred in zip(all_labels, all_preds):
    class_total[label] += 1
    if label == pred:
        class_correct[label] += 1

class_acc = [class_correct[i]/class_total[i] for i in range(10)]

plt.figure(figsize=(10,5))
plt.bar(CLASS_NAMES, class_acc)
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Class-wise Accuracy")

wandb.log({"Class-wise Accuracy": wandb.Image(plt)})
plt.close()

# Log numeric class-wise accuracy
for i, name in enumerate(CLASS_NAMES):
    wandb.log({f"class_accuracy/{name}": class_acc[i]})

# ============================================================
# 6️⃣ LOG 20 TEST SAMPLES (10 CORRECT + 10 INCORRECT)
# ============================================================
def unnormalize(img):
    mean = torch.tensor([0.4914,0.4822,0.4465]).view(3,1,1)
    std = torch.tensor([0.2470,0.2435,0.2616]).view(3,1,1)
    return img * std + mean

sample_images = []
sample_images.extend(correct_samples)
sample_images.extend(incorrect_samples)

wandb_images = []

for img, pred, label in sample_images:
    img = unnormalize(img).clamp(0,1)
    caption = f"Pred: {CLASS_NAMES[pred]} | Actual: {CLASS_NAMES[label]}"
    wandb_images.append(wandb.Image(img, caption=caption))

wandb.log({"Test Samples (Correct & Incorrect)": wandb_images})

# ============================================================
# 7️⃣ OVERALL ACCURACY
# ============================================================
overall_acc = np.mean(np.array(all_preds) == np.array(all_labels))
wandb.log({"Overall Test Accuracy": overall_acc})

print("Overall Test Accuracy:", overall_acc)

wandb.finish()
