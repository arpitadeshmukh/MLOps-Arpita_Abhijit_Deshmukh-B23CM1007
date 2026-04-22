import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class CityScapesDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found or corrupted: {img_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found or corrupted: {mask_path}")

        image = cv2.imread(img_path) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 96), interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0

        # mask = cv2.imread(mask_path)    
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)    
        mask = cv2.resize(mask, (128, 96), interpolation=cv2.INTER_NEAREST)

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        return image, mask

transform = T.Compose([
    T.Resize((128, 96)),
    T.ToTensor(),
])


dataset = CityScapesDataset(
    img_dir="CameraRGB/",
    mask_dir="CameraMask/",
    transform=transform
)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

print("Loading Data")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.middle = DoubleConv(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))

        m = self.middle(self.pool2(d2))

        u1 = self.up1(m)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.conv2(u2)

        return self.out(u2)


def compute_iou(pred, target, num_classes=23):
    ious = []
    pred = torch.argmax(pred, dim=1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious)


def compute_dice(pred, target, num_classes=23):
    dices = []
    pred = torch.argmax(pred, dim=1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        total = pred_inds.sum().item() + target_inds.sum().item()
        if total == 0:
            continue
        dices.append((2 * intersection) / total)
    return np.mean(dices)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device ", device)
model = UNet(n_classes=23).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 15

train_losses = []
miou_scores = []
dice_scores = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_iou = 0
    total_dice = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += compute_iou(outputs.detach(), masks)
        total_dice += compute_dice(outputs.detach(), masks)

    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    avg_dice = total_dice / len(train_loader)

    train_losses.append(avg_loss)
    miou_scores.append(avg_iou)
    dice_scores.append(avg_dice)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} mIoU: {avg_iou:.4f} Dice: {avg_dice:.4f}")

plt.figure()
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss.png")

plt.figure()
plt.plot(miou_scores)
plt.title("mIoU")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.savefig("miou.png")

plt.figure()
plt.plot(dice_scores)
plt.title("Dice Score")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.savefig("dice.png")

print("Training complete. Plots saved.")

torch.save(model.state_dict(), "unet_model.pth")
print("Model saved as unet_model.pth")

print("Training complete. Plots saved.")

model.eval()
test_iou = 0
test_dice = 0

with torch.no_grad():
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        outputs = model(imgs)

        test_iou += compute_iou(outputs, masks)
        test_dice += compute_dice(outputs, masks)

test_iou /= len(test_loader)
test_dice /= len(test_loader)

print(f"\nTest mIoU: {test_iou:.4f}")
print(f"Test Dice: {test_dice:.4f}")