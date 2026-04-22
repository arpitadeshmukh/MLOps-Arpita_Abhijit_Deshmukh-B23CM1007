import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image

# ---------------- MODEL ---------------- #
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

@st.cache_resource
def load_model():
    model = UNet(n_classes=23)
    model.load_state_dict(torch.load("unet_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

def preprocess(image):
    image = cv2.resize(image, (128, 96))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image).unsqueeze(0)

def predict(image):
    tensor = preprocess(image)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).squeeze().numpy()
    return pred

st.title("🚗 Cityscapes Segmentation App")

page = st.sidebar.selectbox("Choose Page", ["Training Metrics", "Inference"])

if page == "Training Metrics":
    st.header("📊 Training Performance")

    st.subheader("Loss Curve")
    st.image("loss.png")

    st.subheader("mIoU Curve")
    st.image("miou.png")

    st.subheader("Dice Score Curve")
    st.image("dice.png")

    st.subheader("Test Metrics")
    st.write("mIoU: **(replace with your value)**")
    st.write("Dice: **(replace with your value)**")

# ---------------- PAGE 2 ---------------- #
elif page == "Inference":
    st.header("🧠 Segmentation Results")

    uploaded_files = st.file_uploader(
        "Upload 4 test images",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_files and len(uploaded_files) == 4:
        for file in uploaded_files:
            image = np.array(Image.open(file).convert("RGB"))

            pred_mask = predict(image)

            # Try to find GT mask (optional if naming matches)
            mask_path = os.path.join("CameraMask", file.name)
            gt_mask = None
            if os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            st.subheader(file.name)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(image, caption="Input Image")

            with col2:
                if gt_mask is not None:
                    st.image(gt_mask, caption="Ground Truth")
                else:
                    st.write("GT not found")

            with col3:
                st.image(pred_mask, caption="Prediction")

    else:
        st.info("Please upload exactly 4 images.")