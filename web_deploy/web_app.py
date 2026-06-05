# -*- coding: utf-8 -*-
"""
Handwritten Digit Recognition - Web Deployment
ShuffledFusionNet V4 - 39K - 98.46% - Data Augmentation
Preprocessing: adaptive threshold + morphological opening + stroke dilation + pure bg
"""

import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import os

# ============================================================
# Model: ShuffledFusionNet (V4, 39K)
# ============================================================
def channel_shuffle(x, g):
    b, c, h, w = x.shape
    return x.view(b, g, c // g, h, w).transpose(1, 2).contiguous().view(b, c, h, w)

class AMKF(nn.Module):
    def __init__(self, ic, oc):
        super().__init__(); m = oc // 2
        self.b3 = nn.Sequential(nn.Conv2d(ic, m, 3, 1, 1, bias=False), nn.BatchNorm2d(m), nn.ReLU(True))
        self.b5 = nn.Sequential(nn.Conv2d(ic, m, 5, 1, 2, bias=False), nn.BatchNorm2d(m), nn.ReLU(True))
        self.fw = nn.Parameter(torch.ones(2, 1, 1, 1) * 0.5)
    def forward(self, x):
        w = F.softmax(self.fw, 0)
        return torch.cat([self.b3(x) * w[0] * 2, self.b5(x) * w[1] * 2], 1)

class SGDR(nn.Module):
    def __init__(self, ic, oc, s=1, g=4):
        super().__init__(); self.g = g
        self.dw = nn.Sequential(nn.Conv2d(ic, ic, 3, s, 1, groups=ic // g, bias=False), nn.BatchNorm2d(ic), nn.ReLU(True))
        self.gconv = nn.Sequential(nn.Conv2d(ic, ic, 1, groups=g, bias=False), nn.BatchNorm2d(ic), nn.ReLU(True))
        self.pw = nn.Sequential(nn.Conv2d(ic, oc, 1, bias=False), nn.BatchNorm2d(oc))
        self.sc = nn.Identity() if s == 1 and ic == oc else nn.Sequential(nn.Conv2d(ic, oc, 1, s, bias=False), nn.BatchNorm2d(oc))
    def forward(self, x):
        out = self.dw(x); out = channel_shuffle(out, self.g); out = self.gconv(out)
        return F.relu(self.pw(out) + self.sc(x))

class CSA(nn.Module):
    def __init__(self, c, k=3):
        super().__init__(); self.c1 = nn.Conv1d(1, 1, k, 1, k // 2, bias=False)
    def forward(self, x):
        b, c, h, w = x.shape; g = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        return x * torch.sigmoid(self.c1(g.unsqueeze(1)).squeeze(1).unsqueeze(-1).unsqueeze(-1))

class ShuffledFusionNet(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.amkf = AMKF(1, 32); self.bn0 = nn.BatchNorm2d(32)
        self.sgdr1 = SGDR(32, 64, 2); self.csa1 = CSA(64)
        self.sgdr2 = SGDR(64, 96, 2); self.csa2 = CSA(96)
        self.sgdr3 = SGDR(96, 96, 1); self.csa3 = CSA(96)
        self.cls = nn.Conv2d(96, nc, 1); self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.bn0(self.amkf(x))
        x = self.csa1(self.sgdr1(x))
        x = self.csa2(self.sgdr2(x))
        x = self.csa3(self.sgdr3(x))
        return self.gap(self.cls(x)).squeeze(-1).squeeze(-1)

# ============================================================
# Preprocessing
#   - Local adaptive threshold C=30 (filters bleed-through / grid lines)
#   - Morphological opening: MinFilter(3) erode -> MaxFilter(7) dilate
#   - Pure black bg via mask, ROI from eroded image
#   - Final invert: digit=dark, bg=bright (80% training majority)
# ============================================================
def preprocess(pil_img):
    # Step 0: auto-correct EXIF orientation (phone camera rotation metadata)
    pil_img = ImageOps.exif_transpose(pil_img)

    # Step 1: grayscale + resize to 280x280
    img_small = pil_img.convert("L").resize((280, 280), Image.LANCZOS)
    arr = np.array(img_small, dtype=np.float32)

    # Step 2: background detection
    edge_width = 5
    h, w = arr.shape
    if h > edge_width * 2 and w > edge_width * 2:
        edge_pixels = np.concatenate([
            arr[:edge_width, :].ravel(),
            arr[-edge_width:, :].ravel(),
            arr[edge_width:-edge_width, :edge_width].ravel(),
            arr[edge_width:-edge_width, -edge_width:].ravel()
        ])
        bg_mean = edge_pixels.mean()
    else:
        bg_mean = arr.mean()

    if bg_mean > 127.0:
        arr_inv = 255.0 - arr
        img_inv = Image.fromarray(arr_inv.astype(np.uint8))
    else:
        arr_inv = arr
        img_inv = img_small

    # Step 3: local adaptive threshold (C=30 filters bleed-through text)
    local_bg = img_inv.filter(ImageFilter.BoxBlur(radius=25))
    local_bg_arr = np.array(local_bg, dtype=np.float32)
    C = 30.0  # high threshold: pen ink >> bleed-through text
    binary_mask = arr_inv > (local_bg_arr + C)
    mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))

    # Step 4: morphological opening (erode -> dilate)
    #   4a. MinFilter(3): erase isolated noise speckles < 3px wide
    img_eroded = mask_img.filter(ImageFilter.MinFilter(size=3))
    #   4b. MaxFilter(7): dilate clean digit strokes (7 avoids filling closed loops)
    img_thick = img_eroded.filter(ImageFilter.MaxFilter(size=7))
    #   4c. GaussianBlur: soft anti-aliased edges
    img_smoothed = img_thick.filter(ImageFilter.GaussianBlur(radius=1.0))
    arr_processed = np.array(img_smoothed, dtype=np.float32)

    # Step 5: ROI from eroded image (tight on real digit, not noise)
    eroded_arr = np.array(img_eroded, dtype=np.float32) > 127
    rows = np.any(eroded_arr, axis=1)
    cols = np.any(eroded_arr, axis=0)
    if not rows.any() or not cols.any():
        return None

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y1, y2 = y_indices[0], y_indices[-1]
    x1, x2 = x_indices[0], x_indices[-1]
    pad = 8
    y1, y2 = max(0, y1 - pad), min(arr.shape[0], y2 + pad + 1)
    x1, x2 = max(0, x1 - pad), min(arr.shape[1], x2 + pad + 1)
    roi = arr_processed[y1:y2, x1:x2]

    # Step 6: scale to 20x20
    h_roi, w_roi = roi.shape
    scale = 20.0 / max(h_roi, w_roi)
    nh, nw = max(1, int(h_roi * scale)), max(1, int(w_roi * scale))
    roi_pil = Image.fromarray(roi.astype(np.uint8))
    roi_rs = roi_pil.resize((nw, nh), Image.LANCZOS)

    # Step 7: center on 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.float32)
    ox, oy = (28 - nw) // 2, (28 - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = np.array(roi_rs, dtype=np.float32)

    # Step 8: normalize + polarity fix
    #   Final invert: digit=dark(0), bg=bright(1) -> 80% MNIST training majority
    canvas = canvas / 255.0
    canvas = 1.0 - canvas
    canvas = (canvas - 0.1307) / 0.3081
    return torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0)

# ============================================================
# Model loader
# ============================================================
@st.cache_resource
def load_model():
    m = ShuffledFusionNet()
    path = os.path.join(os.path.dirname(__file__), "best_model.pth")
    state = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="=", layout="centered")

st.title("Handwritten Digit Recognition")
st.caption("ShuffledFusionNet V4 . 39K . 98.46% . Adaptive Preprocessing")

model = load_model()

tab1, tab2 = st.tabs(["Camera", "Upload"])

with tab1:
    st.caption("Take a photo (even lighting, white paper + black pen works best)")
    img_file = st.camera_input("", label_visibility="collapsed")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, width=224)
        tensor = preprocess(image)

        if tensor is None:
            st.error("No digit detected. Make sure the digit is clear and centered.")
        else:
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze().numpy()
                pred = int(np.argmax(probs))
                conf = float(probs[pred])

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"<h1 style='font-size:80px;text-align:center;margin:0;'>{pred}</h1>", unsafe_allow_html=True)
            with col2:
                st.progress(float(conf), text=f"Confidence: {conf*100:.1f}%")
                if conf < 0.7:
                    st.warning("Low confidence, try again with better lighting")

with tab2:
    st.caption("Upload a handwritten digit image (PNG/JPG/BMP)")
    img_file = st.file_uploader("", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, width=224)
        tensor = preprocess(image)

        if tensor is None:
            st.error("No digit detected")
        else:
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze().numpy()
                pred = int(np.argmax(probs))
                conf = float(probs[pred])

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"<h1 style='font-size:80px;text-align:center;margin:0;'>{pred}</h1>", unsafe_allow_html=True)
            with col2:
                st.progress(float(conf), text=f"Confidence: {conf*100:.1f}%")

            top3 = np.argsort(probs)[::-1][:3]
            st.markdown("---")
            st.caption("Top-3 Predictions:")
            cols = st.columns(3)
            for i, idx in enumerate(top3):
                with cols[i]:
                    st.metric(f"#{i+1}", str(idx), f"{probs[idx]*100:.1f}%")

with st.expander("Debug: View Preprocessing"):
    st.caption("Upload an image to see the preprocessed 28x28 output")
    debug_file = st.file_uploader("Debug image", type=["png", "jpg", "jpeg", "bmp"],
                                   key="debug_upload", label_visibility="collapsed")
    if debug_file is not None:
        image = Image.open(debug_file).convert("RGB")
        tensor = preprocess(image)
        if tensor is not None:
            disp = tensor.squeeze().numpy()
            disp = (disp * 0.3081) + 0.1307
            disp = 1.0 - disp
            disp = np.clip(disp, 0, 1)
            st.image(disp, width=280, caption="Preprocessed 28x28")
        else:
            st.error("No digit detected in debug image")

st.markdown("---")
st.caption("ShuffledFusionNet V4 . 39K params . 98.46% test accuracy . Adaptive Preprocessing")

