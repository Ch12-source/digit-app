# -*- coding: utf-8 -*-
"""
鎵嬪啓鏁板瓧璇嗗埆 - Web绔儴缃?ShuffledFusionNet V4 路 39K 路 98.46% 路 鏁版嵁澧炲己
"""

import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import os

# ============================================================
# 妯″瀷瀹氫箟: ShuffledFusionNet (V4, 39K)
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
# 棰勫鐞嗭紙瀵规瘮搴︽媺浼?+ 鏅鸿兘鑳屾櫙鍒ゆ柇锛?# ============================================================
def preprocess(pil_img):
    gray = pil_img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    # 高斯模糊降噪（减少真实纸张纹理和阴影噪声）
    arr = np.array(gray.filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32)

    # 1. 鏅鸿兘鑳屾櫙鑹插垽鏂?    edge_width = 5
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
    if bg_mean > 100.0:
        arr = 255.0 - arr  # 鐧藉簳 -> 鍙嶈壊涓洪粦搴?
    # 2. 瀵规瘮搴︽媺浼革紙瑙ｅ喅绾稿紶鍋忕伆銆佸厜绾夸笉鍧囷級
    p_low = np.percentile(arr, 5)
    p_high = np.percentile(arr, 95)
    if p_high - p_low > 10:
        arr = (arr - p_low) / (p_high - p_low) * 255.0
    arr = np.clip(arr, 0, 255)

    # 3. 浜屽€煎寲 + 瀹氫綅鏁板瓧
    arr = np.where(arr > 80, 255.0, 0.0)
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    if not rows.any() or not cols.any():
        return None

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    pad = 4
    y1, y2 = max(0, y1 - pad), min(h, y2 + pad + 1)
    x1, x2 = max(0, x1 - pad), min(w, x2 + pad + 1)
    roi = arr[y1:y2, x1:x2]

    # 4. 缂╂斁鍒?0x20锛屽眳涓埌28x28
    rh, rw = roi.shape
    scale = 20.0 / max(rh, rw)
    nh, nw = int(rh * scale), int(rw * scale)
    if nh < 1 or nw < 1: return None
    roi_pil = Image.fromarray(roi.astype(np.uint8))
    roi_rs = roi_pil.resize((nw, nh), Image.LANCZOS)
    canvas = np.zeros((28, 28), dtype=np.float32)
    ox, oy = (28 - nw) // 2, (28 - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = np.array(roi_rs, dtype=np.float32)
    canvas = canvas / 255.0

    # 5. MNIST鏍囧噯鍖?    canvas = (canvas - 0.1307) / 0.3081
    return torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0)

# ============================================================
# 鍔犺浇妯″瀷
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
st.set_page_config(page_title="鎵嬪啓鏁板瓧璇嗗埆", page_icon="鉁嶏笍", layout="centered")

st.title("鉁嶏笍 鎵嬪啓鏁板瓧璇嗗埆")
st.caption("ShuffledFusionNet V4 路 39K 路 98.46% 路 缁煎悎鏁版嵁澧炲己")

model = load_model()

tab1, tab2 = st.tabs(["馃摳 鎷嶇収璇嗗埆", "馃搧 涓婁紶鍥剧墖"])

with tab1:
    st.caption("瀵瑰噯鐧界焊涓婄殑鎵嬪啓鏁板瓧鎷嶇収")
    img_file = st.camera_input("", label_visibility="collapsed")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, width=224)
        tensor = preprocess(image)

        if tensor is None:
            st.error("鏈娴嬪埌鏁板瓧锛岃纭繚鏁板瓧娓呮櫚灞呬腑")
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
                st.progress(float(conf), text=f"缃俊搴? {conf*100:.1f}%")

with tab2:
    st.caption("涓婁紶鎵嬪啓鏁板瓧鍥剧墖锛圥NG/JPG/BMP锛?)
    img_file = st.file_uploader("", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, width=224)
        tensor = preprocess(image)

        if tensor is None:
            st.error("鏈娴嬪埌鏁板瓧")
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
                st.progress(float(conf), text=f"缃俊搴? {conf*100:.1f}%")

st.markdown("---")
st.caption("ShuffledFusionNet V4 路 钄＄瀹炶返璇?路 澶ф暟鎹患鍚堝疄璺?)
