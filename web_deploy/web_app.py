# -*- coding: utf-8 -*-
"""
手写数字识别 - Web端部署
ShuffledFusionNetPlus V4 · 83K参数 · 99.19%准确率
部署: Streamlit Cloud
"""

import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
import os

# ============================================================
# 模型定义: ShuffledFusionNetPlus (V4)
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

class ShuffledFusionNetPlus(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.amkf = AMKF(1, 32); self.bn0 = nn.BatchNorm2d(32)
        self.sgdr1 = SGDR(32, 64, 2); self.csa1 = CSA(64)
        self.sgdr2 = SGDR(64, 96, 2); self.csa2 = CSA(96)
        self.sgdr3 = SGDR(96, 96, 1); self.csa3 = CSA(96)
        self.sgdr4 = SGDR(96, 128, 1); self.csa4 = CSA(128)
        self.cross_skip = nn.Sequential(nn.Conv2d(96, 128, 1, bias=False), nn.BatchNorm2d(128))
        self.cls = nn.Conv2d(128, nc, 1); self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.bn0(self.amkf(x))
        x = self.csa1(self.sgdr1(x))
        x = self.csa2(self.sgdr2(x))
        skip = self.cross_skip(x)
        x = self.csa3(self.sgdr3(x))
        x = self.sgdr4(x) + skip
        x = self.csa4(x)
        return self.gap(self.cls(x)).squeeze(-1).squeeze(-1)

# ============================================================
# 预处理
# ============================================================
def preprocess(pil_img):
    gray = pil_img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    arr = 255.0 - arr                      # 反色
    arr = np.where(arr > 80, 255.0, 0.0)   # 二值化

    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    if not rows.any() or not cols.any():
        return None

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    pad = 4
    y1, y2 = max(0, y1 - pad), min(arr.shape[0], y2 + pad + 1)
    x1, x2 = max(0, x1 - pad), min(arr.shape[1], x2 + pad + 1)
    roi = arr[y1:y2, x1:x2]

    h, w = roi.shape
    scale = 20.0 / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    if nh < 1 or nw < 1: return None

    roi_pil = Image.fromarray(roi.astype(np.uint8))
    roi_rs = roi_pil.resize((nw, nh), Image.LANCZOS)

    canvas = np.zeros((28, 28), dtype=np.float32)
    ox, oy = (28 - nw) // 2, (28 - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = np.array(roi_rs, dtype=np.float32)

    canvas = canvas / 255.0
    canvas = (canvas - 0.1307) / 0.3081
    return torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0)

# ============================================================
# 加载模型
# ============================================================
@st.cache_resource
def load_model():
    m = ShuffledFusionNetPlus()
    path = os.path.join(os.path.dirname(__file__), "best_model.pth")
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="手写数字识别", page_icon="✍️", layout="centered")

st.title("✍️ 手写数字识别")
st.caption("ShuffledFusionNetPlus V4 · 83K · 99.19% 准确率 · 综合数据增强训练")

model = load_model()

tab1, tab2 = st.tabs(["📸 拍照识别", "📁 上传图片"])

with tab1:
    st.caption("对准白纸上的手写数字拍照（光线均匀效果最佳）")
    img_file = st.camera_input("", label_visibility="collapsed")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, width=224)
        tensor = preprocess(image)

        if tensor is None:
            st.warning("⚠️ 未检测到数字，请确保数字清晰居中")
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
                st.progress(float(conf), text=f"置信度: {conf*100:.1f}%")
                if conf < 0.7:
                    st.warning("⚠ 置信度偏低，建议重拍")

with tab2:
    st.caption("上传手写数字图片（白底黑字，PNG/JPG/BMP）")
    img_file = st.file_uploader("", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, width=224)
        tensor = preprocess(image)

        if tensor is None:
            st.warning("⚠️ 未检测到数字")
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
                st.progress(float(conf), text=f"置信度: {conf*100:.1f}%")

            top3 = np.argsort(probs)[::-1][:3]
            st.markdown("---")
            st.caption("Top-3 预测:")
            cols = st.columns(3)
            for i, idx in enumerate(top3):
                with cols[i]:
                    st.metric(f"#{i+1}", str(idx), f"{probs[idx]*100:.1f}%")

# ---- 说明 ----
with st.expander("💡 使用技巧"):
    st.markdown("""
    - **拍照**: 白色纸张 + 黑色笔书写，光线均匀、避免阴影
    - **上传**: 支持 PNG/JPG/BMP，白底黑字效果最佳
    - 数字尽量居中、笔画清晰，占画面 60% 以上
    """)

st.markdown("---")
st.caption("ShuffledFusionNetPlus V4 · 蔡磊实践课 · 大数据综合实践")
