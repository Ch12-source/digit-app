# -*- coding: utf-8 -*-
"""
手写数字识别 - Web端部署
ShuffledFusionNet V4 · 83K · 99.19% · 综合数据增强
"""

import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
import os

# ============================================================
# 预处理（保留像素渐变，匹配 MNIST 分布）
# ============================================================
def preprocess(pil_img):
    gray = pil_img.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # 智能背景色判断
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
    if bg_mean > 100.0:
        arr = 255.0 - arr

    # 仅用阈值定位数字位置（不改变像素值！）
    binary = arr > 120
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not rows.any() or not cols.any():
        binary = arr > 80
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        if not rows.any() or not cols.any():
            return None

    y_indices = np.where(rows)[0]; x_indices = np.where(cols)[0]
    y1, y2 = y_indices[0], y_indices[-1]; x1, x2 = x_indices[0], x_indices[-1]

    # 加 padding，提取 ROI（保留原始渐变！）
    pad_y = max(2, int((y2 - y1) * 0.15))
    pad_x = max(2, int((x2 - x1) * 0.15))
    y1, y2 = max(0, y1 - pad_y), min(h, y2 + pad_y)
    x1, x2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
    # 从原始像素提取 ROI（保留渐变），二值化仅用于定位
    roi = arr[y1:y2, x1:x2]

    # 缩放到 20x20
    rh, rw = roi.shape
    scale = 20.0 / max(rh, rw)
    nh, nw = max(1, int(rh * scale)), max(1, int(rw * scale))
    roi_pil = Image.fromarray(np.clip(roi, 0, 255).astype(np.uint8))
    roi_rs = roi_pil.resize((nw, nh), Image.LANCZOS)

    # 居中到 28x28
    canvas = np.zeros((28, 28), dtype=np.float32)
    ox, oy = (28 - nw) // 2, (28 - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = np.array(roi_rs, dtype=np.float32)

    # MNIST 标准化
    canvas = canvas / 255.0
    canvas = (canvas - 0.1307) / 0.3081
    return torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0)
# ============================================================
# 加载模型
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
st.set_page_config(page_title="手写数字识别", page_icon="✍️", layout="centered")

st.title("✍️ 手写数字识别")
st.caption("ShuffledFusionNet V4 · 83K · 99.19% · 综合数据增强训练")

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
            st.error("❌ 未检测到数字，请确保数字清晰居中、背景简洁")
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
            st.error("❌ 未检测到数字")
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

with st.expander("💡 使用技巧"):
    st.markdown("""
    - **拍照**: 白色纸张 + 黑色笔书写，光线均匀、避免阴影
    - **上传**: 支持 PNG/JPG/BMP，白底黑字效果最佳
    - 数字尽量居中、笔画清晰
    """)

st.markdown("---")
st.caption("ShuffledFusionNet V4 · 99.19% · 蔡磊实践课 · 大数据综合实践")
