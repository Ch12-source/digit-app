# -*- coding: utf-8 -*-
"""
ShuffledFusionNet - Web端手写数字识别
浏览器拍照 → 云端推理 → 即时识别
部署: Streamlit Cloud  /  Hugging Face Spaces
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io, os

# ============================================================
# 模型定义 (ShuffledFusionNet, 39K, 99.07%)
# ============================================================
def channel_shuffle(x, g):
    b, c, h, w = x.shape
    return x.view(b, g, c // g, h, w).transpose(1, 2).contiguous().view(b, c, h, w)

class AMKF(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        m = oc // 2
        self.b3 = nn.Sequential(nn.Conv2d(ic, m, 3, 1, 1, bias=False), nn.BatchNorm2d(m), nn.ReLU(True))
        self.b5 = nn.Sequential(nn.Conv2d(ic, m, 5, 1, 2, bias=False), nn.BatchNorm2d(m), nn.ReLU(True))
        self.fw = nn.Parameter(torch.ones(2, 1, 1, 1) * 0.5)
    def forward(self, x):
        w = F.softmax(self.fw, 0)
        return torch.cat([self.b3(x) * w[0] * 2, self.b5(x) * w[1] * 2], 1)

class SGDR(nn.Module):
    def __init__(self, ic, oc, s=1, g=4):
        super().__init__()
        self.g = g
        self.dw = nn.Sequential(nn.Conv2d(ic, ic, 3, s, 1, groups=ic // g, bias=False), nn.BatchNorm2d(ic), nn.ReLU(True))
        self.gconv = nn.Sequential(nn.Conv2d(ic, ic, 1, groups=g, bias=False), nn.BatchNorm2d(ic), nn.ReLU(True))
        self.pw = nn.Sequential(nn.Conv2d(ic, oc, 1, bias=False), nn.BatchNorm2d(oc))
        self.sc = nn.Identity() if s == 1 and ic == oc else nn.Sequential(nn.Conv2d(ic, oc, 1, s, bias=False), nn.BatchNorm2d(oc))
    def forward(self, x):
        out = self.dw(x)
        out = channel_shuffle(out, self.g)
        out = self.gconv(out)
        return F.relu(self.pw(out) + self.sc(x))

class CSA(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.c1 = nn.Conv1d(1, 1, k, 1, k // 2, bias=False)
    def forward(self, x):
        b, c, h, w = x.shape
        g = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        return x * torch.sigmoid(self.c1(g.unsqueeze(1)).squeeze(1).unsqueeze(-1).unsqueeze(-1))

class ShuffledFusionNet(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.amkf = AMKF(1, 32)
        self.bn0 = nn.BatchNorm2d(32)
        self.sgdr1 = SGDR(32, 64, 2)
        self.csa1 = CSA(64)
        self.sgdr2 = SGDR(64, 96, 2)
        self.csa2 = CSA(96)
        self.sgdr3 = SGDR(96, 96, 1)
        self.csa3 = CSA(96)
        self.cls = nn.Conv2d(96, nc, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.bn0(self.amkf(x))
        x = self.csa1(self.sgdr1(x))
        x = self.csa2(self.sgdr2(x))
        x = self.csa3(self.sgdr3(x))
        return self.gap(self.cls(x)).squeeze(-1).squeeze(-1)

# ============================================================
# 预处理 (PIL版本, 无需OpenCV)
# ============================================================
def preprocess_pil(pil_image):
    """PIL Image -> 28x28 MNIST格式 tensor"""
    # 转灰度
    gray = pil_image.convert("L")
    img_np = np.array(gray, dtype=np.float32)

    # 反色 + 阈值 (用numpy替代cv2)
    img_np = 255.0 - img_np  # invert
    img_np = np.where(img_np > 100, 255.0, 0.0)  # threshold

    # 找数字区域（找非零像素的bounding box）
    rows = np.any(img_np > 0, axis=1)
    cols = np.any(img_np > 0, axis=0)
    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 加padding
    p = 5
    h_img, w_img = img_np.shape
    y_min = max(0, y_min - p)
    y_max = min(h_img, y_max + p)
    x_min = max(0, x_min - p)
    x_max = min(w_img, x_max + p)

    roi = img_np[y_min:y_max + 1, x_min:x_max + 1]

    # 保持宽高比缩放, 目标: 20x20区域内
    h, w = roi.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    if new_h < 1 or new_w < 1:
        return None

    roi_pil = Image.fromarray(roi.astype(np.uint8))
    roi_resized = roi_pil.resize((new_w, new_h), Image.LANCZOS)

    # 居中放在28x28画布
    canvas = np.zeros((28, 28), dtype=np.float32)
    ox, oy = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = np.array(roi_resized, dtype=np.float32)

    # 归一化 (MNIST std)
    canvas = canvas / 255.0
    canvas = (canvas - 0.1307) / 0.3081

    tensor = torch.from_numpy(canvas).unsqueeze(0).unsqueeze(0)
    return tensor

# ============================================================
# 加载模型 (缓存)
# ============================================================
@st.cache_resource
def load_model():
    model = ShuffledFusionNet()
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(
    page_title="ShuffledFusionNet - 手写数字识别",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ ShuffledFusionNet")
st.caption("手写数字识别 · 39K参数 · 99.07%准确率 · 浏览器端拍照识别")

# 加载模型
with st.spinner("加载模型中..."):
    model = load_model()
st.success("模型就绪 ✓")

# ---- 输入方式选择 ----
st.subheader("📷 选择输入方式")
tab1, tab2 = st.tabs(["📸 拍照识别", "📁 上传图片"])

with tab1:
    img_file = st.camera_input("对准手写数字拍照")
    source = img_file

with tab2:
    img_file = st.file_uploader("上传手写数字图片", type=["png", "jpg", "jpeg", "bmp"])
    source = img_file

# ---- 推理 ----
if source is not None:
    # 显示原图
    image = Image.open(source).convert("RGB")
    st.image(image, caption="输入图片", width=280)

    with st.spinner("识别中..."):
        tensor = preprocess_pil(image)

        if tensor is None:
            st.warning("⚠️ 未检测到数字区域，请确保数字清晰、背景简洁")
        else:
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze().numpy()
                pred = int(np.argmax(probs))
                conf = float(probs[pred])

            # 显示结果
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"# {pred}")
            with col2:
                st.progress(float(conf), text=f"置信度: {conf*100:.1f}%")
                if conf > 0.7:
                    st.success("✓ 高置信度")
                else:
                    st.warning("⚠ 置信度较低，请重新拍照")

            # Top-3 预测
            top3 = np.argsort(probs)[::-1][:3]
            st.markdown("---")
            st.caption("Top-3 预测:")
            cols = st.columns(3)
            for i, idx in enumerate(top3):
                pct = probs[idx] * 100
                with cols[i]:
                    st.metric(f"#{i+1}", str(idx), f"{pct:.1f}%")

# ---- 页脚 ----
st.markdown("---")
st.caption("ShuffledFusionNet · 蔡磊实践课 · 大数据综合实践")

