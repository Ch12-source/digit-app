# -*- coding: utf-8 -*-
"""
閹靛鍟撻弫鏉跨摟鐠囧棗鍩?- Web缁旑垶鍎寸純?璺?妞翠焦顥楁０鍕槱閻炲棛澧?
ShuffledFusionNetPlus V4 璺?83K 璺?99.19%
"""

import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import os

# ============================================================
# 濡€崇€风€规矮绠?(閸氬奔绗?
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
# 妫板嫬顦╅悶?# ============================================================
def preprocess(pil_img):
    gray = pil_img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    # 閺呴缚鍏橀懗灞炬珯閼规彃鍨介弬顓ㄧ窗閸欐牞绔熺紓妯哄剼缁辩姴鍨介弬顓犳鎼?姒涙垵绨?
    edge_width = 5
    h, w = arr.shape
    if h > edge_width * 2 and w > edge_width * 2:
        edge_pixels = np.concatenate([
            arr[:edge_width, :],
            arr[:edge_width, :].ravel(),
            arr[-edge_width:, :].ravel(),
            arr[edge_width:-edge_width, :edge_width].ravel(),
            arr[edge_width:-edge_width, -edge_width:].ravel()
        bg_mean = edge_pixels.mean()
    else:
        bg_mean = arr.mean()
    if bg_mean > 127.0:
        arr = 255.0 - arr  # 閻ц棄绨虫鎴濈摟 閳?閸欏秷澹婃稉娲拨鎼存洜娅х€?    # 閸氾箑鍨鎴濈俺閻ц棄鐡ч敍灞肩箽閹镐礁甯弽?
    arr = np.where(arr > 80, 255.0, 0.0)

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
# ============================================================
# 閸旂姾娴囧Ο鈥崇€?
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
st.set_page_config(page_title="閹靛鍟撻弫鏉跨摟鐠囧棗鍩?, page_icon="閴佸稄绗?, layout="centered")

st.title("閴佸稄绗?閹靛鍟撻弫鏉跨摟鐠囧棗鍩?)
st.caption("ShuffledFusionNetPlus V4 璺?99.19% 璺?缂佺厧鎮庨弫鐗堝祦婢х偛宸?)

model = load_model()

tab1, tab2 = st.tabs(["棣冩懗 閹峰秶鍙庣拠鍡楀焼", "棣冩惂 娑撳﹣绱堕崶鍓у"])

with tab1:
    st.caption("鐎电懓鍣惂鐣岀剨娑撳﹦娈戦幍瀣晸閺佹澘鐡ч幏宥囧弾閿涘牆鍘滅痪鍨綆閸栤偓閵嗕焦鏆熺€涙鐪虫稉顓熸櫏閺嬫粍娓舵担绛圭礆")
    img_file = st.camera_input("", label_visibility="collapsed")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, width=224)

        result = preprocess(image)
        debug_vis = None

        if result is None:
            st.error("閴?閺堫亝顥呭ù瀣煂閺佹澘鐡ч崠鍝勭厵閵嗗倽顕涵顔荤箽閿涙瓡n- 閻у€熷缁剧绱?+ 濞ｈ精澹婄粭鏂惧姛閸愭┙n- 閺佹澘鐡ч崡鐘垫暰闂堫澀鑵戞径?30%~70%\n- 閸忓鍤庨崸鍥у瘧閿涘本妫ら弰搴㈡▔闂冩潙濂?)
        elif isinstance(result, tuple):
            tensor, debug_vis = result
        else:
            tensor = result

        if result is not None:
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze().numpy()
                pred = int(np.argmax(probs))
                conf = float(probs[pred])

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"<h1 style='font-size:80px;text-align:center;margin:0;'>{pred}</h1>", unsafe_allow_html=True)
            with col2:
                st.progress(float(conf), text=f"缂冾喕淇婃惔? {conf*100:.1f}%")

            # 鐠嬪啳鐦崣顖濐潒閸?            if st.checkbox("棣冩敵 閺屻儳婀呮０鍕槱閻炲棙鏅ラ弸婊愮礄濡€崇€风€圭偤妾惇瀣煂閻ㄥ嫬娴橀崓蹇ョ礆"):
                if debug_vis:
                    st.image(debug_vis, width=140, caption="28x28 妫板嫬顦╅悶鍡欑波閺?)
                else:
                    vis = (tensor.squeeze().numpy() - tensor.min()) / (tensor.max() - tensor.min() + 1e-8) * 255
                    st.image(Image.fromarray(vis.astype(np.uint8)), width=140, caption="28x28 妫板嫬顦╅悶鍡欑波閺?)

            if conf < 0.85:
                st.warning(f"閳?缂冾喕淇婃惔?{conf*100:.1f}%閿涘苯缂撶拋顕€鍣搁幏宥忕礄绾喕绻氶惂鐣岀剨姒涙垵鐡ч妴浣稿帨缁惧灝娼庨崠鈧妴浣规殶鐎涙鐪虫稉顓ㄧ礆")

with tab2:
    st.caption("娑撳﹣绱堕幍瀣晸閺佹澘鐡ч崶鍓у閿涘牏娅х痪鎼佺拨鐎?PNG/JPG/BMP閿?)
    img_file = st.file_uploader("", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, width=224)

        result = preprocess(image)

        if result is None:
            st.error("閴?閺堫亝顥呭ù瀣煂閺佹澘鐡ч崠鍝勭厵")
        else:
            tensor = result
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze().numpy()
                pred = int(np.argmax(probs))
                conf = float(probs[pred])

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"<h1 style='font-size:80px;text-align:center;margin:0;'>{pred}</h1>", unsafe_allow_html=True)
            with col2:
                st.progress(float(conf), text=f"缂冾喕淇婃惔? {conf*100:.1f}%")

            if st.checkbox("棣冩敵 閺屻儳婀呮０鍕槱閻炲棙鏅ラ弸?):
                vis = (tensor.squeeze().numpy() - tensor.min()) / (tensor.max() - tensor.min() + 1e-8) * 255
                st.image(Image.fromarray(vis.astype(np.uint8)), width=140, caption="28x28 妫板嫬顦╅悶鍡欑波閺?)

            top3 = np.argsort(probs)[::-1][:3]
            st.markdown("---")
            st.caption("Top-3 妫板嫭绁?")
            cols = st.columns(3)
            for i, idx in enumerate(top3):
                with cols[i]:
                    st.metric(f"#{i+1}", str(idx), f"{probs[idx]*100:.1f}%")

with st.expander("棣冩寱 閹峰秶鍙庨幎鈧褝绱欓柌宥堫洣閿涗緤绱?):
    st.markdown("""
    **閴?閻炲棙鍏傞弶鈥叉閿?*
    - 棣冩惈 缁绢垳娅х痪绋跨炊閵嗕焦妫ら弽鑲╁殠閵嗕焦妫ら崗鏈电铂閺傚洤鐡?
    - 棣冩瀳閿?姒涙垼澹婇幋鏍ㄧ箒閼硅尙鐟稊锕€鍟撻敍宀€鐟潻瑙勭閺?    - 棣冩寱 閸忓鍤庨崸鍥у瘧閿涘矂浼╅崗宥夋Ь瑜板崬鎷伴崣宥呭帨
    - 棣冨箚 閺佹澘鐡ч崡鐘垫暰闂?**30%~70%**閿涘苯鐪虫稉?
    **閴?闁灝鍘ら敍?*
    - 棣冩懌 鐎电懓鐫嗛獮鏇熷閻撗嶇礄娴溠呮晸閹解晛鐨电痪鐧哥礆
    - 棣冨 閸楀﹨绔熸禍顔煎磹鏉堣娈惃鍕帨缁?    - 棣冩惢 閸婄偓鏋╃憴鎺戝鏉╁洤銇?
    - 棣冩憫 閺嶇厧鐡欑痪鍛婂灗閺堝绨崇痪鍦畱缁剧绱?
    """)

st.markdown("---")
st.caption("ShuffledFusionNetPlus V4 璺?閽勶紕顥忕€圭偠杩旂拠?璺?婢堆勬殶閹诡喚鎮ｉ崥鍫濈杽鐠?)
