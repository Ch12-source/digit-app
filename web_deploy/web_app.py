# -*- coding: utf-8 -*-
"""
闁归潧顑呴崯鎾诲极閺夎法鎽熼悹鍥ф閸?- Web缂佹棏鍨堕崕瀵哥磾?鐠?濡炵繝鐒﹂ˉ妤侊紣閸曨偒妲遍柣鐐叉婢?
ShuffledFusionNetPlus V4 鐠?83K 鐠?99.19%
"""

import streamlit as st
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import os

# ============================================================
# 婵☆垪鈧磭鈧鈧鐭粻?(闁告艾濂旂粭?
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
# 濡澘瀚ˇ鈺呮偠?# ============================================================
def preprocess(pil_img):
    gray = pil_img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    # 闁哄懘缂氶崗姗€鎳楃仦鐐彲闁艰褰冮崹浠嬪棘椤撱劎绐楅柛娆愮墳缁旂喓绱撳Ο鍝勫壖缂佽京濮撮崹浠嬪棘椤撶姵顏ら幖?濮掓稒鍨电花?
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
        arr = 255.0 - arr  # 闁谎嗘缁ㄨ櫕顪€閹存繄鎽?闁?闁告瑥绉锋竟濠冪▔濞差亞鎷ㄩ幖瀛樻礈濞呇呪偓?    # 闁告熬绠戦崹顖涱渶閹存繄淇洪柣褑妫勯悺褔鏁嶇仦鑲╃闁归晲绀佺敮顐﹀冀?
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
# 闁告梻濮惧ù鍥熼垾宕団偓?
# ============================================================
@st.cache_resource
def load_model():
    m = ShuffledFusionNetPlus()
    path = os.path.join(os.path.dirname(__file__), "best_model.pth")
    state = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="闁归潧顑呴崯鎾诲极閺夎法鎽熼悹鍥ф閸?, page_icon="闁翠礁绋勭粭?, layout="centered")

st.title("闁翠礁绋勭粭?闁归潧顑呴崯鎾诲极閺夎法鎽熼悹鍥ф閸?)
st.caption("ShuffledFusionNetPlus V4 鐠?99.19% 鐠?缂備胶鍘ч幃搴ㄥ极閻楀牆绁﹀褏鍋涘?)

model = load_model()

tab1, tab2 = st.tabs(["妫ｅ啯鎳?闁瑰嘲绉堕崣搴ｆ嫚閸℃鐒?, "妫ｅ啯鎯?濞戞挸锕ｇ槐鍫曞炊閸撗冾暬"])

with tab1:
    st.caption("閻庣數鎳撻崳顖炴儌閻ｅ瞼鍓ㄥ☉鎾筹功濞堟垿骞嶇€ｎ亜鏅搁柡浣规緲閻⊙囧箯瀹ュ洤寮鹃柨娑樼墕閸樻粎鐥崹顔界秵闁告牑鍋撻柕鍡曠劍閺嗙喓鈧稒顨呴惇铏▔椤撶喐娅忛柡瀣矋濞撹埖鎷呯粵鍦")
    img_file = st.camera_input("", label_visibility="collapsed")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, width=224)

        result = preprocess(image)
        debug_vis = None

        if result is None:
            st.error("闁?闁哄牜浜濋ˉ鍛圭€ｎ亜鐓傞柡浣规緲閻⊙囧礌閸濆嫮鍘甸柕鍡楀€介顒傛兜椤旇崵绠介柨娑欑摗n- 闁谎冣偓鐔奉棌缂佸墽顭堢槐?+ 婵烇綀绮炬竟濠勭箔閺傛儳濮涢柛鎰敊n- 闁轰焦婢橀悺褔宕￠悩鍨毎闂傚牜婢€閼垫垶寰?30%~70%\n- 闁稿繐顦遍崵搴ㄥ锤閸パ冪槯闁挎稑鏈Λ銈夊及鎼淬垺鈻旈梻鍐╂綑婵?)
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
                st.progress(float(conf), text=f"缂傚喚鍠曟穱濠冩償? {conf*100:.1f}%")

            # 閻犲鍟抽惁顖炲矗椤栨繍娼掗柛?            if st.checkbox("妫ｅ啯鏁?闁哄被鍎冲﹢鍛紣閸曨偒妲遍柣鐐叉閺呫儵寮稿鎰婵☆垪鈧磭鈧鈧湱鍋ゅ顖炴儑鐎ｎ亜鐓傞柣銊ュ濞存﹢宕撹箛銉х"):
                if debug_vis:
                    st.image(debug_vis, width=140, caption="28x28 濡澘瀚ˇ鈺呮偠閸℃瑧娉㈤柡?)
                else:
                    vis = (tensor.squeeze().numpy() - tensor.min()) / (tensor.max() - tensor.min() + 1e-8) * 255
                    st.image(Image.fromarray(vis.astype(np.uint8)), width=140, caption="28x28 濡澘瀚ˇ鈺呮偠閸℃瑧娉㈤柡?)

            if conf < 0.85:
                st.warning(f"闁?缂傚喚鍠曟穱濠冩償?{conf*100:.1f}%闁挎稑鑻紓鎾舵媼椤曗偓閸ｆ悂骞忓蹇曠缁绢収鍠曠换姘舵儌閻ｅ瞼鍓ㄥ娑欏灥閻⊙囧Υ娴ｇ甯ㄧ紒鎯х仢濞煎酣宕犻埀顒勫Υ娴ｈ娈堕悗娑欘殔閻櫕绋夐銊х")

with tab2:
    st.caption("濞戞挸锕ｇ槐鍫曞箥鐎ｎ亜鏅搁柡浣规緲閻⊙囧炊閸撗冾暬闁挎稑鐗忓▍褏鐥幖浣烘嫧閻?PNG/JPG/BMP闁?)
    img_file = st.file_uploader("", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, width=224)

        result = preprocess(image)

        if result is None:
            st.error("闁?闁哄牜浜濋ˉ鍛圭€ｎ亜鐓傞柡浣规緲閻⊙囧礌閸濆嫮鍘?)
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
                st.progress(float(conf), text=f"缂傚喚鍠曟穱濠冩償? {conf*100:.1f}%")

            if st.checkbox("妫ｅ啯鏁?闁哄被鍎冲﹢鍛紣閸曨偒妲遍柣鐐叉閺呫儵寮?):
                vis = (tensor.squeeze().numpy() - tensor.min()) / (tensor.max() - tensor.min() + 1e-8) * 255
                st.image(Image.fromarray(vis.astype(np.uint8)), width=140, caption="28x28 濡澘瀚ˇ鈺呮偠閸℃瑧娉㈤柡?)

            top3 = np.argsort(probs)[::-1][:3]
            st.markdown("---")
            st.caption("Top-3 濡澘瀚粊?")
            cols = st.columns(3)
            for i, idx in enumerate(top3):
                with cols[i]:
                    st.metric(f"#{i+1}", str(idx), f"{probs[idx]*100:.1f}%")

with st.expander("妫ｅ啯瀵?闁瑰嘲绉堕崣搴ㄥ箮閳ь剙顔忚缁辨瑩鏌屽鍫矗闁挎稐绶ょ槐?):
    st.markdown("""
    **闁?闁荤偛妫欓崗鍌炲级閳ュ弶顐介柨?*
    - 妫ｅ啯鎯?缂佺虎鍨冲▍褏鐥粙璺ㄧ倞闁靛棔鐒﹀Λ銈夊冀閼测晛娈犻柕鍡曠劍濡倝宕楅張鐢甸搨闁哄倸娲ら悺?
    - 妫ｅ啯鐎抽柨?濮掓稒鍨兼竟濠囧箣閺嶃劎绠掗柤纭呭皺閻燁亝绋婇敃鈧崯鎾绘晬瀹€鈧悷顏呮交鐟欏嫮顏搁柡?    - 妫ｅ啯瀵?闁稿繐顦遍崵搴ㄥ锤閸パ冪槯闁挎稑鐭傛导鈺呭礂瀹ュ鞋鐟滄澘宕幏浼村矗瀹ュ懎甯?
    - 妫ｅ啫绠?闁轰焦婢橀悺褔宕￠悩鍨毎闂?**30%~70%**闁挎稑鑻惇铏▔?
    **闁?闂侇剙鐏濋崢銈夋晬?*
    - 妫ｅ啯鎳?閻庣數鎳撻惈鍡涚嵁閺囩喎顎為柣鎾楀秶绀勫ù婧犲懏鏅搁柟瑙ｆ櫅閻ㄧ數鐥惂鍝ョ
    - 妫ｅ啫顕?闁告锕ㄧ粩鐔哥椤旂厧纾归弶鍫ｎ潐濞堫偊鎯冮崟顐㈠辅缂?    - 妫ｅ啯鎯?闁稿﹦鍋撻弸鈺冩喆閹烘垵顔婇弶鈺佹搐閵?
    - 妫ｅ啯鎲?闁哄秶鍘ч悺娆戠棯閸涘﹤鐏楅柡鍫濐槸缁ㄥ磭鐥崷顓熺暠缂佸墽顭堢槐?
    """)

st.markdown("---")
st.caption("ShuffledFusionNetPlus V4 鐠?闁藉嫸绱曢ˉ蹇曗偓鍦仩鏉╂梻鎷?鐠?濠㈠爢鍕闁硅鍠氶幃锝夊触閸繄鏉介悹?)
