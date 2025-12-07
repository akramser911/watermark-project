import os
import io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ----------------------------
# LOCAL CONFIG
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "trained-model.pth")

IMG_SIZE = 224
WM_SIZE  = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="MRI Watermark Demo", layout="wide")

# ----------------------------
# MODEL (must match training)
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, k, s, p)]
        if use_bn:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(2, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, mri, wm_small, alpha=0.03):
        # Kaggle training used bilinear upsample for embedding path
        wm_up = F.interpolate(wm_small, size=mri.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([mri, wm_up], dim=1)
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x)
        r = torch.tanh(self.out(x))
        wm_img = torch.clamp(mri + alpha * r, 0.0, 1.0)
        return wm_img

class DecoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 64)
        self.conv5 = ConvBlock(64, 32)
        self.attention = nn.Sequential(nn.Conv2d(32, 32, 1), nn.Sigmoid())
        self.head = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, wm_img):
        x = self.conv1(wm_img); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x); x = self.conv5(x)
        x = x * self.attention(x)
        out = torch.sigmoid(self.head(x))
        out_small = F.interpolate(out, size=(WM_SIZE, WM_SIZE), mode="bilinear", align_corners=False)
        return out_small

@st.cache_resource
def load_model():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=device)

    enc = EncoderNet().to(device).eval()
    dec = DecoderNet().to(device).eval()

    if isinstance(ckpt, dict) and "encoder" in ckpt and "decoder" in ckpt:
        enc.load_state_dict(ckpt["encoder"])
        dec.load_state_dict(ckpt["decoder"])
        alpha = float(ckpt.get("cfg", {}).get("ALPHA", 0.03))
    else:
        raise ValueError("Bad checkpoint format. Expected dict with keys: encoder, decoder, cfg.")

    return enc, dec, alpha

# ----------------------------
# PREPROCESSING (match Kaggle)
# ----------------------------
def pil_to_tensor_gray_01(im: Image.Image, size: int) -> torch.Tensor:
    """
    Kaggle-style:
      - grayscale
      - resize
      - scale to [0,1]
      - shape (1,1,H,W)
    """
    im = im.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(im).astype(np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None, ...]
    return t

def tensor_to_pil_gray_01(t: torch.Tensor) -> Image.Image:
    arr = t.detach().cpu().clamp(0, 1)[0, 0].numpy()
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")

def compute_psnr_ssim(a_t, b_t):
    a = a_t.detach().cpu().numpy()[0, 0]
    b = b_t.detach().cpu().numpy()[0, 0]
    return float(psnr(a, b, data_range=1.0)), float(ssim(a, b, data_range=1.0))

def watermark_accuracy(wm_true_small, wm_pred_small, thr=0.5):
    gt = (wm_true_small > thr).float()
    pr = (wm_pred_small > thr).float()
    return float((gt == pr).float().mean().item())

# UI
st.title("🧠 MRI Watermarking")

enc, dec, alpha_ckpt = load_model()

with st.sidebar:
    st.header("Upload")
    mri_file  = st.file_uploader("Upload MRI image", type=["png","jpg","jpeg","bmp","webp"])
    logo_file = st.file_uploader("Upload logo (watermark)", type=["png","jpg","jpeg","bmp","webp"])

    st.header("Settings")
    alpha_used = st.slider("alpha (embedding strength)", 0.001, 0.08, float(alpha_ckpt), 0.001)
    thr = st.slider("WM threshold (accuracy)", 0.0, 1.0, 0.5, 0.01)

    run_btn = st.button("Run Embed + Recover", type="primary")

st.caption(f"Loaded: `{CKPT_PATH}` | device=`{device}` | alpha=`{alpha_used}`")

if run_btn:
    if mri_file is None or logo_file is None:
        st.error("Upload BOTH an MRI image and a logo image.")
        st.stop()

    mri_pil  = Image.open(mri_file)
    logo_pil = Image.open(logo_file)

    # Kaggle-matched tensors
    mri_t = pil_to_tensor_gray_01(mri_pil, IMG_SIZE).to(device)
    wm_t  = pil_to_tensor_gray_01(logo_pil, WM_SIZE).to(device)

    with torch.no_grad():
        wm_img_t = enc(mri_t, wm_t, alpha=alpha_used)
        wm_hat_t = dec(wm_img_t)

    # Display upscaled watermark like Kaggle visualizations
    wm_gt_up  = F.interpolate(wm_t,     size=(IMG_SIZE, IMG_SIZE), mode="nearest")
    wm_hat_up = F.interpolate(wm_hat_t, size=(IMG_SIZE, IMG_SIZE), mode="nearest")

    # Metrics (image fidelity)
    p, s = compute_psnr_ssim(mri_t, wm_img_t)

    # Metric (watermark recovery)
    wm_acc = watermark_accuracy(wm_t, wm_hat_t, thr=thr)

    # Convert for display
    mri_out    = tensor_to_pil_gray_01(mri_t)
    wm_img_out = tensor_to_pil_gray_01(wm_img_t)
    wm_gt_out  = tensor_to_pil_gray_01(wm_gt_up)
    wm_hat_out = tensor_to_pil_gray_01(wm_hat_up)

    # Layout
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("MRI")
        st.image([mri_out, wm_img_out], caption=["Original MRI", "Watermarked MRI"], width=360)
        st.write(f"**PSNR:** {p:.2f} dB | **SSIM:** {s:.3f}")

    with col2:
        st.subheader("Watermark")
        st.image([wm_gt_out, wm_hat_out], caption=["Watermark (GT, upscaled)", "Recovered (upscaled)"], width=360)
        st.write(f"**WM Recovery Accuracy:** {wm_acc*100:.2f}% (thr={thr:.2f})")

    st.divider()

    # Downloads
    b1 = io.BytesIO(); wm_img_out.save(b1, format="PNG")
    b2 = io.BytesIO(); wm_hat_out.save(b2, format="PNG")
    st.download_button("Download Watermarked MRI (PNG)", b1.getvalue(), "watermarked_mri.png", "image/png")
    st.download_button("Download Recovered Watermark (PNG)", b2.getvalue(), "recovered_watermark.png", "image/png")
