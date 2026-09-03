import streamlit as st
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

# Resolve import path ke src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from model_finetune import create_resnet18_finetune
from transforms import val_test_transform, IMAGENET_MEAN, IMAGENET_STD
from gradcam import GradCAM, overlay_heatmap

st.set_page_config(page_title="Demo Klasifikasi", page_icon="🔍", layout="wide")

# === Konfigurasi ===
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "resnet18_finetuned.pth")
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASS_EMOJI = {
    'cardboard': '📦', 'glass': '🥃', 'metal': '🥫',
    'paper': '📄', 'plastic': '🧴', 'trash': '🗑️'
}


@st.cache_resource
def load_model():
    """Load model sekali saja, di-cache oleh Streamlit."""
    device = torch.device("cpu")  # CPU untuk deployment (lebih portable)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    model = create_resnet18_finetune(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    return model, device


def predict(model, image, device):
    """Jalankan prediksi pada satu gambar PIL."""
    img_tensor = val_test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]

    pred_idx = probs.argmax().item()
    return pred_idx, probs.cpu().numpy(), img_tensor


def generate_gradcam(model, img_tensor, pred_idx, original_img_np):
    """Generate Grad-CAM overlay."""
    gradcam = GradCAM(model, target_layer=model.layer4)
    # Perlu enable grad sementara untuk backward pass
    img_tensor.requires_grad_(True)
    cam = gradcam.generate(img_tensor, class_idx=pred_idx)
    overlaid = overlay_heatmap(original_img_np, cam, alpha=0.5)
    return overlaid


# === UI ===
st.title("🔍 Demo Klasifikasi Sampah")
st.markdown("Upload gambar sampah dan lihat prediksi model beserta visualisasi **Grad-CAM**.")

model, device = load_model()

uploaded_file = st.file_uploader(
    "Upload gambar sampah (JPG/PNG)",
    type=["jpg", "jpeg", "png"],
    help="Ukuran maksimal 200MB"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # === Prediksi ===
    pred_idx, probs, img_tensor = predict(model, image, device)
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx] * 100

    st.divider()

    # === Hasil Prediksi ===
    if confidence > 70:
        color = "green"
        badge = "✅ Tinggi"
    elif confidence > 40:
        color = "orange"
        badge = "⚠️ Sedang"
    else:
        color = "red"
        badge = "❌ Rendah"

    st.markdown(
        f"### Prediksi: {CLASS_EMOJI.get(pred_class, '')} "
        f"**:{color}[{pred_class.upper()}]** — "
        f"Confidence: **:{color}[{confidence:.1f}%]** ({badge})"
    )

    # === Gambar + Grad-CAM ===
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Gambar Asli")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔥 Grad-CAM Heatmap")
        with st.spinner("Generating Grad-CAM..."):
            original_np = np.array(image.resize((224, 224)))
            gradcam_img = generate_gradcam(model, img_tensor, pred_idx, original_np)
            st.image(gradcam_img, use_container_width=True)
            st.caption("Area merah = region yang paling berpengaruh terhadap prediksi model.")

    st.divider()

    # === Bar Chart Confidence ===
    st.subheader("📊 Confidence Score Semua Kelas")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#2ecc71' if i == pred_idx else '#bdc3c7' for i in range(len(CLASSES))]
    bars = ax.barh(CLASSES, probs * 100, color=colors, edgecolor='white')
    ax.set_xlabel('Confidence (%)')
    ax.set_xlim(0, 105)

    for bar, prob in zip(bars, probs * 100):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{prob:.1f}%', va='center', fontsize=10)

    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # === Catatan Keterbatasan ===
    with st.expander("⚠️ Catatan Keterbatasan Model"):
        st.markdown("""
        Model ini memiliki beberapa keterbatasan yang perlu diperhatikan:

        1. **Glass ↔ Metal**: Botol kaca dengan permukaan reflektif sering tertukar dengan
           logam karena kedua material memiliki pantulan cahaya yang serupa.

        2. **Cardboard ↔ Paper**: Kardus tipis atau kardus berlabel cetak (printed)
           sering diprediksi sebagai kertas karena tekstur permukaannya mirip.

        3. **Kelas Trash**: Kategori "trash" sangat heterogen dan memiliki sample paling
           sedikit dalam dataset, sehingga recall-nya lebih rendah dari kelas lain.

        4. **Domain**: Model dilatih pada dataset TrashNet (~2,500 gambar) dengan
           background yang relatif bersih. Performa mungkin menurun pada gambar dengan
           background kompleks atau pencahayaan ekstrem.
        """)

else:
    # Placeholder saat belum ada upload
    st.info("👆 Upload gambar sampah untuk memulai klasifikasi.")

    st.markdown("### Kelas yang Didukung")
    cols = st.columns(6)
    for i, cls in enumerate(CLASSES):
        with cols[i]:
            st.markdown(f"#### {CLASS_EMOJI.get(cls, '')} {cls}")
