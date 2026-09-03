import streamlit as st
import os

st.set_page_config(page_title="Studi Kasus", page_icon="📊", layout="wide")

# Path ke assets
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')


def img_path(filename):
    """Return path jika file exists, else None."""
    path = os.path.join(ASSETS_DIR, filename)
    return path if os.path.exists(path) else None


# ============================================================
# HEADER
# ============================================================
st.title("📊 Studi Kasus: Waste Classifier")
st.markdown("Perjalanan pengembangan model klasifikasi sampah dari **baseline CNN** hingga **ResNet18 fine-tuned** dengan 3 strategi improvement.")

st.divider()

# ============================================================
# SECTION 1: Progression 3 Model
# ============================================================
st.header("1️⃣ Progression Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Baseline CNN", "62.53%", help="Val accuracy")
    st.caption("CNN 3-layer dari scratch, 25.8M params")

with col2:
    st.metric("ResNet18 Frozen", "80.26%", delta="+17.73 pp", help="Test accuracy")
    st.caption("Transfer learning, hanya FC layer dilatih (132.9K params)")

with col3:
    st.metric("ResNet18 Fine-tuned", "90.79%", delta="+10.53 pp", help="Test accuracy")
    st.caption("Unfreeze layer3+4, class weighting, augmentasi agresif")

st.markdown("#### Tabel Perbandingan Lengkap")

st.table({
    "Metrik": ["Val Accuracy (best)", "Test Accuracy", "Trainable Params", "Training Epochs", "Best Epoch"],
    "Baseline CNN": ["62.53%", "—", "25.8M", "10", "10"],
    "ResNet18 Frozen": ["78.36%", "80.26%", "132.9K", "10", "7"],
    "ResNet18 Fine-tuned": ["92.35%", "90.79%", "10.6M", "20", "12"],
})

st.divider()

# ============================================================
# SECTION 2: Diagnosis Masalah
# ============================================================
st.header("2️⃣ Diagnosis Masalah (Sebelum Fine-tuning)")

st.markdown("Evaluasi pada model ResNet18 Frozen mengungkap **2 masalah utama**:")

tab1, tab2 = st.tabs(["🗑️ Masalah: Kelas Trash", "🔄 Masalah: Confusion Pairs"])

with tab1:
    st.subheader("Kelas Trash — Recall Hanya 20%")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Recall", "20%", delta="-80 pp dari ideal", delta_color="inverse")
        st.metric("F1-Score", "0.32")
        st.metric("Support (test)", "20 gambar")

    with col2:
        st.markdown("""
        **Mengapa gagal?**
        - Kelas **trash** adalah kategori paling **minoritas** (96 gambar di train set, vs 416 untuk paper)
        - Kategori ini juga paling **heterogen** — berisi campuran barang yang tidak masuk kelas lain
        - Model cenderung memprediksi trash sebagai kelas lain yang lebih dominan

        **Dampak**: Hanya 4 dari 20 gambar trash yang benar dikenali.
        """)

with tab2:
    st.subheader("Top 5 Confusion Pairs = 60% dari Semua Error")

    st.table({
        "#": ["1", "2", "3", "4", "5"],
        "Actual → Predicted": ["glass → metal", "cardboard → paper", "plastic → glass", "trash → metal", "metal → plastic"],
        "Jumlah Error": [15, 11, 7, 7, 5],
        "% Total Error": ["20.0%", "14.7%", "9.3%", "9.3%", "6.7%"],
        "Penyebab Utama": [
            "Permukaan reflektif serupa",
            "Material serat yang mirip",
            "Objek transparan identik",
            "Kategori heterogen + minoritas",
            "Bentuk silindris & warna overlap"
        ],
    })

    cm_frozen = img_path("confusion_matrix_frozen.png")
    if cm_frozen:
        st.image(cm_frozen, caption="Confusion Matrix — ResNet18 Frozen", use_container_width=True)

st.divider()

# ============================================================
# SECTION 3: 3 Strategi Fine-tuning
# ============================================================
st.header("3️⃣ Strategi Improvement")

st.markdown("Tiga strategi diterapkan secara **bersamaan** untuk mengatasi masalah yang teridentifikasi:")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("⚖️ Class Weighting")
    st.markdown("""
    **Target**: Kelas trash yang minoritas

    Menggunakan **inverse frequency weighting** pada `CrossEntropyLoss`:

    | Kelas | Weight |
    |---|---|
    | paper | 0.71 |
    | glass | 0.84 |
    | plastic | 0.87 |
    | metal | 1.03 |
    | cardboard | 1.04 |
    | **trash** | **3.07** |

    Kelas trash mendapat weight **3x lebih tinggi**, memaksa model memberikan penalti lebih besar saat salah mengklasifikasikan trash.
    """)

with col2:
    st.subheader("🔓 Partial Unfreeze")
    st.markdown("""
    **Target**: Fitur terlalu generic

    Unfreeze **layer3 + layer4** (2 block terakhir ResNet18) dengan **discriminative learning rates**:

    | Layer | LR | Status |
    |---|---|---|
    | layer1-2 | — | ❄️ Frozen |
    | layer3-4 | 1e-4 | 🔥 Unfreeze |
    | FC head | 1e-3 | 🔥 Unfreeze |

    LR backbone 10x lebih kecil dari head untuk mencegah **catastrophic forgetting** — fitur low-level (edges, textures) dipertahankan, fitur high-level diadaptasi.
    """)

with col3:
    st.subheader("🎨 Augmentasi Agresif")
    st.markdown("""
    **Target**: Confusion glass↔metal

    `train_transform_v2` dengan variasi lebih tinggi:

    | Augmentasi | V1 | V2 |
    |---|---|---|
    | Rotation | 15° | **20°** |
    | Brightness | 0.2 | **0.4** |
    | Contrast | 0.2 | **0.4** |
    | Saturation | 0.2 | **0.4** |
    | Hue | 0.1 | **0.15** |
    | AutoContrast | ❌ | **✅ (p=0.3)** |

    `RandomAutocontrast` membantu model membedakan **refleksi vs matte** — kunci untuk glass vs metal.
    """)

st.divider()

# ============================================================
# SECTION 4: Hasil Setelah Fine-tuning
# ============================================================
st.header("4️⃣ Hasil Setelah Fine-tuning")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Test Accuracy", "90.79%", delta="+10.53 pp")
with col2:
    st.metric("Trash Recall", "75.0%", delta="+55 pp")
with col3:
    st.metric("Glass→Metal Error", "6", delta="-60% (dari 15)")

st.markdown("#### F1-Score Per Kelas: Before vs After")

st.table({
    "Kelas": ["cardboard", "glass", "metal", "paper", "plastic", "trash", "Macro Avg"],
    "Frozen (F1)": ["0.88", "0.77", "0.71", "0.88", "0.85", "0.32", "0.74"],
    "Fine-tuned (F1)": ["0.93", "0.89", "0.89", "0.95", "0.90", "0.81", "0.89"],
    "Δ F1": ["+0.05", "+0.12", "+0.18", "+0.07", "+0.05", "+0.49", "+0.16"],
})

st.markdown("#### Confusion Matrix Perbandingan")

cm_frozen = img_path("confusion_matrix_frozen.png")
cm_finetuned = img_path("confusion_matrix_finetuned.png")

col1, col2 = st.columns(2)
with col1:
    if cm_frozen:
        st.image(cm_frozen, caption="Sebelum (Frozen)", use_container_width=True)
    else:
        st.info("Confusion matrix frozen tidak tersedia.")

with col2:
    if cm_finetuned:
        st.image(cm_finetuned, caption="Sesudah (Fine-tuned)", use_container_width=True)
    else:
        st.info("Confusion matrix finetuned tidak tersedia.")

st.divider()

# ============================================================
# SECTION 5: Explainability (Grad-CAM)
# ============================================================
st.header("5️⃣ Explainability: Grad-CAM")

st.markdown("""
**Grad-CAM** (Gradient-weighted Class Activation Mapping) digunakan untuk memvisualisasikan
area gambar yang paling berpengaruh terhadap keputusan model.
""")

gradcam_img = img_path("gradcam_visualization.png")
if gradcam_img:
    st.image(gradcam_img, caption="Grad-CAM: Baris 1 = Prediksi Benar, Baris 2 = Prediksi Salah", use_container_width=True)

tab1, tab2 = st.tabs(["✅ Prediksi Benar", "❌ Prediksi Salah"])

with tab1:
    st.markdown("""
    Pada prediksi benar, model konsisten **fokus ke objek utama**:
    - **Glass**: Fokus ke badan botol kaca (bentuk silindris, material transparan)
    - **Plastic**: Fokus ke badan botol dan label kemasan
    - **Paper**: Fokus ke area tengah kertas yang datar
    - **Cardboard**: Fokus ke teks/label dan tekstur serat kardus

    ✅ **Tidak ada bias background** — model belajar mengenali objek, bukan lingkungan.
    """)

with tab2:
    st.markdown("""
    Pada prediksi salah, pola error terungkap:

    **Glass → Metal** (conf 93-97%):
    - Model **salah fokus ke area reflektif lokal** — pantulan cahaya dan tutup logam
    - Menafsirkan refleksi sebagai material logam, bukan memahami bentuk keseluruhan

    **Cardboard → Paper** (conf 96-99%):
    - Model fokus ke **permukaan datar dan label printed** (misal "CHOBANI")
    - Melihat tekstur halus mirip kertas, bukan struktur bergelombang kardus

    ⚠️ Model sangat **overconfident** pada kesalahannya (93-99% confidence).
    """)

st.divider()

# ============================================================
# SECTION 6: Keterbatasan & Rencana Lanjutan
# ============================================================
st.header("6️⃣ Keterbatasan & Rencana Lanjutan")

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚠️ Keterbatasan")
    st.markdown("""
    1. **Glass ↔ Metal**: Masih 6 error di test set karena kemiripan permukaan reflektif
    2. **Cardboard ↔ Paper**: 5 error — secara material memang serupa
    3. **Kelas Trash heterogen**: Recall 75% masih bisa ditingkatkan dengan lebih banyak data
    4. **Dataset terbatas**: Hanya ~2,500 gambar TrashNet, background relatif bersih
    5. **Overconfidence**: Model sering sangat yakin (>90%) bahkan saat salah prediksi
    """)

with col2:
    st.subheader("🚀 Rencana Lanjutan")
    st.markdown("""
    1. **Mobile Deployment**: Konversi model ke TFLite/ONNX untuk deployment di Android/iOS
    2. **Data Augmentation Lanjutan**: Mixup, CutMix untuk meningkatkan generalisasi
    3. **Model Ensemble**: Kombinasi ResNet18 + EfficientNet untuk voting
    4. **Calibration**: Temperature scaling untuk mengatasi overconfidence
    5. **Real-world Testing**: Uji dengan gambar dari kamera HP di berbagai kondisi pencahayaan
    """)

st.divider()
st.caption("♻️ Waste Classifier — Dibuat dengan PyTorch & Streamlit")
