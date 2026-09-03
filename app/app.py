import streamlit as st

st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻️",
    layout="wide",
)

# === Header ===
st.title("♻️ Waste Classifier")
st.markdown("Klasifikasi jenis sampah otomatis menggunakan **PyTorch** dan **Transfer Learning** (ResNet18).")

st.divider()

# === Metric Cards ===
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Test Accuracy", value="90.79%", delta="+28.26 pp dari baseline")

with col2:
    st.metric(label="Jumlah Kelas", value="6 Kelas")

with col3:
    st.metric(label="Model", value="ResNet18", delta="Fine-tuned")

st.divider()

# === Deskripsi Project ===
st.header("Tentang Project")

st.markdown("""
Project ini mengklasifikasikan gambar sampah ke dalam **6 kategori**:
**cardboard**, **glass**, **metal**, **paper**, **plastic**, dan **trash**.

Dataset yang digunakan adalah **TrashNet** (~2,500 gambar).
Model dikembangkan melalui **3 iterasi** peningkatan:

1. 🧱 **Baseline CNN** dari scratch → 62.53% val accuracy
2. 🧊 **ResNet18 Frozen** (transfer learning) → 78.36% val / 80.26% test
3. 🔥 **ResNet18 Fine-tuned** (unfreeze + class weighting) → 92.35% val / **90.79% test**
""")

# === Navigasi ===
st.divider()
st.subheader("📌 Halaman Aplikasi")

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/1_🔍_Demo.py", label="🔍 Demo Klasifikasi", icon="🔍")
    st.caption("Upload gambar sampah dan lihat prediksi model + Grad-CAM heatmap.")

with col2:
    st.page_link("pages/2_📊_Studi_Kasus.py", label="📊 Studi Kasus", icon="📊")
    st.caption("Perjalanan pengembangan model dari baseline hingga fine-tuning.")

# === Footer ===
st.divider()
st.markdown(
    "Dibuat dengan ❤️ menggunakan PyTorch & Streamlit | "
    "[GitHub Repo](https://github.com/kal-ngga/waste-classifier)",
    unsafe_allow_html=True
)
