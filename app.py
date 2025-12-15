import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="sEMG Biometric ID",
    page_icon="🖐️",
    layout="wide"
)

st.title("🖐️ Sistem Identifikasi Biometrik sEMG")
st.markdown("Identifikasi subjek berdasarkan pola sinyal otot menggunakan **MiniRocket + Ridge Classifier**.")

# --- FUNGSI LOAD ASSETS (CACHED) ---
@st.cache_resource
def load_resources():
    # 1. Load Model Utama
    model_path = "models/best_model_minirocket_84acc.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model tidak ditemukan di {model_path}")
        return None, None
    pipeline = joblib.load(model_path)
    
    # 2. Load Scaler (PENTING untuk data upload)
    scaler_path = "models/scaler.pkl"
    if not os.path.exists(scaler_path):
        st.error(f"❌ Scaler tidak ditemukan di {scaler_path}. Harap jalankan 'joblib.dump(scaler, ...)' di notebook.")
        return None, None
    scaler = joblib.load(scaler_path)
    
    return pipeline, scaler

@st.cache_data
def load_test_data():
    # Load data simulasi (NPY files)
    try:
        X_test = np.load('processed_data/X_test_final.npy')
        y_test = np.load('processed_data/y_test_final.npy')
        return X_test, y_test
    except FileNotFoundError:
        return None, None

# --- INISIALISASI ---
pipeline, scaler = load_resources()
X_test_sim, y_test_sim = load_test_data()

# Mapping Label
label_map = {0: 'Subjek 1 (Wanita)', 1: 'Subjek 2 (Wanita)', 2: 'Subjek 3 (Wanita)', 
             3: 'Subjek 4 (Pria)', 4: 'Subjek 5 (Pria)'}

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["🎲 Simulasi Acak", "📂 Upload Data Baru"])

# ==============================================================================
# TAB 1: SIMULASI ACAK (Fitur Lama)
# ==============================================================================
with tab1:
    st.info("Mengambil sampel acak dari Dataset Uji (X_test) yang sudah ada.")
    
    col_btn, col_res = st.columns([1, 4])
    
    with col_btn:
        if st.button("Ambil Sampel Acak", type="primary"):
            if X_test_sim is None:
                st.error("Data test npy tidak ditemukan.")
            else:
                # Ambil Index Acak
                idx = np.random.randint(0, len(X_test_sim))
                signal_sim = X_test_sim[idx]
                true_label_idx = y_test_sim[idx]
                
                # Simpan ke Session State Tab 1
                st.session_state['sim_signal'] = signal_sim
                st.session_state['sim_true_label'] = label_map[true_label_idx]
                st.session_state['sim_idx'] = idx

    with col_res:
        if 'sim_signal' in st.session_state:
            sig = st.session_state['sim_signal']
            lbl = st.session_state['sim_true_label']
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(sig, color='#1f77b4')
            ax.set_title(f"Sinyal Sampel #{st.session_state['sim_idx']}")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Prediksi
            input_data = sig.reshape(1, 1, 1500) # Data simulasi sudah discale, langsung reshape
            pred_idx = pipeline.predict(input_data)[0]
            pred_str = label_map[pred_idx]
            
            # Hasil
            c1, c2 = st.columns(2)
            c1.metric("Prediksi AI", pred_str)
            c2.metric("Label Asli", lbl)
            
            if pred_str == lbl:
                st.success("✅ Identifikasi BENAR!")
            else:
                st.error("❌ Identifikasi SALAH.")

# ==============================================================================
# TAB 2: UPLOAD DATA MANUAL (Fitur Baru)
# ==============================================================================
with tab2:
    st.info("Upload file CSV/TXT berisi deretan angka sinyal (harus panjang 1500 titik).")
    
    uploaded_file = st.file_uploader("Upload File Sinyal (.csv / .txt)", type=['csv', 'txt'])
    
    if uploaded_file is not None and pipeline is not None and scaler is not None:
        try:
            # 1. Baca File
            # Asumsi: File berisi angka dipisah koma dalam satu baris, atau satu angka per baris
            df_upload = pd.read_csv(uploaded_file, header=None)
            
            # Flatten menjadi 1 array panjang
            raw_signal = df_upload.values.flatten()
            
            # 2. Validasi Panjang Data
            if len(raw_signal) != 1500:
                st.error(f"❌ Error: Panjang sinyal harus tepat 1500 titik. Data kamu: {len(raw_signal)} titik.")
            else:
                st.success("✅ File valid! Sinyal berhasil dimuat.")
                
                # 3. Visualisasi Data RAW (Sebelum Scaling)
                st.subheader("Visualisasi Sinyal Input")
                fig_up, ax_up = plt.subplots(figsize=(10, 3))
                ax_up.plot(raw_signal, color='orange')
                ax_up.set_title("Sinyal Input (Raw Data)")
                ax_up.set_ylabel("Amplitudo")
                st.pyplot(fig_up)
                
                # 4. PREPROCESSING (KRUSIAL!!)
                # Kita harus men-scale data raw ini menggunakan scaler yang sama dengan training
                
                # Reshape ke (1, 1500) karena scaler butuh 2D array
                raw_signal_2d = raw_signal.reshape(1, -1)
                
                # Lakukan Scaling
                scaled_signal = scaler.transform(raw_signal_2d)
                
                # Reshape ke (1, 1, 1500) untuk MiniRocket
                final_input = scaled_signal.reshape(1, 1, 1500)
                
                # 5. PREDIKSI
                if st.button("🔍 Identifikasi Pemilik Sinyal"):
                    with st.spinner("Sedang memproses..."):
                        pred_idx_up = pipeline.predict(final_input)[0]
                        pred_result = label_map[pred_idx_up]
                    
                    st.divider()
                    st.markdown(f"<h3 style='text-align: center; color: green;'>Hasil Identifikasi:</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h1 style='text-align: center;'>{pred_result}</h1>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")