import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import plotly.graph_objects as go

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="KidneyScan AI: Sistem Pendukung Keputusan Klinis", layout="wide", page_icon="🍃")

# --- 2. CSS CUSTOM: TEMA SOFT PASTEL MEDICAL ---
st.markdown("""
    <style>
    /* Background Utama Pastel */
    .stApp {
        background: #f0f7ff;
        color: #334155;
    }
    
    /* Box Putih Bersih */
    .main-box {
        background: white;
        border-radius: 25px;
        padding: 40px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 30px rgba(148, 163, 184, 0.1);
    }
    
    /* Judul Soft Blue */
    .neon-text {
        font-size: 45px;
        font-weight: 800;
        background: linear-gradient(to right, #0ea5e9, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }

    /* Input Styling (Light Mode) */
    .stNumberInput, .stSelectbox, .stSlider {
        border-radius: 12px !important;
    }
    
    /* Tab Styling Pastel */
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 12px;
        padding: 8px 25px;
        color: #64748b;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #bae6fd !important;
        color: #0369a1 !important;
        font-weight: bold;
    }

    /* Tombol Pastel Gradient */
    div.stButton > button {
        background: linear-gradient(45deg, #99f6e4, #bae6fd);
        color: #0369a1;
        border: none;
        height: 60px;
        border-radius: 15px;
        font-size: 20px;
        font-weight: bold;
        transition: 0.4s;
        width: 100%;
        box-shadow: 0 4px 15px rgba(186, 230, 253, 0.6);
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(186, 230, 253, 0.9);
        color: #0369a1;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. FUNGSI LOAD ASSETS ---
@st.cache_resource
def load_all():
    lr = pickle.load(open('model_lr.pkl', 'rb'))
    xgb = pickle.load(open('model_ckd.pkl', 'rb'))
    sc = pickle.load(open('scaler.pkl', 'rb'))
    return lr, xgb, sc

try:
    lr, xgb, sc = load_all()
except Exception as e:
    st.error(f"File model tidak ditemukan! Error: {e}")

# --- 4. HEADER ---
st.markdown('<p class="neon-text">KidneyScan AI: Sistem Pendukung Keputusan Klinis</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#64748b;">Asisten Cerdas untuk Skrining Kesehatan Ginjal Berbasis Data Laboratorium</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- 5. CONTAINER FORM INPUT ---
st.markdown('<div class="main-box">', unsafe_allow_html=True)
tabs = st.tabs(["📊 Vital & Lab Dasar", "🔬 Analisis Darah Detail", "🩺 Kondisi & Riwayat"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Usia Pasien", 1, 100, 45)
    bp = c2.number_input("Tekanan Darah (mm/Hg)", 50, 200, 80)
    sg = c3.selectbox("Berat Jenis Urin (sg)", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
    
    c4, c5, c6 = st.columns(3)
    al = c4.select_slider("Albumin (Urin)", options=[0,1,2,3,4,5], value=0)
    su = c5.select_slider("Gula (Urin)", options=[0,1,2,3,4,5], value=0)
    hemo = c6.slider("Hemoglobin (g/dL)", 3.0, 18.0, 13.0)

with tabs[1]:
    st.markdown("<p style='color:#64748b;'><i>*Lengkapi parameter darah untuk hasil lebih presisi</i></p>", unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    sc_val = c7.number_input("Serum Kreatinin (mg/dL)", 0.1, 15.0, 1.2)
    sod = c8.number_input("Natrium / Sodium (mEq/L)", 50, 200, 138)
    pot = c9.number_input("Kalium / Potassium (mEq/L)", 2.0, 10.0, 4.4)
    
    c10, c11, c12 = st.columns(3)
    pcv = c10.slider("PCV (%)", 5, 60, 40)
    wbcc = c11.number_input("Sel Darah Putih (WBCC)", 2000, 30000, 8000)
    rbcc = c12.number_input("Sel Darah Merah (RBCC)", 2.0, 10.0, 4.8)
    
    bgr = st.number_input("Gula Darah Acak (BGR)", 50, 500, 120)
    bu = st.number_input("Ureum Darah (BU)", 1, 400, 40)

with tabs[2]:
    c13, c14 = st.columns(2)
    rbc = c13.radio("Sel Darah Merah Urin", ["Normal", "Abnormal"], horizontal=True)
    pc = c14.radio("Sel Nanah Urin", ["Normal", "Abnormal"], horizontal=True)
    
    st.markdown("---")
    c15, c16, c17 = st.columns(3)
    htn = c15.checkbox("Riwayat Hipertensi")
    dm = c16.checkbox("Diabetes Mellitus")
    ane = c17.checkbox("Anemia")
    
    c18, c19 = st.columns(2)
    appet = c18.selectbox("Nafsu Makan", ["Baik", "Buruk"])
    pe = c19.selectbox("Bengkak Kaki (Edema)", ["Tidak", "Ya"])

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. LOGIKA PREDIKSI ---
if st.button("MULAI ANALISIS PRESISI"):
    with st.spinner("Memproses data klinis..."):
        final_input = [
            age, bp, sg, al, su,
            1 if rbc=="Normal" else 0, 1 if pc=="Normal" else 0, 
            0, 0, bgr, bu, sc_val, sod, pot, hemo, pcv, wbcc, rbcc,
            1 if htn else 0, 1 if dm else 0, 
            0, 1 if appet=="Baik" else 0, 1 if pe=="Ya" else 0, 1 if ane else 0
        ]
        
        data_scaled = sc.transform(np.array(final_input).reshape(1, -1))
        prob_lr = lr.predict_proba(data_scaled)[0][1]
        prob_xgb = xgb.predict_proba(data_scaled)[0][1]
        time.sleep(1)

    st.markdown("---")
    
    # --- 7. TAMPILAN HASIL (CARD PASTEL) ---
    res_c1, res_c2 = st.columns(2)
    
    def draw_card(name, prob, side_color, bg_color):
        status_txt = "BERISIKO TINGGI" if prob > 0.5 else "RISIKO RENDAH"
        txt_color = "#b91c1c" if prob > 0.5 else "#065f46"
        st.markdown(f"""
            <div style="background:{bg_color}; padding:25px; border-radius:20px; border-left:8px solid {side_color};">
                <small style="color:#64748b;">Sistem: {name}</small>
                <h2 style="color:{txt_color}; margin:10px 0;">{status_txt}</h2>
                <h3 style="margin:0; color:#334155;">Skor Risiko: {prob:.1%}</h3>
            </div>
        """, unsafe_allow_html=True)

    # Pastel Red bg: #fee2e2, Pastel Green bg: #dcfce7
    bg_lr = "#fee2e2" if prob_lr > 0.5 else "#dcfce7"
    bg_xgb = "#fee2e2" if prob_xgb > 0.5 else "#dcfce7"
    
    with res_c1: draw_card("Logistic Regression", prob_lr, "#0ea5e9", bg_lr)
    with res_c2: draw_card("XGBoost Classifier", prob_xgb, "#8b5cf6", bg_xgb)

    # Gauge Chart Pastel
    st.markdown("<br>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    
    def create_gauge(prob, title):
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob * 100,
            number = {'suffix': "%", 'font': {'color': '#334155'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickcolor': "#64748b"},
                'bar': {'color': "#3b82f6"},
                'bgcolor': "#f1f5f9",
                'steps': [{'range': [0, 50], 'color': '#d1fae5'}, {'range': [50, 100], 'color': '#fecaca'}]}
        ))
        fig.update_layout(height=280, title={'text': title, 'font': {'color': '#64748b'}}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=0))
        return fig

    g1.plotly_chart(create_gauge(prob_lr, "Indeks Risiko LR"), use_container_width=True)
    g2.plotly_chart(create_gauge(prob_xgb, "Indeks Risiko XGB"), use_container_width=True)

    # --- 8. REKOMENDASI ---
    st.markdown("### 📝 Saran & Rekomendasi")
    with st.expander("Klik untuk panduan pemulihan", expanded=True):
        col_s1, col_s2 = st.columns(2)
        if prob_xgb > 0.5:
            with col_s1:
                st.markdown("**🚑 Medis:**\n* Segera periksa GFR & USG Ginjal.\n* Evaluasi elektrolit darah.")
            with col_s2:
                st.markdown("**🥗 Gaya Hidup:**\n* Batasi garam & protein hewani.\n* Cukupi istirahat, hindari stres.")
        else:
            with col_s1:
                st.markdown("**✅ Pencegahan:**\n* Minum air putih cukup (2L+).\n* Rutin olahraga ringan.")
            with col_s2:
                st.markdown("**🍎 Nutrisi:**\n* Perbanyak serat dari sayur hijau.\n* Hindari makanan kemasan.")
st.warning("⚠️ **Peringatan:** Alat ini hanya untuk skrining awal. Hasil tidak menggantikan diagnosa klinis dari Nefrolog (Dokter Spesialis Ginjal).")

st.markdown('</div>', unsafe_allow_html=True)