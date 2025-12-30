#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
import time 
import os

warnings.filterwarnings('ignore')

MODEL_PATH = 'rf_model_pipeline.pkl'
MAPPING_TABLE_PATH = 'shipping_mapping_table.pkl'
DATA_PATH = 'df_clean.csv'  
RISK_THRESHOLD = 0.50 # Ambang batas risiko: 50% (0.50)

if 'is_loaded' not in st.session_state:
    st.session_state.is_loaded = False
if 'mapping_table' not in st.session_state:
    st.session_state.mapping_table = None
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None 
if 'col_options' not in st.session_state: 
    st.session_state.col_options = None 
if 'model_risk' not in st.session_state:
    st.session_state.model_risk = None

def generate_options_from_data(df):
    options = {}
    
    cat_features = [
        'Type', 'Customer Country', 'Customer Segment', 'Department Name',
        'Market', 'Order Country', 'Order Region', 'Shipping Mode', 'Item_Bucket'
    ]
    for col in cat_features:
        unique_values = df[col].astype(str).unique().tolist()
        if 'Other' in unique_values: unique_values.remove('Other')
        if 'nan' in unique_values: unique_values.remove('nan')
        
        unique_values = sorted(unique_values)
        
        if col in ['Type', 'Customer Country', 'Department Name', 'Order Country', 'Order Region']:
             unique_values.append('Other')
             
        options[col] = unique_values

    # Fitur Boolean/Binary (Checkbox)
    options['Is_Weekend'] = [True, False] 

    # Kunci untuk fitur Numerical (Slider)
    num_features = [
        'Days for shipment (scheduled)', 'Sales per customer', 'Category Id',
        'Order Item Discount Rate', 'order month', 'order hour'
    ]
    for col in num_features:
        min_val = df[col].min()
        max_val = df[col].max()
        # Gunakan nilai tengah (median) sebagai default
        median_val = df[col].median()
        
        if df[col].dtype in (np.int64, np.int32, 'int'):
            options[col] = (int(min_val), int(max_val), int(median_val))
        else:
            options[col] = (float(min_val), float(max_val), float(median_val))
            
    return options

# FUNGSI MEMUAT SEMUA ASET
@st.cache_resource(show_spinner=False)
def load_all_assets():
    """Memuat model, mapping table, dan menghasilkan opsi dinamis."""
    
    ml_model = None
    mapping_table = None
    col_options = None
    
    st.markdown("---")

    # Muat Mapping Table
    if os.path.exists(MAPPING_TABLE_PATH):
        try:
            mapping_table = joblib.load(MAPPING_TABLE_PATH)
            st.success("✅ Tabel Logistik Historis berhasil dimuat.")
        except Exception as e:
            st.error(f"❌ GAGAL memuat Tabel Logistik Historis: {e}")
    else:
        st.error(f"❌ File Tabel Logistik Historis '{MAPPING_TABLE_PATH}' tidak ditemukan.")

    # Muat Model Prediksi
    if os.path.exists(MODEL_PATH):
        try:
            ml_model = joblib.load(MODEL_PATH)
            st.success(f"✅ Model Prediksi Risiko berhasil dimuat.")
        except Exception as e:
            st.error(f"❌ GAGAL memuat model prediksi '{MODEL_PATH}' karena Error: {e}")
            st.warning("⚠️ Aplikasi akan berjalan dalam Mode Rekomendasi Manual saja.")
    else:
        st.error(f"❌ File Model Prediksi '{MODEL_PATH}' tidak ditemukan.")
        
    # Muat Dataset dan Buat Pilihan Dinamis
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            col_options = generate_options_from_data(df)
            st.success("✅ Opsi Input (16 Fitur) berhasil diambil dari data sumber.")
        except Exception as e:
            st.error(f"❌ GAGAL memuat atau memproses data sumber '{DATA_PATH}': {e}")
            st.warning("⚠️ Aplikasi akan menggunakan opsi default. Harap perbaiki data sumber.")
    else:
        st.error(f"❌ File Data Sumber '{DATA_PATH}' tidak ditemukan. Opsi input TIDAK dinamis.")
        # Fallback Opsi
        col_options = {
            'Type': ['CASH', 'DEBIT', 'TRANSFER', 'PAYMENT', 'Other'],
            'Days for shipment (scheduled)': (1, 30, 5),
            'Sales per customer': (10.0, 500.0, 150.0),
            'Category Id': (1, 100, 20),
            'Customer Country': ['USA', 'Mexico', 'UK', 'France', 'Germany', 'Japan', 'Other'],
            'Customer Segment': ['Consumer', 'Corporate', 'Home Office', 'Small Business'],
            'Department Name': ['Fitness', 'Apparel', 'Electronics', 'Footwear', 'Outdoor', 'Other'],
            'Market': ['LATAM', 'Europe', 'APAC', 'USCA', 'Africa'],
            'Order Country': ['USA', 'Mexico', 'UK', 'France', 'Germany', 'Japan', 'Other'],
            'Order Item Discount Rate': (0.0, 0.5, 0.1),
            'Order Region': ['Central', 'West', 'East', 'South', 'North', 'Other'],
            'Shipping Mode': ['Standard Class', 'First Class', 'Second Class', 'Same Day'],
            'Is_Weekend': [True, False],
            'order month': (1, 12, 6),
            'order hour': (0, 23, 12),
            'Item_Bucket': ['≤2', '3–4', '≥5'],
        }


    st.session_state.mapping_table = mapping_table
    st.session_state.ml_model = ml_model
    st.session_state.col_options = col_options
    st.session_state.is_loaded = True
    time.sleep(1)
    st.rerun() 
    
# LOGIKA REKOMENDASI MANUAL
def recommend_from_mapping_table(order_data, mapping_table):
    
    # Cek risiko model yang sudah dihitung di fungsi utama
    model_risk_proba = st.session_state.model_risk
    is_risky_by_model = model_risk_proba is not None and model_risk_proba > RISK_THRESHOLD
    
    # 4 Kolom Kunci Logistik/Segmentasi yang digunakan sebagai FILTER (tanpa Shipping Mode)
    grouping_cols = ["Order Region", "Item_Bucket", "Customer Segment", "Market"]
    
    current_segment = order_data[grouping_cols].iloc[0]
    current_shipping_mode = order_data['Shipping Mode'].iloc[0]
    
    filter_mask = pd.Series(True, index=mapping_table.index)
    for col in grouping_cols:
        filter_mask &= (mapping_table[col] == current_segment[col])
        
    safe_alternatives = mapping_table[filter_mask].sort_values('Late_Rate')
    
    if safe_alternatives.empty:
        st.warning(
            f"**Tidak ada rekomendasi historis aman** untuk segmen order ini."
        )
        return
        
    best_reco = safe_alternatives.iloc[0]
    reco_mode = best_reco['Shipping Mode']
    reco_rate = best_reco['Late_Rate']
    
    # Cari Late Rate historis untuk mode yang sedang diinput
    current_rate_df = safe_alternatives[safe_alternatives['Shipping Mode'] == current_shipping_mode]['Late_Rate']
    current_rate = current_rate_df.iloc[0] if not current_rate_df.empty else None

    
    st.markdown("---")
    st.subheader("Rekomendasi Pengiriman")

    # KONSISTENSI LOGIKA UTAMA
    
    if current_shipping_mode == reco_mode:
        # KASUS 1: Mode saat ini adalah yang TERBAIK secara historis
        st.info(
            f"Mode pengiriman saat ini, **{current_shipping_mode}**, adalah pilihan **terbaik** di segmen historis ini (Risiko Keterlambatan Historis: **{reco_rate:.1%}**)."
        )
    elif is_risky_by_model:
        # KASUS 2: Mode saat ini BUKAN yang terbaik DAN MODEL memprediksi TERLAMBAT
        st.error(
            f"Berdasarkan risiko tinggi yang terprediksi, disarankan **beralih** ke mode **{reco_mode}** (Risiko Keterlambatan Historis: **{reco_rate:.1%}**)."
        )
    else:
        # KASUS 3: Mode saat ini BUKAN yang terbaik, tapi MODEL memprediksi TEPAT WAKTU.
        # Jangan menyarankan beralih, tapi konfirmasi bahwa saat ini OK.
        st.success(
            f"Mode pengiriman **{current_shipping_mode}** dinilai cukup aman. "
            f"Alternatif tercepat/teraman adalah **{reco_mode}** (Risiko Historis: **{reco_rate:.1%}**)."
        )


    # Tampilkan catatan historis
    if current_rate is not None:
        st.caption(f"*Catatan: Risiko keterlambatan historis untuk {current_shipping_mode} adalah {current_rate:.1%} di segmen ini.*")
    else:
        st.caption(f"*Catatan: Tidak ada data historis yang memadai untuk {current_shipping_mode} di segmen ini.*")
    


# LOGIKA PREDIKSI / REKOMENDASI UTAMA (MODE PENUH)

def get_prediction_and_recommendation(input_df, ml_model, mapping_table):
    
    # A. LOGIKA UTAMA (PREDIKSI DENGAN MODEL)
    st.session_state.model_risk = None
    
    if ml_model is not None:
        st.subheader("Hasil Prediksi Waktu Kirim")
        
        try:
            # 1. Prediksi Probabilitas
            proba = ml_model.predict_proba(input_df)[0]
            risk_proba = proba[1] # Probabilitas kelas 1 (terlambat)
            
            # Simpan risiko model di session state untuk dipakai di fungsi rekomendasi
            st.session_state.model_risk = risk_proba
            
            # 2. Terapkan logika revisi dari user (Threshold = 0.50)
            is_risky = risk_proba > RISK_THRESHOLD 
            risk_percent = f"{risk_proba * 100:.2f}%"
            
            if is_risky:
                # > 0.50: Tampilkan Terlambat dan Probabilitas
                st.error(f"🚨 PREDIKSI: **TERLAMBAT**")
                st.markdown(f"**Risiko Keterlambatan:** **{risk_percent}**")
            else:
                # <= 0.50: Tampilkan Tepat Waktu saja
                st.success(f"✅ PREDIKSI: **TEPAT WAKTU**")
                st.caption(f"*Probabilitas keterlambatan: {risk_percent}.*")
            
            # 3. Rekomendasi 
            recommend_from_mapping_table(input_df, mapping_table)

        except Exception as e:
            st.error(f"⚠️ Error saat menjalankan prediksi model: {e}")
            st.warning("Terjadi masalah saat prediksi. Beralih ke Mode Logistik Manual.")
            st.markdown("---")
            # Jalankan mode fallback jika model error
            recommend_from_mapping_table(input_df, mapping_table)
            
    # B. LOGIKA FALLBACK (MODE LOGISTIK MANUAL)
    else:
        st.subheader("Hasil Analisis Logistik (Mode Manual)")
        st.warning("⚠️ Prediksi Risiko DITONAKTIFKAN. Aplikasi berjalan dalam Mode Rekomendasi Historis saja.")
        recommend_from_mapping_table(input_df, mapping_table)


# ANTARMUKA STREAMLIT UTAMA

st.set_page_config(layout="centered", page_title="Prediksi & Rekomendasi Logistik")


# INITALIZATION LOGIC
if not st.session_state.is_loaded:
    st.title("🚛 Memuat Aset Aplikasi...")
    with st.spinner("⏳ Mencoba memuat model prediksi dan data historis..."):
        load_all_assets()
# END INITALIZATION LOGIC


# MAIN APP UI
if st.session_state.mapping_table is None or st.session_state.col_options is None:
    st.title("Aplikasi Logistik Gagal Total")
    st.error("Aplikasi tidak dapat dilanjutkan. Harap pastikan file aset ('rf_model_pipeline.pkl', 'shipping_mapping_table.pkl' tersedia.")

else:
    # Aset yang berhasil dimuat
    mapping_table = st.session_state.mapping_table
    ml_model = st.session_state.ml_model
    col_options = st.session_state.col_options

    st.title("🚛 Prediksi & Rekomendasi Pengiriman")
    st.caption(f"Ambang batas risiko keterlambatan: {RISK_THRESHOLD*100:.0f}%.")
    
    if ml_model is not None:
        st.caption("Status: Mode Penuh (Prediksi & Rekomendasi) AKTIF.")
    else:
        st.caption("Status: Mode Fallback (Rekomendasi Historis) AKTIF.")

    input_data = {}

    st.sidebar.header("Input Data Order (16 Fitur)")
    st.sidebar.caption("Gunakan data historis untuk mendapatkan rekomendasi terbaik.")

    # Loop menggunakan opsi dinamis
    for col, options in col_options.items():
        # Kontrol input di sidebar
        if isinstance(options, tuple): 
            # Fitur numerik (Slider)
            if col in ['Sales per customer', 'Order Item Discount Rate']:
                input_data[col] = st.sidebar.slider(
                    col, 
                    float(options[0]), float(options[1]), float(options[2]), 
                    format="%.2f", key=col
                )
            else:
                input_data[col] = st.sidebar.slider(
                    col, 
                    options[0], options[1], options[2], 
                    key=col
                )
        elif col == 'Is_Weekend': 
            # Fitur Boolean (Checkbox)
            input_data[col] = st.sidebar.checkbox(col, value=options[0], key=col)
        else: 
            # Fitur kategorikal (Selectbox)
            default_index = 0
            if 'Other' in options:
                 default_index = options.index('Other')
            
            input_data[col] = st.sidebar.selectbox(col, options, index=default_index, key=col)

    st.sidebar.markdown("---")

    if st.sidebar.button("ANALISIS ORDER", use_container_width=True):
        
        with st.spinner('Menganalisis dan memproses rekomendasi...'):
            
            # 1. PERSIAPAN DATA INPUT (16 Fitur)
            ordered_cols = list(col_options.keys()) 
            input_list = [input_data[col] for col in ordered_cols]
            input_df = pd.DataFrame([input_list], columns=ordered_cols)
            
            # 2. EKSEKUSI LOGIKA UTAMA
            get_prediction_and_recommendation(input_df, ml_model, mapping_table)

    st.sidebar.markdown("---")
    st.sidebar.caption("Mode analisis: Berbasis Random Forest dan data historis.")
    
# In[ ]:





