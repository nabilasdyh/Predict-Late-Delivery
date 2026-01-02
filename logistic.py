#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
import os

# Mengabaikan peringatan agar tampilan terminal bersih
warnings.filterwarnings('ignore')

# --- KONFIGURASI FILE ---
# Pastikan file-file ini berada di folder yang sama dengan script ini
MODEL_PATH = 'rf_model_pipeline.pkl'
MAPPING_TABLE_PATH = 'shipping_mapping_table.pkl'
DATA_PATH = 'df_clean.csv' 
RISK_THRESHOLD = 0.50 

# --- INISIALISASI STATE ---
if 'is_loaded' not in st.session_state:
    st.session_state.is_loaded = False
if 'mapping_table' not in st.session_state:
    st.session_state.mapping_table = None
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None 
if 'col_options' not in st.session_state: 
    st.session_state.col_options = None 
if 'model_features' not in st.session_state:
    st.session_state.model_features = None 
if 'model_risk' not in st.session_state:
    st.session_state.model_risk = None

def generate_options_from_data(df):
    """Mengekstrak opsi unik untuk 16 fitur utama di UI."""
    options = {}
    
    # Fitur Kategorikal
    cat_features = [
        'Type', 'Customer Country', 'Customer Segment', 'Department Name',
        'Market', 'Order Country', 'Order Region', 'Shipping Mode', 'Item_Bucket'
    ]
    for col in cat_features:
        if col in df.columns:
            unique_values = df[col].astype(str).unique().tolist()
            unique_values = sorted([v for v in unique_values if v not in ['Other', 'nan']])
            if col in ['Type', 'Customer Country', 'Department Name', 'Order Country', 'Order Region']:
                 unique_values.append('Other')
            options[col] = unique_values

    # Fitur Boolean
    options['Is_Weekend'] = [True, False] 

    # Fitur Numerik (Min, Max, Default)
    num_features = [
        'Days for shipment (scheduled)', 'Order Item Discount Rate', 
        'order month', 'order hour', 'Sales per customer', 'Category Id'
    ]
    for col in num_features:
        if col in df.columns:
            min_val, max_val, median_val = df[col].min(), df[col].max(), df[col].median()
            options[col] = (float(min_val), float(max_val), float(median_val))
                
    return options

def align_inputs_with_model(input_df, required_features):
    """
    Menyuntikkan kolom dummy untuk fitur yang tidak ada di UI
    tapi dibutuhkan oleh model untuk prediksi.
    """
    df_aligned = input_df.copy()
    for col in required_features:
        if col not in df_aligned.columns:
            df_aligned[col] = 0 
            
    # Pastikan urutan kolom sesuai dengan saat training
    return df_aligned[required_features]

@st.cache_resource(show_spinner=False)
def load_all_assets():
    """Memuat model, data mapping, dan struktur fitur."""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"File model '{MODEL_PATH}' tidak ditemukan!")
            return False
            
        ml_model = joblib.load(MODEL_PATH)
        mapping_table = joblib.load(MAPPING_TABLE_PATH)
        df_sample = pd.read_csv(DATA_PATH)
        
        # Ambil daftar fitur yang dibutuhkan model
        if hasattr(ml_model, 'feature_names_in_'):
            model_features = ml_model.feature_names_in_.tolist()
        else:
            model_features = [c for c in df_sample.columns if c != 'Late_delivery_risk']
            
        col_options = generate_options_from_data(df_sample)
        
        st.session_state.mapping_table = mapping_table
        st.session_state.ml_model = ml_model
        st.session_state.col_options = col_options
        st.session_state.model_features = model_features
        st.session_state.is_loaded = True
        return True
    except Exception as e:
        st.error(f"Gagal memuat aset: {e}")
        return False

# --- UI STREAMLIT ---
st.set_page_config(layout="centered", page_title="AI Logistik")

if not st.session_state.is_loaded:
    with st.spinner("Menginisialisasi Sistem AI..."):
        if load_all_assets():
            st.rerun()

st.title("🚛 Prediksi Risiko Keterlambatan Logistik")
st.markdown("Aplikasi ini memprediksi risiko pengiriman berdasarkan data historis dan parameter order.")

# SIDEBAR INPUT
input_data = {}
st.sidebar.header("Parameter Pengiriman")

if st.session_state.col_options:
    for col, options in st.session_state.col_options.items():
        if isinstance(options, tuple):
            input_data[col] = st.sidebar.slider(col, options[0], options[1], options[2])
        elif col == 'Is_Weekend':
            input_data[col] = st.sidebar.checkbox(col, value=options[0])
        else:
            input_data[col] = st.sidebar.selectbox(col, options)

if st.sidebar.button("ANALISIS RISIKO", use_container_width=True):
    ui_df = pd.DataFrame([input_data])
    
    try:
        # Penyelarasan fitur
        final_input_df = align_inputs_with_model(ui_df, st.session_state.model_features)
        
        # Prediksi
        proba = st.session_state.ml_model.predict_proba(final_input_df)[0][1]
        st.session_state.model_risk = proba
        
        st.subheader("Hasil Analisis")
        col1, col2 = st.columns(2)
        
        with col1:
            if proba > RISK_THRESHOLD:
                st.error("🚨 RISIKO TINGGI")
            else:
                st.success("✅ RISIKO RENDAH")
        
        with col2:
            st.metric("Probabilitas Terlambat", f"{proba:.2%}")

        # Rekomendasi Sederhana
        st.markdown("---")
        st.subheader("Rekomendasi Strategi")
        if proba > RISK_THRESHOLD:
            st.warning("Disarankan untuk meninjau kembali 'Shipping Mode' atau mempercepat proses packaging.")
        else:
            st.info("Parameter saat ini terlihat aman untuk dikirim tepat waktu.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("Sistem AI menggunakan Random Forest Classifier")
    
# In[ ]:





