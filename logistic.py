#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
import time 
import os

warnings.filterwarnings('ignore')

# Ambil path file model dan ambang batas risiko
MODEL_PATH = 'rf_model_pipeline.pkl'
MAPPING_TABLE_PATH = 'shipping_mapping_table.pkl'
DATA_PATH = 'df_clean.csv' 
RISK_THRESHOLD = 0.50 # Ambang batas risiko: 50% (0.50)

# --- INISIALISASI STATE STREAMLIT ---
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
if 'feature_medians' not in st.session_state:
    st.session_state.feature_medians = {} # TAMBAHAN: Untuk menyimpan nilai normal fitur non-UI

# --- FUNGSI: MENGEKSTRAK OPSI DARI DATASET ---
def generate_options_from_data(df):
    options = {}
    
    # Simpan median untuk semua kolom numerik agar prediksi model akurat (tidak nol)
    medians = df.median(numeric_only=True).to_dict()
    
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

    options['Is_Weekend'] = [True, False] 

    num_features = [
        'Days for shipment (scheduled)', 
        'Order Item Discount Rate', 'order month', 'order hour'
    ]
    for col in num_features:
        min_val = df[col].min()
        max_val = df[col].max()
        median_val = df[col].median()
        
        if df[col].dtype in (np.int64, np.int32, 'int'):
            options[col] = (int(min_val), int(max_val), int(median_val))
        else:
            options[col] = (float(min_val), float(max_val), float(median_val))
            
    return options, medians

# --- 1. FUNGSI MEMUAT SEMUA ASET ---
@st.cache_resource(show_spinner=False)
def load_all_assets():
    ml_model = None
    mapping_table = None
    col_options = None
    medians = {}
    
    st.markdown("---")

    if os.path.exists(MAPPING_TABLE_PATH):
        try:
            mapping_table = joblib.load(MAPPING_TABLE_PATH)
            st.success("✅ Tabel Logistik Historis berhasil dimuat.")
        except Exception as e:
            st.error(f"❌ GAGAL memuat Tabel Logistik Historis: {e}")

    if os.path.exists(MODEL_PATH):
        try:
            ml_model = joblib.load(MODEL_PATH)
            st.success(f"✅ Model Prediksi Risiko berhasil dimuat.")
        except Exception as e:
            st.error(f"❌ GAGAL memuat model: {e}")
        
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            col_options, medians = generate_options_from_data(df)
            st.success("✅ Opsi Input (16 Fitur) berhasil diambil.")
        except Exception as e:
            st.error(f"❌ GAGAL memproses data: {e}")

    st.session_state.mapping_table = mapping_table
    st.session_state.ml_model = ml_model
    st.session_state.col_options = col_options
    st.session_state.feature_medians = medians
    st.session_state.is_loaded = True
    time.sleep(1)
    st.rerun() 

# --- LOGIKA REKOMENDASI MANUAL (DIKEMBALIKAN SESUAI ASLI) ---
def recommend_from_mapping_table(order_data, mapping_table):
    model_risk_proba = st.session_state.model_risk
    is_risky_by_model = model_risk_proba is not None and model_risk_proba > RISK_THRESHOLD
    
    grouping_cols = ["Order Region", "Item_Bucket", "Customer Segment", "Market"]
    current_segment = order_data[grouping_cols].iloc[0]
    current_shipping_mode = order_data['Shipping Mode'].iloc[0]
    
    filter_mask = pd.Series(True, index=mapping_table.index)
    for col in grouping_cols:
        filter_mask &= (mapping_table[col] == current_segment[col])
        
    safe_alternatives = mapping_table[filter_mask].sort_values('Late_Rate')
    
    if safe_alternatives.empty:
        st.warning(f"**Tidak ada rekomendasi historis aman** untuk segmen order ini.")
        return
        
    best_reco = safe_alternatives.iloc[0]
    reco_mode = best_reco['Shipping Mode']
    reco_rate = best_reco['Late_Rate']
    
    current_rate_df = safe_alternatives[safe_alternatives['Shipping Mode'] == current_shipping_mode]['Late_Rate']
    current_rate = current_rate_df.iloc[0] if not current_rate_df.empty else None

    st.markdown("---")
    st.subheader("Rekomendasi Pengiriman")

    if current_shipping_mode == reco_mode:
        st.info(f"Mode pengiriman saat ini, **{current_shipping_mode}**, adalah pilihan **terbaik** (Risiko: **{reco_rate:.1%}**).")
    elif is_risky_by_model:
        st.error(f"Berdasarkan risiko tinggi, disarankan **beralih** ke mode **{reco_mode}** (Risiko: **{reco_rate:.1%}**).")
    else:
        st.success(f"Mode pengiriman **{current_shipping_mode}** dinilai cukup aman. Alternatif: **{reco_mode}** (**{reco_rate:.1%}**).")

    if current_rate is not None:
        st.caption(f"*Catatan: Risiko keterlambatan historis untuk {current_shipping_mode} adalah {current_rate:.1%}.*")

# --- 2. LOGIKA PREDIKSI / REKOMENDASI UTAMA ---
def get_prediction_and_recommendation(input_df, ml_model, mapping_table):
    st.session_state.model_risk = None
    if ml_model is not None:
        st.subheader("Hasil Prediksi Waktu Kirim")
        try:
            proba = ml_model.predict_proba(input_df)[0]
            risk_proba = proba[1] 
            st.session_state.model_risk = risk_proba
            
            if risk_proba > RISK_THRESHOLD:
                st.error(f"🚨 PREDIKSI: **TERLAMBAT**")
                st.markdown(f"**Risiko Keterlambatan:** **{risk_proba * 100:.2f}%**")
            else:
                st.success(f"✅ PREDIKSI: **TEPAT WAKTU**")
                st.caption(f"*Probabilitas keterlambatan: {risk_proba * 100:.2f}%.*")
            
            recommend_from_mapping_table(input_df, mapping_table)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            recommend_from_mapping_table(input_df, mapping_table)
    else:
        st.subheader("Hasil Analisis Logistik (Mode Manual)")
        recommend_from_mapping_table(input_df, mapping_table)

# --- 3. ANTARMUKA STREAMLIT UTAMA ---
st.set_page_config(layout="centered", page_title="Prediksi & Rekomendasi Logistik")

if not st.session_state.is_loaded:
    st.title("🚛 Memuat Aset Aplikasi...")
    load_all_assets()

if st.session_state.mapping_table is None or st.session_state.col_options is None:
    st.error("Aplikasi tidak dapat dimuat. Periksa file aset.")
else:
    mapping_table = st.session_state.mapping_table
    ml_model = st.session_state.ml_model
    col_options = st.session_state.col_options

    st.title("🚛 Prediksi & Rekomendasi Pengiriman")
    
    input_data = {}
    st.sidebar.header("Input Data Order (16 Fitur)")

    for col, options in col_options.items():
        if isinstance(options, tuple): 
            input_data[col] = st.sidebar.slider(col, options[0], options[1], options[2], key=col)
        elif col == 'Is_Weekend': 
            input_data[col] = st.sidebar.checkbox(col, value=options[0], key=col)
        else: 
            default_index = options.index('Other') if 'Other' in options else 0
            input_data[col] = st.sidebar.selectbox(col, options, index=default_index, key=col)

    if st.sidebar.button("ANALISIS ORDER", use_container_width=True):
        with st.spinner('Menganalisis...'):
            # --- TAHAP SINKRONISASI 22 KOLOM (DIPERBAIKI) ---
            full_model_cols = [
                'Type', 'Days for shipment (scheduled)', 'Benefit per order',
                'Customer Country', 'Customer Segment', 'Customer Zipcode',
                'Department Name', 'Market', 'Order Country', 'Order Item Discount Rate',
                'Order Item Profit Ratio', 'Order Item Quantity', 'Sales per customer',
                'Order Region', 'Category Id', 'Product Price', 'Shipping Mode', 
                'Is_Weekend', 'order month', 'order weekday', 'order hour', 'Item_Bucket'
            ]
            
            final_input = []
            for col in full_model_cols:
                if col in input_data:
                    final_input.append(input_data[col])
                else:
                    # AMBIL MEDIAN (Bukan 0) agar tidak selalu direkomendasikan Same Day
                    final_input.append(st.session_state.feature_medians.get(col, 0))
            
            input_df = pd.DataFrame([final_input], columns=full_model_cols)
            get_prediction_and_recommendation(input_df, ml_model, mapping_table)
