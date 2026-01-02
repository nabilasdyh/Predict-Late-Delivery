#!/usr/bin/env python
# coding: utf-8

# In[ ]:

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

# Konfigurasi Path dan Threshold
MODEL_PATH = 'rf_model_pipeline.pkl'
MAPPING_TABLE_PATH = 'shipping_mapping_table.pkl'
DATA_PATH = 'df_clean.csv' 
RISK_THRESHOLD = 0.50 

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

# --- FUNGSI: MENGEKSTRAK OPSI DARI DATASET (SESUAI 20 FITUR TRAINING) ---
def generate_options_from_data(df):
    options = {}
    
    # Fitur Kategorikal (Berdasarkan df.info)
    cat_features = [
        'Type', 'Customer Country', 'Customer Segment', 'Department Name',
        'Market', 'Order Country', 'Order Region', 'Shipping Mode', 'Item_Bucket'
    ]
    
    for col in cat_features:
        if col in df.columns:
            unique_values = df[col].astype(str).unique().tolist()
            if 'Other' in unique_values: unique_values.remove('Other')
            if 'nan' in unique_values: unique_values.remove('nan')
            unique_values = sorted(unique_values)
            if col in ['Type', 'Customer Country', 'Department Name', 'Order Country', 'Order Region']:
                 unique_values.append('Other')
            options[col] = unique_values

    # Fitur Boolean
    options['Is_Weekend'] = [True, False] 

    # Fitur Numerik (Lengkap 10 fitur numerik dari df.info)
    num_features = [
        'Days for shipment (scheduled)', 'Benefit per order', 'Customer Zipcode',
        'Order Item Discount Rate', 'Order Item Profit Ratio', 'Order Item Quantity',
        'Product Price', 'order month', 'order weekday', 'order hour'
    ]
    
    for col in num_features:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            median_val = df[col].median()
            
            if df[col].dtype in (np.int64, np.int32, 'int'):
                options[col] = (int(min_val), int(max_val), int(median_val))
            else:
                options[col] = (float(min_val), float(max_val), float(median_val))
                
    return options

# --- 1. FUNGSI MEMUAT ASET ---
@st.cache_resource(show_spinner=False)
def load_all_assets():
    ml_model = None
    mapping_table = None
    col_options = None
    
    # Load Mapping Table
    if os.path.exists(MAPPING_TABLE_PATH):
        try:
            mapping_table = joblib.load(MAPPING_TABLE_PATH)
        except Exception as e:
            st.error(f"Gagal muat mapping: {e}")

    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
            ml_model = joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Gagal muat model: {e}")
        
    # Load Data & Generate Options
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            col_options = generate_options_from_data(df)
        except Exception as e:
            st.error(f"Gagal muat data: {e}")

    st.session_state.mapping_table = mapping_table
    st.session_state.ml_model = ml_model
    st.session_state.col_options = col_options
    st.session_state.is_loaded = True
    time.sleep(0.5)
    st.rerun() 

# --- 2. LOGIKA REKOMENDASI ---
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
        st.warning("Tidak ada rekomendasi historis aman untuk segmen ini.")
        return
        
    best_reco = safe_alternatives.iloc[0]
    reco_mode = best_reco['Shipping Mode']
    reco_rate = best_reco['Late_Rate']
    
    current_rate_df = safe_alternatives[safe_alternatives['Shipping Mode'] == current_shipping_mode]['Late_Rate']
    current_rate = current_rate_df.iloc[0] if not current_rate_df.empty else None

    st.markdown("---")
    st.subheader("Rekomendasi Pengiriman")
    
    if current_shipping_mode == reco_mode:
        st.info(f"Mode **{current_shipping_mode}** adalah pilihan terbaik (Risiko Historis: **{reco_rate:.1%}**).")
    elif is_risky_by_model:
        st.error(f"Disarankan **beralih** ke mode **{reco_mode}** (Risiko Historis: **{reco_rate:.1%}**).")
    else:
        st.success(f"Mode **{current_shipping_mode}** cukup aman. Alternatif terbaik: **{reco_mode}** (**{reco_rate:.1%}**).")

# --- 3. UI UTAMA ---
st.set_page_config(layout="centered", page_title="Prediksi Logistik")

if not st.session_state.is_loaded:
    with st.spinner("Memuat Aset..."):
        load_all_assets()

if st.session_state.mapping_table is None or st.session_state.col_options is None:
    st.error("File aset tidak lengkap.")
else:
    mapping_table = st.session_state.mapping_table
    ml_model = st.session_state.ml_model
    col_options = st.session_state.col_options

    st.title("🚛 Prediksi & Rekomendasi Pengiriman")
    
    input_data = {}
    st.sidebar.header("Input Data Order (20 Fitur)")

    # Render Sidebar secara dinamis sesuai col_options
    for col, options in col_options.items():
        if isinstance(options, tuple): 
            # Fitur numerik (Slider)
            input_data[col] = st.sidebar.slider(col, options[0], options[1], options[2], key=col)
        elif col == 'Is_Weekend': 
            input_data[col] = st.sidebar.checkbox(col, value=options[0], key=col)
        else: 
            # Fitur kategorikal (Selectbox)
            default_idx = options.index('Other') if 'Other' in options else 0
            input_data[col] = st.sidebar.selectbox(col, options, index=default_idx, key=col)

    if st.sidebar.button("ANALISIS ORDER", use_container_width=True):
        with st.spinner('Memproses...'):
            # URUTAN KOLOM HARUS PERSIS DENGAN DF.INFO (Training)
            ordered_cols = [
                'Type', 'Days for shipment (scheduled)', 'Benefit per order',
                'Customer Country', 'Customer Segment', 'Customer Zipcode',
                'Department Name', 'Market', 'Order Country', 'Order Item Discount Rate',
                'Order Item Profit Ratio', 'Order Item Quantity', 'Order Region',
                'Product Price', 'Shipping Mode', 'Is_Weekend', 'order month',
                'order weekday', 'order hour', 'Item_Bucket'
            ]
            
            try:
                input_list = [input_data[col] for col in ordered_cols]
                input_df = pd.DataFrame([input_list], columns=ordered_cols)
                
                if ml_model is not None:
                    st.subheader("Hasil Prediksi Waktu Kirim")
                    proba = ml_model.predict_proba(input_df)[0]
                    risk_proba = proba[1]
                    st.session_state.model_risk = risk_proba
                    
                    if risk_proba > RISK_THRESHOLD:
                        st.error(f"🚨 PREDIKSI: **TERLAMBAT** (Risiko: {risk_proba*100:.1f}%)")
                    else:
                        st.success(f"✅ PREDIKSI: **TEPAT WAKTU** (Risiko: {risk_proba*100:.1f}%)")
                
                recommend_from_mapping_table(input_df, mapping_table)
            except Exception as e:
                st.error(f"Error pada model: {e}")
    
# In[ ]:
