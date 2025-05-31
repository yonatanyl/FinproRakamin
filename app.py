# app.py
# ---------------------------------------------------------------------
# Aplikasi Streamlit: Prediksi Karyawan Resign
# ---------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import re
import matplotlib.pyplot as plt # <-- Pindahkan import ke sini

# ------------------------------------------------------------------ #
# 0. Konfigurasi halaman                                             #
# ------------------------------------------------------------------ #
st.set_page_config(page_title="Prediksi Resign Karyawan", layout="wide", page_icon="💼")

# ------------------------------------------------------------------ #
# 1. Loader artefak                                                  #
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner="Memuat model …")
def load_pipeline(path: str):
    return joblib.load(path)

@st.cache_resource
def load_label_encoders(path: str):
    return joblib.load(path)

@st.cache_resource
def load_feature_columns(path: str):
    return joblib.load(path)

@st.cache_resource
def load_imputation(path: str):
    return joblib.load(path) if Path(path).exists() else {}

@st.cache_data
def load_city_mapping(path: str):
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

# ------------------------------------------------------------------ #
# 2. Inisialisasi artefak                                            #
# ------------------------------------------------------------------ #
MODEL_PATH = "model_pipeline_logreg.pkl"
ENCODER_PATH = "label_encoders.pkl"
COL_ORDER_PATH = "model_feature_columns.pkl"
IMPUTATION_PATH = "imputation_values.pkl"
CITY_CDI_MAP_PATH = "city_cdi_mapping.json"
DEFAULT_CDI_FALLBACK = 0.80

pipeline_model = load_pipeline(MODEL_PATH)
label_encoders = load_label_encoders(ENCODER_PATH)
model_cols = load_feature_columns(COL_ORDER_PATH)
imputation_defaults = load_imputation(IMPUTATION_PATH)
city_cdi_map = load_city_mapping(CITY_CDI_MAP_PATH)

if pipeline_model is None or not model_cols or label_encoders is None:
    st.error("Model, daftar kolom, atau encoder gagal dimuat. Aplikasi berhenti.")
    st.stop()

# ------------------------------------------------------------------ #
# 3. List opsi untuk widget                                          #
# ------------------------------------------------------------------ #
def classes_or_fallback(col, default):
    encoder = label_encoders.get(col)
    if encoder is not None and hasattr(encoder, 'classes_'):
        return list(encoder.classes_)
    return default

def sort_experience(values):
    def sort_key(val):
        if val == "<1":
            return 0
        elif val == ">20":
            return 21
        try:
            return int(val)
        except:
            return 999
    return sorted(values, key=sort_key)

def sort_education(values):
    urutan = {
        "Primary School": 0,
        "High School": 1,
        "Graduate": 2,
        "Masters": 3,
        "Phd": 4
    }
    return sorted(values, key=lambda x: urutan.get(x, 999))

city_options_keys = list(city_cdi_map.keys()) if city_cdi_map else ["city_103"] # Fallback if map empty
city_options = classes_or_fallback("city", sorted(city_options_keys))
gender_options = classes_or_fallback("gender", ["Male", "Female", "Other"])
relevent_exp_options = classes_or_fallback("relevent_experience", ["Has relevent experience", "No relevent experience"])
enrolled_uni_options = classes_or_fallback("enrolled_university", ["no_enrollment", "Full time course", "Part time course"])
education_classes = classes_or_fallback("education_level", ["Graduate", "Masters", "High School", "Phd", "Primary School"])
education_options = sort_education(education_classes)
major_options = classes_or_fallback("major_discipline", ["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"])
experience_classes = classes_or_fallback("experience", ["<1"] + [str(i) for i in range(1, 21)] + [">20"])
experience_options = sort_experience(experience_classes)
last_new_job_options = classes_or_fallback("last_new_job", ["never", "1", "2", "3", "4", ">4"])

# ------------------------------------------------------------------ #
# 4. Helper fungsi                                                   #
# ------------------------------------------------------------------ #
salary_mapping = {
    "Primary School": 3_000_000,
    "High School": 5_000_000,
    "Graduate": 8_000_000,
    "Masters": 12_000_000,
    "Phd": 16_000_000,
}

def parse_experience(exp):
    if pd.isna(exp):
        return np.nan
    exp = str(exp).strip()
    if exp == "<1":
        return 0.5
    if exp == ">20":
        return 21.0
    try:
        return float(exp)
    except ValueError:
        return np.nan

def estimate_salary(level, exp_numeric):
    base = salary_mapping.get(level, 0)
    if pd.isna(exp_numeric) or base == 0:
        return np.nan
    if exp_numeric <= 1:
        return base
    elif 2 <= exp_numeric <= 5:
        return base * 1.10
    elif 6 <= exp_numeric <= 10:
        return base * 1.20
    return base * 1.30

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

city_options_sorted = sorted(city_options, key=natural_key)

def get_risk_level_and_recommendation(prob):
    if prob < 0.45:
        risk_level = "Low Risk"
        training_recommendation = "Low Risk – 15% Segmentasi Alokasi Budget."
    elif prob < 0.70:
        risk_level = "Medium Risk"
        training_recommendation = "Medium Risk – 35% Segmentasi Alokasi Budget."
    else:
        risk_level = "High Risk"
        training_recommendation = "High Risk – 50% Segmentasi Alokasi Budget."
    return risk_level, training_recommendation

# Fungsi untuk melakukan pra-pemrosesan pada DataFrame (untuk batch)
def preprocess_batch_data(df_input, city_map, default_cdi, label_encs, impute_vals, model_feature_cols):
    df = df_input.copy()

    for col, val in impute_vals.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
        else: 
            df[col] = val

    df["city_development_index"] = df["city"].apply(lambda x: city_map.get(str(x), default_cdi))
    df["city_development_index"] = np.log1p(df["city_development_index"].astype(float))
    df["experience_numeric"] = df["experience"].apply(parse_experience)

    if "estimated_salary" in model_feature_cols or "estimated_salary" in impute_vals:
        df["estimated_salary"] = df.apply(lambda r: estimate_salary(r.get("education_level"), r.get("experience_numeric")), axis=1)
        if "estimated_salary" in impute_vals and df["estimated_salary"].isnull().any():
            df["estimated_salary"] = df["estimated_salary"].fillna(impute_vals["estimated_salary"])

    for col, enc in label_encs.items():
        if col in df.columns:
            df[col] = df[col].astype(str) 
            transformed_values = []
            default_transformed_value = enc.transform([enc.classes_[0]])[0] if len(enc.classes_) > 0 else 0
            for item in df[col]:
                try:
                    transformed_values.append(enc.transform([item])[0])
                except ValueError: 
                    transformed_values.append(default_transformed_value)
            df[col] = transformed_values
    
    for m_col in model_feature_cols:
        if m_col not in df.columns:
            df[m_col] = 0 
            st.warning(f"Kolom '{m_col}' tidak ditemukan di CSV dan ditambahkan dengan nilai default 0.")

    df_processed = df[model_feature_cols].astype(float).fillna(0)
    return df_processed

# ------------------------------------------------------------------ #
# 5. UI : Form input (REFINED)                                       #
# ------------------------------------------------------------------ #
st.title("💼 Prediksi Kemungkinan Karyawan Resign")
st.markdown("---")

tab1, tab2 = st.tabs(["👤 Prediksi Tunggal", "📂 Prediksi Batch dari CSV"])

with tab1:
    st.markdown("## 📝 Input Data Karyawan (Tunggal)")
    with st.expander("📜 Lihat Daftar Kota & CDI (klik untuk buka)", expanded=False):
        city_cdi_df = pd.DataFrame([
            {"City": k, "CDI": v}
            for k, v in city_cdi_map.items()
        ]).sort_values("CDI", ascending=False)
        st.dataframe(city_cdi_df, hide_index=True, use_container_width=True)
        st.info("Tabel diurutkan dari CDI tertinggi ke terendah.")

    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("Kota", city_options_sorted, help="Pilih kota domisili karyawan.", key="city_single")
        cdi_value = float(city_cdi_map.get(city, DEFAULT_CDI_FALLBACK))
        st.text_input(
            "City Development Index (CDI)",
            value=f"{cdi_value:.3f}",
            disabled=True,
            help="Indeks pembangunan kota (otomatis sesuai kota).",
            key="cdi_single"
        )
        gender = st.selectbox("Jenis Kelamin", gender_options, key="gender_single")
        education_level = st.selectbox("Tingkat Pendidikan", education_options, key="edu_single")
        major = st.selectbox("Jurusan", major_options, key="major_single")
    with col2:
        enrolled_uni = st.selectbox("Status Universitas", enrolled_uni_options, key="uni_single")
        exp_str = st.selectbox("Pengalaman Kerja (tahun)", experience_options, key="exp_single")
        relevent_exp = st.selectbox("Pengalaman Relevan", relevent_exp_options, key="relexp_single")
        last_new_job_str = st.selectbox("Terakhir Ganti Pekerjaan", last_new_job_options, key="lnj_single")

    with st.expander("📋 Lihat Ringkasan Input (Live Update)", expanded=False):
        st.table(pd.DataFrame([{
            "Kota": city, "CDI": f"{cdi_value:.3f}", "Jenis Kelamin": gender,
            "Pendidikan": education_level, "Jurusan": major, "Pengalaman": exp_str,
            "Pengalaman Relevan": relevent_exp, "Status Universitas": enrolled_uni,
            "Last New Job": last_new_job_str
        }]))

    st.markdown("---")
    prediksi_btn_single = st.button("📊 Prediksi Karyawan Ini")

    if prediksi_btn_single:
        raw_single = {
            "city": city, "city_development_index": cdi_value, "gender": gender,
            "relevent_experience": relevent_exp, "enrolled_university": enrolled_uni,
            "education_level": education_level, "major_discipline": major,
            "experience": exp_str, "last_new_job": last_new_job_str,
        }
        df_single = pd.DataFrame([raw_single])

        for col, val in imputation_defaults.items():
            if col in df_single.columns:
                df_single[col] = df_single[col].fillna(val)

        df_single["city_development_index"] = np.log1p(df_single["city_development_index"].astype(float))
        df_single["experience_numeric"] = df_single["experience"].apply(parse_experience)
        if "estimated_salary" in model_cols or "estimated_salary" in imputation_defaults:
            df_single["estimated_salary"] = df_single.apply(lambda r: estimate_salary(r["education_level"], r["experience_numeric"]), axis=1)
            if "estimated_salary" in imputation_defaults and df_single["estimated_salary"].isnull().any():
                df_single["estimated_salary"] = df_single["estimated_salary"].fillna(imputation_defaults["estimated_salary"])

        for col, enc in label_encoders.items():
            if col in df_single.columns:
                try:
                    df_single[col] = enc.transform(df_single[col].astype(str))
                except ValueError: 
                    st.warning(f"Nilai '{df_single[col].iloc[0]}' untuk kolom '{col}' tidak dikenali encoder. Menggunakan nilai default.")
                    df_single[col] = enc.transform([enc.classes_[0]])[0] 

        missing_model_cols = [c for c in model_cols if c not in df_single.columns]
        if missing_model_cols:
            for m_col in missing_model_cols:
                df_single[m_col] = 0 
                st.warning(f"Kolom model '{m_col}' tidak ada dalam input dan diisi dengan 0.")

        df_model_single = df_single[model_cols].astype(float).fillna(0)
        st.write("### Debug: Fitur Akhir untuk Model (Tunggal)", df_model_single)

        y_pred_single = pipeline_model.predict(df_model_single)[0]
        y_proba_res_single = pipeline_model.predict_proba(df_model_single)[0][1]
        risk_level_single, training_recommendation_single = get_risk_level_and_recommendation(y_proba_res_single)

        if y_pred_single == 1: st.toast("⚠️ Karyawan diprediksi akan RESIGN!", icon="⚠️")
        else: st.toast("✅ Karyawan diprediksi TIDAK resign.", icon="✅")

        risk_color_single = "#22c55e" if risk_level_single == "Low Risk" else "#ffb100" if risk_level_single == "Medium Risk" else "#dc2626"
        risk_emoji_single = "🟢" if risk_level_single == "Low Risk" else "🟧" if risk_level_single == "Medium Risk" else "🔴"

        st.markdown(f"""
        <div style="background-color: #1e293b; border-radius: 14px; border-left: 8px solid {risk_color_single}; margin-top: 1.5rem; margin-bottom: 1rem; padding: 2rem 1.5rem 1rem 2rem; box-shadow: 0 4px 24px rgba(0,0,0,0.12);">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 18px; font-weight: bold; color: #fff;">📉 Probabilitas Resign</div>
                    <div style="font-size: 38px; font-weight: bold; color: {risk_color_single}; margin-top: 0.25em;">{y_proba_res_single:.2%}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 18px; font-weight: bold; color: #fff;">🧩 Segmentasi Risiko</div>
                    <div style="font-size: 28px; font-weight: bold; color: {risk_color_single}; margin-top: 0.25em;">{risk_emoji_single} {risk_level_single}</div>
                </div>
            </div>
            <hr style="border: none; border-top: 2px solid #334155; margin: 1.3em 0;">
            <div style="font-size: 20px; font-weight: bold; color: #fff; margin-bottom: 0.7em;">⏱️ Rekomendasi Tindakan</div>
            <div style="background: #334155; border-radius: 8px; padding: 0.8em 1.2em; color: #e0eefa; font-size: 18px; font-weight: 500; margin-bottom: 0.5em;">💡 {training_recommendation_single}</div>
        </div>""", unsafe_allow_html=True)

        def render_progress_bar(percentage, risk_level_text):
            color = "#22c55e" if risk_level_text == "Low Risk" else "#ffb100" if risk_level_text == "Medium Risk" else "#dc2626"
            st.markdown(f"""
            <div style="margin-top:1em">
                <div style="font-weight:600; color:#fff; margin-bottom:0.3em;">📉 Probabilitas Resign: {percentage:.2f}%</div>
                <div style="background-color:#334155; border-radius:6px; height:24px; width:100%;">
                    <div style="background-color:{color}; width:{percentage}%; height:100%; border-radius:6px; text-align:right; padding-right:10px; color:white; font-weight:600; line-height:24px;">{percentage:.0f}%</div>
                </div>
            </div>""", unsafe_allow_html=True)
        render_progress_bar(y_proba_res_single * 100, risk_level_single)

        with st.expander("📈 Insight Strategis Training & Development"):
            st.markdown("""
            - 🔴 **High Risk**: naik **+16.7 pp** → fokus pada upaya preventif intensif
            - 🟧 **Medium Risk**: naik **+1.7 pp** → menjaga momentum pengembangan
            - 🟢 **Low Risk**: turun **–18.3 pp** → alokasi efisien karena risiko rendah
            📌 Dengan redistribusi ini, proporsi anggaran pelatihan kini **lebih mencerminkan prioritas retensi**.
            """)

with tab2:
    st.markdown("## 📤 Unggah CSV untuk Prediksi Batch")
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    REQUIRED_RAW_COLS_BATCH = [
        "city", "gender", "relevent_experience", "enrolled_university",
        "education_level", "major_discipline", "experience", "last_new_job"
    ]

    if uploaded_file is not None:
        try:
            df_batch_raw = pd.read_csv(uploaded_file)
            st.write("### Pratinjau Data CSV yang Diunggah (5 baris pertama):")
            st.dataframe(df_batch_raw.head())

            missing_csv_cols = [col for col in REQUIRED_RAW_COLS_BATCH if col not in df_batch_raw.columns]
            if missing_csv_cols:
                st.error(f"File CSV yang diunggah kekurangan kolom berikut yang wajib ada: {', '.join(missing_csv_cols)}")
                st.stop()

            if st.button("🚀 Proses Prediksi Batch untuk File CSV Ini"):
                with st.spinner("Sedang memproses prediksi batch... Ini mungkin memakan waktu beberapa saat."):
                    output_cols_to_keep = [col for col in df_batch_raw.columns if col not in model_cols and col not in ["experience_numeric", "estimated_salary", "city_development_index"]]
                    if 'employee_id' in df_batch_raw.columns and 'employee_id' not in output_cols_to_keep:
                        output_cols_to_keep.append('employee_id')
                    elif 'enrollee_id' in df_batch_raw.columns and 'enrollee_id' not in output_cols_to_keep: 
                        output_cols_to_keep.append('enrollee_id')

                    df_batch_processed = preprocess_batch_data(
                        df_batch_raw, city_cdi_map, DEFAULT_CDI_FALLBACK,
                        label_encoders, imputation_defaults, model_cols
                    )
                    st.write("### Debug: Fitur Akhir untuk Model (Batch)", df_batch_processed.head())

                    batch_predictions = pipeline_model.predict(df_batch_processed)
                    batch_probabilities = pipeline_model.predict_proba(df_batch_processed)[:, 1]

                    results_df = df_batch_raw[output_cols_to_keep].copy()
                    if not any(id_col in results_df.columns for id_col in ['employee_id', 'enrollee_id']):
                        results_df['Record_ID'] = results_df.index

                    results_df['Probabilitas_Resign'] = batch_probabilities
                    results_df['Prediksi_Resign_Code'] = batch_predictions
                    results_df['Prediksi_Resign_Label'] = results_df['Prediksi_Resign_Code'].apply(lambda x: "YA (Resign)" if x == 1 else "TIDAK (Tidak Resign)")

                    risk_levels_batch = []
                    recommendations_batch = []
                    for prob in batch_probabilities:
                        rl, rec = get_risk_level_and_recommendation(prob)
                        risk_levels_batch.append(rl)
                        recommendations_batch.append(rec)

                    results_df['Segmentasi_Risiko'] = risk_levels_batch
                    results_df['Rekomendasi_Tindakan'] = recommendations_batch
                    
                    st.success("Prediksi batch selesai!")
                    st.write("### Hasil Prediksi Batch:")
                    
                    display_cols_results = [col for col in results_df.columns if col not in ['Prediksi_Resign_Code']]
                    st.dataframe(results_df[display_cols_results], use_container_width=True)

                    # --- PIE CHART SECTION ---
                    st.markdown("---") # Pemisah visual
                    st.write("#### Distribusi Segmentasi Risiko Karyawan (Batch)")
                    
                    risk_counts = results_df['Segmentasi_Risiko'].value_counts().reindex(['Low Risk', 'Medium Risk', 'High Risk'], fill_value=0)
                    # risk_labels digunakan untuk memastikan urutan dan kelengkapan, sudah dicover oleh reindex
                    # risk_labels = ['Low Risk', 'Medium Risk', 'High Risk'] 
                    risk_colors = ['#22c55e', '#ffb100', '#dc2626'] # Low, Medium, High
                    
                    if not risk_counts.empty:
                        fig, ax = plt.subplots(figsize=(6, 5)) # Sedikit penyesuaian figsize
                        ax.pie(
                            risk_counts, 
                            labels=[f"{label} ({count})" for label, count in zip(risk_counts.index, risk_counts)], 
                            autopct='%1.1f%%',
                            colors=risk_colors,
                            startangle=90, # Mulai dari atas untuk Low Risk jika ada
                            wedgeprops={'edgecolor': 'white'} # Garis tepi antar slice
                        )
                        # ax.set_title("Distribusi Segmentasi Risiko (Batch Prediction)", fontsize=13) # Judul dipindah ke st.write
                        ax.axis('equal') # Pastikan pie chart bulat
                        st.pyplot(fig)
                    else:
                        st.info("Tidak ada data risiko untuk ditampilkan dalam pie chart.")
                    # --- END OF PIE CHART SECTION ---
                                        
                    csv_export = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Unduh Hasil Prediksi Batch (CSV)",
                        data=csv_export,
                        file_name="hasil_prediksi_resign_batch.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file CSV: {e}")
            st.error("Pastikan format CSV benar dan semua kolom yang dibutuhkan ada.")
