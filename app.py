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

# ------------------------------------------------------------------ #
# 0. Konfigurasi halaman                                              #
# ------------------------------------------------------------------ #
st.set_page_config(page_title="Prediksi Resign Karyawan", layout="wide", page_icon="💼")

# ------------------------------------------------------------------ #
# 1. Loader artefak                                                   #
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
# 2. Inisialisasi artefak                                             #
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

if pipeline_model is None or not model_cols:
    st.error("Model atau daftar kolom gagal dimuat. Aplikasi berhenti.")
    st.stop()

# ------------------------------------------------------------------ #
# 3. List opsi untuk widget                                           #
# ------------------------------------------------------------------ #
def classes_or_fallback(col, default):
    return list(label_encoders.get(col, type("dummy", (object,), {"classes_": default})) .classes_)

city_options = classes_or_fallback("city", sorted(list(city_cdi_map.keys()) or ["city_103"]))
gender_options = classes_or_fallback("gender", ["Male", "Female", "Other"])
relevent_exp_options = classes_or_fallback("relevent_experience", ["Has relevent experience", "No relevent experience"])
enrolled_uni_options = classes_or_fallback("enrolled_university", ["no_enrollment", "Full time course", "Part time course"])
education_options = classes_or_fallback("education_level", ["Graduate", "Masters", "High School", "Phd", "Primary School"])
major_options = classes_or_fallback("major_discipline", ["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"])
experience_options = classes_or_fallback("experience", ["<1"] + [str(i) for i in range(1, 21)] + [">20"])
last_new_job_options = classes_or_fallback("last_new_job", ["never", "1", "2", "3", "4", ">4"])

# ------------------------------------------------------------------ #
# 4. Helper fungsi                                                    #
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

# ------------------------------------------------------------------ #
# 5. UI : Form input                                                 #
# ------------------------------------------------------------------ #
st.title("💼 Prediksi Kemungkinan Karyawan Resign")

with st.form("prediction_form"):
    ccol1, col2, col3 = st.columns(3)

with col1:
    city = st.selectbox("Kota", city_options, key="city")
    gender = st.selectbox("Jenis Kelamin", gender_options, key="gender")
    education_level = st.selectbox("Tingkat Pendidikan", education_options, key="education_level")
    exp_str = st.selectbox("Pengalaman Kerja (tahun)", experience_options, key="experience")

with col2:
    cdi_value = float(city_cdi_map.get(city, DEFAULT_CDI_FALLBACK))
    st.number_input("City Development Index (CDI)", value=cdi_value, step=0.001, format="%.3f", disabled=True, key="cdi_display")
    relevent_exp = st.selectbox("Pengalaman Relevan", relevent_exp_options, key="relevent_experience")
    major = st.selectbox("Jurusan", major_options, key="major_discipline")

with col3:
    last_new_job_str = st.selectbox("Terakhir Ganti Pekerjaan", last_new_job_options, key="last_new_job")
    enrolled_uni = st.selectbox("Status Universitas", enrolled_uni_options, key="enrolled_university")

submitted = st.button("Prediksi")


# ------------------------------------------------------------------ #
# 6. Pra-proses & Prediksi                                           #
# ------------------------------------------------------------------ #
if submitted:
    raw = {
        "city": city,
        "city_development_index": cdi_value,
        "gender": gender,
        "relevent_experience": relevent_exp,
        "enrolled_university": enrolled_uni,
        "education_level": education_level,
        "major_discipline": major,
        "experience": exp_str,
        "last_new_job": last_new_job_str,
    }

    st.write("### Input Mentah", pd.DataFrame([raw]))

    df = pd.DataFrame([raw])
    for col, val in imputation_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    df["experience_numeric"] = df["experience"].apply(parse_experience)
    df["estimated_salary"] = df.apply(lambda r: estimate_salary(r["education_level"], r["experience_numeric"]), axis=1)
    df["city_development_index"] = np.log1p(df["city_development_index"])

    for col, enc in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = enc.transform(df[col])
            except ValueError:
                df[col] = enc.transform([enc.classes_[0]])

    missing_cols = [c for c in model_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Kolom hilang: {missing_cols}")
        st.stop()

    df_model = df[model_cols].astype(float).fillna(0)
    st.write("### Fitur Akhir untuk Model", df_model)

    y_pred = pipeline_model.predict(df_model)[0]
    y_proba_res = pipeline_model.predict_proba(df_model)[0][1]

    st.subheader("📊 Hasil Prediksi")
    if y_pred == 1:
        st.error(f"⚠️  Karyawan diprediksi **RESIGN** – probabilitas: {y_proba_res:.2%}")
    else:
        st.success(f"✅  Karyawan diprediksi **TIDAK resign** – probabilitas bertahan: {1 - y_proba_res:.2%}")

    st.info(f"Estimasi gaji (berdasarkan pendidikan & pengalaman): Rp {df['estimated_salary'].iloc[0]:,.0f}")

    if y_proba_res < 0.33:
        training_recommendation = "Low – tidak perlu pelatihan tambahan saat ini."
    elif y_proba_res < 0.66:
        training_recommendation = "Medium – disarankan pelatihan pengembangan keterampilan lanjutan."
    else:
        training_recommendation = "High – sangat disarankan pelatihan intensif (soft skill & career development)."

    st.warning("📌 Rekomendasi Training Hour:")
    st.write(f"💡 Kategori risiko: **{'Low' if y_proba_res < 0.33 else 'Medium' if y_proba_res < 0.66 else 'High'}**")
    st.write(f"🕒 Rekomendasi jam pelatihan: {training_recommendation}")
