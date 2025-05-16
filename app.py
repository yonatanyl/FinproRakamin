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

city_options = classes_or_fallback("city", sorted(list(city_cdi_map.keys()) or ["city_103"]))
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
# 5. UI : Form input (REFINED)                                       #
# ------------------------------------------------------------------ #
st.title("💼 Prediksi Kemungkinan Karyawan Resign")
st.markdown("## 📝 Form Input Data Karyawan")
st.divider()

with st.form("prediksi_resign_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        city = st.selectbox(
            "Kota",
            city_options,
            help="Pilih kota domisili karyawan."
        )
        gender = st.selectbox(
            "Jenis Kelamin",
            gender_options,
            help="Pilih gender karyawan."
        )
        education_level = st.selectbox(
            "Tingkat Pendidikan",
            education_options,
            help="Pilih pendidikan terakhir."
        )
        exp_str = st.selectbox(
            "Pengalaman Kerja (tahun)",
            experience_options,
            help="Total tahun pengalaman kerja."
        )
    with col2:
        cdi_value = float(city_cdi_map.get(city, DEFAULT_CDI_FALLBACK))
        st.text_input(
            "City Development Index (CDI)",
            value=f"{cdi_value:.3f}",
            disabled=True,
            help="Indeks pembangunan kota (otomatis sesuai kota)."
        )
        relevent_exp = st.selectbox(
            "Pengalaman Relevan",
            relevent_exp_options,
            help="Apakah pengalaman kerja relevan dengan posisi?"
        )
        major = st.selectbox(
            "Jurusan",
            major_options,
            help="Jurusan pendidikan terakhir."
        )
    with col3:
        last_new_job_str = st.selectbox(
            "Terakhir Ganti Pekerjaan",
            last_new_job_options,
            help="Waktu terakhir kali pindah kerja."
        )
        enrolled_uni = st.selectbox(
            "Status Universitas",
            enrolled_uni_options,
            help="Status universitas saat ini."
        )
    st.markdown("---")
    with st.expander("📋 Lihat Ringkasan Input"):
        st.table(pd.DataFrame([{
            "Kota": city,
            "CDI": f"{cdi_value:.3f}",
            "Jenis Kelamin": gender,
            "Pendidikan": education_level,
            "Jurusan": major,
            "Pengalaman": exp_str,
            "Pengalaman Relevan": relevent_exp,
            "Status Universitas": enrolled_uni,
            "Last New Job": last_new_job_str
        }]))

    submitted = st.form_submit_button("📊 Prediksi")

st.divider()


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

    if y_proba_res < 0.45:
        risk_level = "Low Risk"
        training_recommendation = "Low Risk – 15% Segmentasi Alokasi Budget."
    elif y_proba_res < 0.70:
        risk_level = "Medium Risk"
        training_recommendation = "Medium Risk – 35% Segmentasi Alokasi Budget."
    else:
        risk_level = "High Risk"
        training_recommendation = "High Risk – 50% Segmentasi Alokasi Budget."

# ---- ANIMASI/NOTIFIKASI INTERAKTIF ----
    if y_pred == 1:
        st.toast("⚠️ Karyawan diprediksi akan RESIGN!", icon="⚠️")
    else:
        st.toast("✅ Karyawan diprediksi TIDAK resign.", icon="✅")
    
    # ---- KARTU UTAMA HASIL PREDIKSI ----
    risk_color = "#22c55e" if risk_level == "Low Risk" else "#ffb100" if risk_level == "Medium Risk" else "#dc2626"
    risk_emoji = "🟢" if risk_level == "Low Risk" else "🟧" if risk_level == "Medium Risk" else "🔴"
    
    st.markdown(f"""
    <div style="
        background-color: #1e293b;
        border-radius: 14px;
        border-left: 8px solid {risk_color};
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding: 2rem 1.5rem 1rem 2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.12);
    ">
        <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 18px; font-weight: bold; color: #fff;">📉 Probabilitas Resign</div>
                <div style="font-size: 38px; font-weight: bold; color: {risk_color}; margin-top: 0.25em;">{y_proba_res:.2%}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 18px; font-weight: bold; color: #fff;">🧩 Segmentasi Risiko</div>
                <div style="font-size: 28px; font-weight: bold; color: {risk_color}; margin-top: 0.25em;">
                    {risk_emoji} {risk_level}
                </div>
            </div>
        </div>
        <hr style="border: none; border-top: 2px solid #334155; margin: 1.3em 0;">
        <div style="font-size: 20px; font-weight: bold; color: #fff; margin-bottom: 0.7em;">
            ⏱️ Rekomendasi Tindakan
        </div>
        <div style="
            background: #334155; 
            border-radius: 8px; 
            padding: 0.8em 1.2em;
            color: #e0eefa;
            font-size: 18px;
            font-weight: 500;
            margin-bottom: 0.5em;
            ">
            💡 {training_recommendation}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- PROGRESS BAR VISUAL ----
    def render_progress_bar(percentage, risk_level):
        color = "#22c55e" if risk_level == "Low Risk" else "#ffb100" if risk_level == "Medium Risk" else "#dc2626"
        st.markdown(f"""
        <div style="margin-top:1em">
            <div style="font-weight:600; color:#fff; margin-bottom:0.3em;">📉 Probabilitas Resign: {percentage:.2f}%</div>
            <div style="background-color:#334155; border-radius:6px; height:24px; width:100%;">
                <div style="
                    background-color:{color};
                    width:{percentage}%;
                    height:100%;
                    border-radius:6px;
                    text-align:right;
                    padding-right:10px;
                    color:white;
                    font-weight:600;
                    line-height:24px;
                ">
                    {percentage:.0f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_progress_bar(y_proba_res * 100, risk_level)
    
    # ----INSIGHT TAMBAHAN T&D ----
    with st.expander("📈 Insight Strategis Training & Development"):
        st.markdown("""
        - 🔴 **High Risk**: naik **+16.7 pp** → fokus pada upaya preventif intensif  
        - 🟡 **Medium Risk**: naik **+1.7 pp** → menjaga momentum pengembangan  
        - 🟢 **Low Risk**: turun **–18.3 pp** → alokasi efisien karena risiko rendah  
    
        📌 Dengan redistribusi ini, proporsi anggaran pelatihan kini **lebih mencerminkan prioritas retensi**.
        """)
        import shap
        # --- SHAP untuk pipeline ---
        explainer = shap.Explainer(pipeline_model, df_model)
        shap_values = explainer(df_model)
        
        # Ambil row prediksi yang barusan (pertama dan satu-satunya di batch ini)
        shap_row = shap_values[0].values
        
        # Rank fitur berdasar kontribusi absolut (besar pengaruh)
        top_idx = np.argsort(np.abs(shap_row))[::-1][:3]  # top 3 fitur
        top_feats = [(df_model.columns[i], shap_row[i]) for i in top_idx]
        
        shap_markdown = "#### 🧠 Alasan Utama Prediksi:\n"
        for fname, sval in top_feats:
            arrow = "⬆️" if sval > 0 else "⬇️"
            efek = "menaikkan risiko" if sval > 0 else "menurunkan risiko"
            shap_markdown += f"- **{fname}** {arrow} {efek} (SHAP: {sval:.3f})\n"
        st.info(shap_markdown)

