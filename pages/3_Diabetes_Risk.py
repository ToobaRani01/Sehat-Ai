import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os, base64, time

# THEME COLORS
DEEP_TEAL   = "#214457"   # Primary Color
ACCENT_BLUE = "#4fa5d8"   # Accent Color
SOFT_TEAL   = "#bdd9cd"   # Sidebar secondary
BACKGROUND  = "#E7F1FB"   # App background
DANGER_COLOR  = "#e64a19"
WARNING_COLOR = "#f9a825"
SUCCESS_COLOR = "#388e3c"

st.set_page_config(page_title="Diabetes Risk", layout="wide")

# LOAD MODEL 
try:
    diabetes_model = joblib.load("diabates/diabates.pkl")
    MODEL_LOADED = True
except Exception as e:
    st.error(f"ERROR: Failed to load the model. Details: {e}")
    MODEL_LOADED = False
    st.stop()

#  SIDEBAR (MATCH ABOUT.PY)
def _get_logo_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

logo_path = "pages/logo.jpeg"
logo_b64  = _get_logo_base64(logo_path)
sidebar_logo_img_tag = f'<img class="sidebar-logo" src="data:image/jpeg;base64,{logo_b64}" alt="Logo"/>' if logo_b64 else ""

with st.sidebar:
    st.markdown(
        f"""
        <a class="sidebar-header" href="/" target="_self">
            {sidebar_logo_img_tag}
            <span class="home-text">HOME</span>
        </a>
        """, unsafe_allow_html=True
    )
    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown('<a class="sidebar-link" href="Pneumonia_Detection" target="_self"><img src="https://img.icons8.com/ios-filled/50/ffffff/lungs.png"/> Pneumonia Detection</a>', unsafe_allow_html=True)
    st.markdown('<a class="sidebar-link" href="Prescription_OCR" target="_self"><img src="https://img.icons8.com/ios-filled/50/ffffff/pill.png"/> Prescription OCR</a>', unsafe_allow_html=True)
    st.markdown('<a class="sidebar-link" href="Diabetes_Risk" target="_self"><img src="https://img.icons8.com/ios-filled/50/ffffff/diabetes.png"/> Diabetes Risk Detection</a>', unsafe_allow_html=True)
    st.markdown('<a class="sidebar-link" href="Info" target="_self"><img src="https://img.icons8.com/ios-filled/50/ffffff/info.png"/> Info</a>', unsafe_allow_html=True)
    st.markdown('<a class="sidebar-link" href="About" target="_self"><img src="https://img.icons8.com/ios-filled/50/ffffff/about.png"/> About</a>', unsafe_allow_html=True)

# Sidebar CSS identical to About.py
cache_buster = str(int(time.time()))
st.markdown(f"""
<style id="sidebar-style-{cache_buster}">
[data-testid="stSidebarNav"] > ul {{ display: none; }}
[data-testid="stSidebar"] {{
    background-color: {DEEP_TEAL};
    padding: 12px 10px 10px 10px;
}}
.sidebar-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 20px;
    font-weight: 700;
    color: #F0F8FF;
    padding: 8px 12px;
    border-radius: 8px;
    margin: 4px 0 14px 0;
    background-color: rgba(240, 248, 255, 0.08);
    transition: background-color 0.2s ease;
}}
.sidebar-header:hover {{
    background-color: rgba(240, 248, 255, 0.14);
}}
.sidebar-header img.sidebar-logo {{
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: cover;
}}
.sidebar-link {{
    font-size: 16px;
    margin: 8px 0;
    padding: 10px 12px;
    border-radius: 8px;
    transition: background 0.18s ease, color 0.18s ease;
    color: #E0E8F2 !important;
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.sidebar-link img {{
    width: 18px;
    height: 18px;
    object-fit: contain;
}}
.sidebar-link:hover {{
    background: {ACCENT_BLUE};
    color: #FFFFFF !important;
}}
.sidebar-divider {{
    border-top: 1px solid rgba(240,248,255,0.18);
    margin-bottom: 14px;
}}
</style>
""", unsafe_allow_html=True)

# GLOBAL THEME
st.markdown(f"""
<style>
:root {{
    --bg: {BACKGROUND};
    --deep-teal: {DEEP_TEAL};
    --accent: {ACCENT_BLUE};
    --soft-teal: {SOFT_TEAL};
}}
.stApp {{
    background-color: var(--bg);
    color: var(--deep-teal);
}}
.title-box {{
    background-color: var(--deep-teal);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin-bottom: 25px;
}}
.title-box h2 {{
    color: white;
    text-align: center;
    margin: 0;
}}
h1, h2, h3, h4, h5, h6 {{ color: var(--deep-teal); }}
.stForm, div[data-testid*="stAlert"] {{
    border-radius: 16px;
    padding: 20px;
    background-color: white;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    margin-bottom: 20px;
    border: 1px solid var(--soft-teal);
}}
.stNumberInput label, .stSelectbox label, .stTextInput label {{
    font-weight: 600;
    color: var(--deep-teal);
}}
div.stForm button[type="submit"] {{
    background-color: {ACCENT_BLUE} !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: bold !important;
    border: none !important;
    cursor: pointer !important;
    font-size: 16px !important;
    transition: all 0.2s ease-in-out;
}}
div.stForm button[type="submit"]:hover {{
    background-color: #408ec2 !important;
    transform: scale(1.02);
}}
[data-testid="stMetricValue"] {{
    font-size: 3em !important;
    color: {ACCENT_BLUE} !important;
    font-weight: 800;
}}
div[data-testid="stAlert-error"]   {{ border-left: 6px solid {DANGER_COLOR} !important; }}
div[data-testid="stAlert-warning"] {{ border-left: 6px solid {WARNING_COLOR} !important; }}
div[data-testid="stAlert-success"] {{ border-left: 6px solid {SUCCESS_COLOR} !important; }}
</style>
""", unsafe_allow_html=True)

# TITLE BAR 
st.markdown(
    """
    <div class="title-box">
        <h2>Diabetes Risk Detection </h2>
    </div>
    """, unsafe_allow_html=True
)

# USER FORM 
st.subheader("Patient Diagnostic Measurements ")
with st.form(key="diabetes_form"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, value=1)
        blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0, value=72)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, value=25.0, format="%.1f")
        age = st.number_input("Age (Years)", min_value=1, value=33)
    with col2:
        glucose = st.number_input("Plasma Glucose (mg/dL)", min_value=0, value=100)
        insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0, value=80)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, format="%.3f")
    st.markdown("---")
    submit = st.form_submit_button("Predict Risk")

# PREDICTION & RESULTS
if submit and MODEL_LOADED:
    features = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])
    try:
        prob = diabetes_model.predict_proba(features)[0][1]
        pred = diabetes_model.predict(features)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Prediction Results")
    st.metric("Overall Diabetes Risk Score", f"{prob*100:.1f}%")

    if pred == 1:
        st.error(f"High Risk Detected ({prob*100:.1f}%)")
        st.markdown(
            f"""
            <div style="background-color:#ffe0b2; padding:15px; border-radius:10px;
                        color:{DEEP_TEAL}; border:1px solid {DANGER_COLOR};">
            <p style="font-size:1.1em; font-weight:bold; color:{DANGER_COLOR};">
            High Risk Detected: Immediate Action Advised</p>
            <ul>
                <li>Follow a <strong>low-sugar, high-fiber diet</strong>.</li>
                <li>Avoid sugary drinks and processed foods.</li>
                <li>Exercise <strong>30–45 minutes daily</strong>.</li>
                <li>Monitor <strong>blood sugar levels daily</strong>.</li>
                <li>Schedule an <strong>immediate doctor consultation</strong>.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        if 100 <= glucose <= 125:
            st.warning(f"Moderate Risk (Pre-diabetic Glucose Range: {glucose} mg/dL)")
            st.markdown(
                f"""
                <div style="background-color:#fff9c4; padding:15px; border-radius:10px;
                            color:{DEEP_TEAL}; border:1px solid {WARNING_COLOR};">
                <p style="font-size:1.1em; font-weight:bold; color:{WARNING_COLOR};">
                Moderate Risk: Lifestyle Changes are Crucial</p>
                <ul>
                    <li>Adopt a <strong>balanced diet</strong>, reducing refined carbs.</li>
                    <li>Increase <strong>physical activity</strong>.</li>
                    <li>Manage weight and reduce stress.</li>
                    <li>Get regular <strong>medical check-ups</strong>.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.success(f"Low Risk Detected ({prob*100:.1f}%)")
            st.markdown(
                f"""
                <div style="background-color:#e8f5e9; padding:15px; border-radius:10px;
                            color:{DEEP_TEAL}; border:1px solid {SUCCESS_COLOR};">
                <p style="font-size:1.1em; font-weight:bold; color:{SUCCESS_COLOR};">
                Low Risk: Maintain Healthy Habits</p>
                <ul>
                    <li>Continue your <strong>healthy lifestyle</strong>.</li>
                    <li>Ensure a balanced, nutrient-rich diet.</li>
                    <li>Get an <strong>annual health check-up</strong>.</li>
                    <li>Stay active and manage stress.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True
            )

    # VISUAL ANALYSIS
    st.subheader("Visual Analysis")
    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        st.markdown("#### Patient Risk Distribution")
        df_risk = pd.DataFrame({"Condition": ["Diabetes Risk", "Healthy"], "Probability": [prob, 1 - prob]})
        pie_chart = px.pie(
            df_risk, values="Probability", names="Condition", hole=0.5,
            color="Condition",
            color_discrete_map={"Diabetes Risk": DEEP_TEAL, "Healthy": SOFT_TEAL},
            height=300
        )
        pie_chart.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#ffffff', width=2)))
        pie_chart.update_layout(showlegend=True, margin=dict(t=30, b=30, l=30, r=30),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(pie_chart, use_container_width=True)

    with vis_col2:
        st.markdown("#### Input Parameter Distribution")
        input_data = pd.DataFrame([
            {'Parameter': 'Glucose (mg/dL)', 'Value': glucose},
            {'Parameter': 'Blood Pressure (mm Hg)', 'Value': blood_pressure},
            {'Parameter': 'BMI (kg/m²)', 'Value': bmi},
            {'Parameter': 'Insulin (mu U/ml)', 'Value': insulin},
            {'Parameter': 'DPF (score) x100', 'Value': dpf * 100}
        ])
        param_bar = px.bar(
            input_data, y='Parameter', x='Value', orientation='h',
            color='Value',
            color_continuous_scale=[SOFT_TEAL, ACCENT_BLUE, DEEP_TEAL],
            title='Input Values Comparison', height=350
        )
        param_bar.update_layout(
            xaxis_title="Value (Actual)", yaxis_title="",
            coloraxis_showscale=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(param_bar, use_container_width=True)

    st.markdown("#### Glucose Level vs. Health Zones (mg/dL)")
    NORMAL_END = 99
    PREDIABETES_END = 125
    HIGH_END = 200
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.add_patch(Rectangle((0, 0), NORMAL_END, 0.5, color=SUCCESS_COLOR, alpha=0.3))
    ax.add_patch(Rectangle((NORMAL_END, 0), PREDIABETES_END - NORMAL_END, 0.5, color=WARNING_COLOR, alpha=0.3))
    ax.add_patch(Rectangle((PREDIABETES_END, 0), HIGH_END - PREDIABETES_END, 0.5, color=DANGER_COLOR, alpha=0.3))
    ax.barh(y=0, width=glucose, height=0.2, color=ACCENT_BLUE, label="Your Glucose")
    ax.axvline(x=126, color=DEEP_TEAL, linestyle='--', linewidth=2, label="Diabetes Threshold (126 mg/dL)")
    ax.set_xlim(0, HIGH_END)
    ax.set_yticks([])
    ax.set_xlabel("Plasma Glucose Concentration (mg/dL)")
    ax.set_title("Glucose Status Indicator", fontsize=14, color=DEEP_TEAL)
    legend_handles = [
        Rectangle((0, 0), 1, 1, fc=SUCCESS_COLOR, alpha=0.3),
        Rectangle((0, 0), 1, 1, fc=WARNING_COLOR, alpha=0.3),
        Rectangle((0, 0), 1, 1, fc=DANGER_COLOR, alpha=0.3),
        plt.Line2D([0], [0], color=ACCENT_BLUE, lw=4),
        plt.Line2D([0], [0], color=DEEP_TEAL, linestyle='--')
    ]
    legend_labels = ["Normal (<100)", "Pre-Diabetes (100-125)", "High Risk (>125)", "Your Value", "Threshold"]
    ax.legend(legend_handles, legend_labels, loc='upper center',
              bbox_to_anchor=(0.5, -0.4), ncol=5, frameon=False, fontsize=10)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    st.pyplot(fig)
