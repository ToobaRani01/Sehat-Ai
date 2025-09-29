import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os, base64, time

# Load Model
@st.cache_resource
def load_pneumonia_model():
    return load_model("PnemoniaClassifier/Pneumonia_Classifier_Model.h5")

model = load_pneumonia_model()
st.set_page_config(page_title="Pneumonia Detection", layout="wide")

#  Theme Colors 
DEEP_TEAL   = "#214457"
ACCENT_BLUE = "#4fa5d8"
SOFT_TEAL   = "#bdd9cd"
BACKGROUND  = "#E7F1FB"
SUCCESS     = "#388e3c"
DANGER      = "#e64a19"

# Sidebar Logo Helper
def _get_logo_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

logo_path = "pages/logo.jpeg"
logo_b64  = _get_logo_base64(logo_path)
sidebar_logo_img_tag = f'<img class="sidebar-logo" src="data:image/jpeg;base64,{logo_b64}" alt="Logo"/>' if logo_b64 else ""

#  CSS
cache_buster = str(int(time.time()))
st.markdown(f"""
<style id="custom-style-{cache_buster}">
:root {{
    --deep-teal: {DEEP_TEAL};
    --accent: {ACCENT_BLUE};
    --soft-teal: {SOFT_TEAL};
    --bg: {BACKGROUND};
    --success: {SUCCESS};
    --danger: {DANGER};
}}
body {{
    background-color: var(--bg);
    font-family: 'Segoe UI', sans-serif;
}}
/* Hide default nav and fix sidebar identical to About.py */
[data-testid="stSidebarNav"] > ul {{ display:none; }}
[data-testid="stSidebar"] {{
    background-color: var(--deep-teal);
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
    background: var(--accent);
    color: #FFFFFF !important;
}}
.header-bar {{
    background: linear-gradient(90deg, var(--accent), var(--soft-teal));
    padding: 24px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
}}
.header-bar h1 {{
    color: var(--deep-teal);
    font-size: 42px;
    font-weight: 900;
    margin: 0;
}}
.stButton>button {{
    background: var(--deep-teal);
    color: #ffffff;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 24px;
    border: none;
    transition: background 0.25s;
}}
.stButton>button:hover {{
    background: var(--success);
}}
.result-box {{
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-top: 25px;
    font-size: 20px;
    font-weight: 600;
}}
.normal-result {{
    background-color: #ebfff1;
    color: var(--success);
    border: 2px solid var(--success);
}}
.pneumonia-result {{
    background-color: #ffebeb;
    color: var(--danger);
    border: 2px solid var(--danger);
}}
.stImage > img {{
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
}}
.stFileUploader {{
    background-color: #ffffff;
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}}
</style>
""", unsafe_allow_html=True)

#  Sidebar
with st.sidebar:
    st.markdown(
        f"""
        <a class="sidebar-header" href="/" target="_self">
            {sidebar_logo_img_tag}
            <span class="home-text">HOME</span>
        </a>
        """, unsafe_allow_html=True)
    st.markdown("<div class='sidebar-divider' style='border-top:1px solid rgba(240,248,255,0.18);margin-bottom:14px'></div>", unsafe_allow_html=True)
    
    st.markdown('<a class="sidebar-link" href="Pneumonia_Detection" target="_self"><img src="https://img.icons8.com/ios-filled/50/ffffff/lungs.png"/> Pneumonia Detection</a>', unsafe_allow_html=True)
    st.markdown('<a class="sidebar-link" href="Prescription_OCR" target="_self"><img src="https://img.icons8.com/ios-filled/50/ffffff/pill.png"/> Prescription OCR</a>', unsafe_allow_html=True)
    st.markdown('<a class="sidebar-link" href="Diabetes_Risk" target="_self"><img src="https://img.icons8.com/ios-filled/50/ffffff/diabetes.png"/> Diabetes Detection</a>', unsafe_allow_html=True)
    st.markdown('<a class="sidebar-link" href="Info" target="_self"><img src="https://img.icons8.com/ios-filled/50/ffffff/info.png"/> Info</a>', unsafe_allow_html=True)
    st.markdown('<a class="sidebar-link" href="About" target="_self"><img src="https://img.icons8.com/ios-filled/50/ffffff/about.png"/> About</a>', unsafe_allow_html=True)

#  Page Header
st.markdown("""
<div class="header-bar">
    <h1>Pneumonia Detection (X-ray)</h1>
</div>
""", unsafe_allow_html=True)

#  Image Preprocessing
def preprocess_image(img):
    img = img.resize((200, 200))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    arr = np.expand_dims(arr, axis=-1)
    return arr

# File Upload and Prediction
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"], key="uploader")

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded X-ray", width=400)

    if st.button("Run Prediction"):
        input_img = preprocess_image(image)
        prediction = model.predict(input_img)
        pred_class = "Pneumonia Detected" if prediction[0][0] < 0.5 else "Normal Lungs"

        if pred_class == "Pneumonia Detected":
            st.markdown(
                f'<div class="result-box pneumonia-result"><h3>{pred_class}</h3>'
                f'<p>Please consult a healthcare professional for a proper diagnosis and treatment.</p></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box normal-result"><h3>{pred_class}</h3></div>',
                unsafe_allow_html=True
            )
