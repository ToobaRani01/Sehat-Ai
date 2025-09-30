import streamlit as st
import os
import base64
import time

st.set_page_config(page_title="SehatAI Info", layout="wide")

# Helper to get logo 
def _get_logo_base64(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

logo_path = "pages/logo.jpeg"
logo_b64 = _get_logo_base64(logo_path)

cache_buster = str(int(time.time()))

# Sidebar Diabetes SVG 
DIABETES_SIDEBAR_SVG = """
<svg width="18" height="18" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path fill="#FFFFFF" d="M12 2C8 6 5 9.58 5 13.5 5 17.09 8.41 20 12 20s7-2.91 7-6.5C19 9.58 16 6 12 2z"/>
</svg>
"""

# CSS Styling 
st.markdown(f"""
<style id="custom-style-{cache_buster}">
    :root {{
        --bg: #E7F1FB;
        --deep-teal: #214457;
        --accent: #4fa5d8;
        --soft-teal: #bdd9cd;
        --card-bg: #FFFFFF;
    }}

    body {{
        background-color: var(--bg);
        font-family: 'Segoe UI', sans-serif;
    }}

    [data-testid="stSidebarNav"] > ul {{
        display: none;
    }}

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
        display: inline-block;
    }}
    .sidebar-header .home-text {{
        color: #F0F8FF;
        font-weight: 800;
        letter-spacing: 0.6px;
    }}
    .sidebar-divider {{
        border-top: 1px solid rgba(240,248,255,0.18);
        margin: 0 0 14px 0;
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
    .sidebar-link:hover {{
        background: var(--accent);
        color: #FFFFFF !important;
    }}
    .sidebar-link img {{
        width: 18px;
        height: 18px;
        object-fit: contain;
    }}
    .sidebar-link svg {{
        display: inline-block;
        vertical-align: middle;
    }}

    /* Info page title bar with logo */
    .info-title-bar {{
        background: linear-gradient(90deg, var(--accent), var(--soft-teal));
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }}
    .info-title-bar img {{
        width: 80px;
        height: 80px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
    }}
    .info-title-bar h1 {{
        color: var(--deep-teal);
        font-size: 42px;
        font-weight: 800;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }}
    .card {{
        background: var(--card-bg);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.28s ease, box-shadow 0.28s ease;
    }}
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0px 10px 26px rgba(0,0,0,0.12);
    }}
    .card h2 {{
        color: var(--deep-teal);
        margin-top: 0;
    }}
    .card p {{
        color: #333;
        font-size: 16px;
        line-height: 1.6;
    }}
    .feature-icon {{
        width: 48px;
        height: 48px;
        margin-bottom: 10px;
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar
if logo_b64:
    sidebar_logo_img_tag = f'<img class="sidebar-logo" src="data:image/jpeg;base64,{logo_b64}" alt="Logo"/>'
else:
    sidebar_logo_img_tag = ""

with st.sidebar:
    st.markdown(
        f"""
        <a class="sidebar-header" href="/" target="_self">
            {sidebar_logo_img_tag}
            <span class="home-text">HOME</span>
        </a>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    
    st.markdown(
        '<a class="sidebar-link" href="Pneumonia_Detection" target="_self">'
        '<img src="https://img.icons8.com/ios-filled/50/ffffff/lungs.png" alt="Lungs"/> Pneumonia Detection</a>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<a class="sidebar-link" href="Prescription_OCR" target="_self">'
        '<img src="https://img.icons8.com/ios-filled/50/ffffff/pill.png" alt="Pill"/> Prescription OCR</a>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<a class="sidebar-link" href="Diabetes_Risk" target="_self">{DIABETES_SIDEBAR_SVG}Diabetes Risk Detection</a>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<a class="sidebar-link" href="Info" target="_self">'
        '<img src="https://img.icons8.com/ios-filled/50/ffffff/info.png" alt="Info"/> Info</a>',
        unsafe_allow_html=True)
    st.markdown(
        '<a class="sidebar-link" href="About" target="_self">'
        '<img src="https://img.icons8.com/ios-filled/50/ffffff/about.png" alt="About"/> About</a>',
        unsafe_allow_html=True
    )

# Info Page Title Bar 
logo_html = f'<img src="data:image/jpeg;base64,{logo_b64}" alt="Logo"/>' if logo_b64 else ""
st.markdown(f"""
<div class="info-title-bar">
    {logo_html}
    <h1>About SehatAI</h1>
    <p>Your smart AI-powered health assistant.</p>
</div>
""", unsafe_allow_html=True)

# Info Content
features = [
    {
        "icon": "https://img.icons8.com/color/96/lungs.png",
        "title": "Pneumonia Detection",
        "text": "SehatAI analyzes chest X-rays to detect pneumonia quickly. Results are clear and color-coded."
    },
    {
        "icon": "https://img.icons8.com/color/96/diabetes.png",
        "title": "Diabetes Risk Assessment",
        "text": "Enter health values such as glucose, insulin, BMI, and age. Get an instant personalized risk score with easy-to-read charts ."
    },
    {
        "icon": "https://img.icons8.com/color/96/pill.png",
        "title": "Digital Prescription",
        "text": "Upload handwritten prescriptions in Urdu or English. SehatAI converts them into clean digital text, keeping your records organized."
    }
]

for feature in features:
    st.markdown(f"""
    <div class="card">
        <img src="{feature['icon']}" class="feature-icon"/>
        <h2>{feature['title']}</h2>
        <p>{feature['text']}</p>
    </div>
    """, unsafe_allow_html=True)

# Additional Info
st.markdown("""
<div class="card">
<h2>How SehatAI Works</h2>
<p>Simply upload your X-ray, prescription, or health details. Our AI models (CNN for pneumonia, XGBoost for diabetes, OCR for prescriptions) process the data in seconds. Results are displayed with easy-to-read charts and suggestions.</p>
</div>
<div class="card">
<h2>Why SehatAI Matters</h2>
<p>Millions in rural areas lack access to timely diagnostics. SehatAI bridges this gap by providing fast, accessible, bilingual, and visually clear health insights directly from the browser.</p>
</div>
<div class="card">
<h2>Limitations</h2>
<p>SehatAI provides supportive insights but is not a replacement for doctors. Accuracy depends on input quality. Some handwritten prescriptions may not be read correctly. Currently, only Pneumonia and Diabetes are covered.</p>
</div>
<div class="card">
<h2>Future Plans</h2>
<p>Mobile app offline support, more languages, integration with NADRA e-Health systems, epidemic prediction tools, and more disease detection modules.</p>
</div>
<div class="card">
<h2>Important Note</h2>
<p>SehatAI is a support tool, not a replacement for a doctor. Always consult a medical professional for final advice and treatment.</p>
</div>

<div class="card">
<h2>Support</h2>
<p>Special thanks to tech4health organization for providing us with extremely valuable domain insights and supporting us for the project</p>
</div>

""", unsafe_allow_html=True)
