import streamlit as st
import os
import base64
import time

st.set_page_config(page_title="SEHAT AI", layout="wide")

# ------------------ Helper Function ------------------
def _get_logo_base64(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

# ------------------ Logo Setup ------------------
logo_path = "pages/logo.jpeg"
logo_b64 = _get_logo_base64(logo_path)

DIABETES_SIDEBAR_SVG = """
<svg width="18" height="18" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path fill="#FFFFFF" d="M12 2C8 6 5 9.58 5 13.5 5 17.09 8.41 20 12 20s7-2.91 7-6.5C19 9.58 16 6 12 2z"/>
</svg>
"""

cache_buster = str(int(time.time()))

# ------------------ CSS Styling ------------------
st.markdown(f"""
<style id="custom-style-{cache_buster}">
    :root {{
        --bg: #E7F1FB;
        --deep-teal: #214457;
        --accent: #4fa5d8;
        --soft-teal: #bdd9cd;
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

    .header-bar {{
        background: linear-gradient(90deg, var(--accent), var(--soft-teal));
        padding: 16px 24px;
        border-radius: 12px;
        margin-bottom: 26px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }}
    .header-logo {{
        height: 80px;
        width: 80px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
    }}
    .header-title {{
        color: var(--deep-teal);
        font-size: 44px;
        margin: 0;
        text-align: center;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.12);
    }}

    .card {{
        background: #FFFFFF;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.28s ease, box-shadow 0.28s ease;
        border-top: 6px solid var(--accent);
        cursor: pointer;
        color: inherit;
        text-decoration: none;
    }}
    .card:hover {{
        transform: translateY(-8px);
        box-shadow: 0px 10px 26px rgba(0,0,0,0.12);
        border-top: 6px solid var(--deep-teal);
    }}
    .card h3 {{
        margin: 12px 0 0;
        font-size: 20px;
        font-weight: 700;
        color: var(--deep-teal);
    }}
    .card img {{
        display: block;
        margin: 0 auto 10px auto;
        border-radius: 50%;
        max-height: 90px;
        width: 90px;
        object-fit: cover;
    }}
</style>
""", unsafe_allow_html=True)

# ------------------ Logo HTML ------------------
if logo_b64:
    sidebar_logo_img_tag = f'<img class="sidebar-logo" src="data:image/jpeg;base64,{logo_b64}" alt="Logo"/>'
    header_logo_html = f'<img class="header-logo" src="data:image/jpeg;base64,{logo_b64}" alt="Logo"/>'
else:
    sidebar_logo_img_tag = ""
    header_logo_html = ""

# ------------------ Sidebar ------------------
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
        unsafe_allow_html=True
    )
    st.markdown(
        '<a class="sidebar-link" href="About" target="_self">'
        '<img src="https://img.icons8.com/ios-filled/50/ffffff/about.png" alt="About"/> About</a>',
        unsafe_allow_html=True
    )

# Header 
st.markdown(f"""
<div class="header-bar">
    {header_logo_html}
    <div class="header-title">SEHAT AI</div>
</div>
""", unsafe_allow_html=True)

# ----------------- Feature Cards -----------------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <a href="Pneumonia_Detection" target="_self" class="card">
        <img src="https://img.icons8.com/color/240/lungs.png" alt="Pneumonia"/>
        <h3>Pneumonia Detection</h3>
    </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <a href="Prescription_OCR" target="_self" class="card">
        <img src="https://img.icons8.com/color/240/pill.png" alt="Prescription OCR"/>
        <h3>Prescription OCR</h3>
    </a>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <a href="Diabetes_Risk" target="_self" class="card">
        <img src="https://img.icons8.com/color/240/diabetes.png" alt="Diabetes Risk"/>
        <h3>Diabetes Risk</h3>
    </a>
    """, unsafe_allow_html=True)
