import streamlit as st
import os
import base64
import time

# Configuration and Initialization 
st.set_page_config(page_title="SehatAI About Us", layout="wide")

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

#  Sidebar Diabetes SVG (same as home
DIABETES_SIDEBAR_SVG = """
<svg width="18" height="18" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path fill="#FFFFFF" d="M12 2C8 6 5 9.58 5 13.5 5 17.09 8.41 20 12 20s7-2.91 7-6.5C19 9.58 16 6 12 2z"/>
</svg>
"""

# UPDATED Team Member Data
team_members = [
    {"name": "Dr. Dodo Khan", "role": "AI/ML Expert"},
    {"name": "Azka Fatima", "role": "Team Coordinator"},
    {"name": "Tooba Rani", "role": "Software Developer"},
    {"name": "Asiya Parveen", "role": "Software Developer"},
]

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

    /* --- Main Content Title Bar --- */
    .main-page-title-bar {{
        background: linear-gradient(90deg, var(--accent), var(--soft-teal));
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }}
    .main-page-title-bar img {{
        width: 90px;
        height: 90px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }}
    .main-page-title-bar h1 {{
        color: var(--deep-teal);
        font-size: 48px;
        font-weight: 900;
        margin: 0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
    }}
    .main-page-title-bar p {{
        color: var(--deep-teal);
        font-size: 1.3em;
        font-style: italic;
        margin-top: 10px;
    }}

    /* --- General Card Styling --- */
    .card {{
        background: var(--card-bg);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .card:hover {{
        transform: translateY(-7px);
        box-shadow: 0px 12px 28px rgba(0,0,0,0.15);
    }}
    .card h2 {{
        color: var(--deep-teal);
        margin-top: 0;
        font-size: 28px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .card p {{
        color: #333;
        font-size: 17px;
        line-height: 1.7;
    }}
    
    /* --- Specific Card Styles for Team Members --- */
    .team-card {{
        text-align: center;
        padding: 20px;
        min-height: 160px;
        justify-content: space-between;
    }}
    .team-card h3 {{
        color: var(--accent);
        margin: 0 0 5px 0;
        font-size: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }}
    .team-card p {{
        color: var(--deep-teal);
        font-weight: bold;
        font-size: 18px;
        margin: 0;
    }}

    /* --- Section Headings --- */
    .section-heading {{
        text-align: center;
        color: var(--deep-teal);
        margin: 40px 0 30px 0;
        font-size: 3em;
        font-weight: 700;
        letter-spacing: 1px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.08);
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
        unsafe_allow_html=True
    )
    st.markdown(
        '<a class="sidebar-link" href="About" target="_self">'
        '<img src="https://img.icons8.com/ios-filled/50/ffffff/about.png" alt="About"/> About</a>',
        unsafe_allow_html=True
    )

# Main Page Title Bar
logo_html = f'<img src="data:image/jpeg;base64,{logo_b64}" alt="SehatAI Logo"/>' if logo_b64 else ""
st.markdown(f"""
<div class="main-page-title-bar">
    {logo_html}
    <h1>Welcome to SehatAI!</h1>
    <p>Empowering Health Through Innovation and Accessibility</p>
</div>
""", unsafe_allow_html=True)

# Mission & Vision 
st.markdown("<h2 class='section-heading'>Our Guiding Principles</h2>", unsafe_allow_html=True)

col_mission, col_vision = st.columns(2)

with col_mission:
    st.markdown("""
    <div class="card">
    <h2>Our Mission</h2>
    <p>To revolutionize healthcare accessibility in underserved communities by providing intuitive, AI-powered diagnostic and record-keeping tools, fostering informed health decisions for everyone.</p>
    </div>
    """, unsafe_allow_html=True)

with col_vision:
    st.markdown("""
    <div class="card">
    <h2>Our Vision</h2>
    <p>To be the leading AI-powered health assistant in the region, recognized for accuracy, ease of use, and dedication to improving public health through technology.</p>
    </div>
    """, unsafe_allow_html=True)

# Team Section 
st.markdown("<h2 class='section-heading'>Meet Our Dedicated Team</h2>", unsafe_allow_html=True)

cols = st.columns(2) 
for i, member in enumerate(team_members):
    with cols[i % 2]:
        st.markdown(f"""
        <div class="card team-card">
            <h3>{member['name']}</h3>
            <p>{member['role']}</p>
        </div>
        """, unsafe_allow_html=True)

#  Project Summary / Acknowledgments 
st.markdown("<h2 class='section-heading'>Project Journey & Acknowledgments</h2>", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h2>Project Overview</h2>
    <p>
    SehatAI began with a clear goal to leverage the power of Artificial Intelligence to address critical health challenges. From accurate pneumonia detection using advanced CNN models to intuitive diabetes risk assessment with XGBoost, and the innovative digital transcription of handwritten prescriptions, our project aims to make a tangible impact.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h2>Submission Details</h2>
    <p>
    <b>Project Name:</b> SehatAI – AI-Powered Health Assistant<br>
    <b>Core Technologies:</b> CNN (Pneumonia), XGBoost (Diabetes), OCR (Prescriptions)<br>
    <b>Submitted By:</b> Dr. Dodo Khan, Azka Fatima, Tooba Rani, Asiya Parveen<br>
    <b>Supervisor:</b> Dr. Dodo Khan<br>
    <b>Submitted to:</b> URAAN AI Techathon 1.0<br>
    <b>Status:</b> Prototype/Demonstration for "Tech for Good"
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h2>Special Thanks</h2>
    <p>
    We extend our deepest gratitude to our mentors, the URAAN AI Techathon 1.0 organizers, and everyone who supported us in transforming this vision into a reality. Your guidance and encouragement were invaluable.
    </p>
</div>
""", unsafe_allow_html=True)
