
import streamlit as st
import re

# --- Theme Colors ---
EEP_TEAL = "#214457"      # Primary Color (Deep Teal)
ACCENT_BLUE = "#4fa5d8"   # Accent Color (Blue)
BACKGROUND = "#FFFFFF"    # Background (White)
SUCCESS_COLOR = "#388e3c" # Success
DANGER_COLOR = "#e64a19"  # Danger/Error
BTN_GREEN_TEAL = "#2e7d32"  # Button Green-Teal

# --- Page Config ---
st.set_page_config(page_title="SehatAI Login", layout="centered", initial_sidebar_state="collapsed")

# --- Custom CSS ---
css = f"""
<style>
.stApp {{
    background-color: {BACKGROUND};
    font-family: 'Inter', sans-serif;
    padding-top: 0px !important;
}}

/* Login Card */
.login-card {{
    background-color: white;
    padding: 25px;
    border-radius: 1rem;
    box-shadow: 0 8px 20px rgba(33, 68, 87, 0.15);
    max-width: 380px;
    margin: 30px auto;
    text-align: center;
}}

/* Header */
.main-header {{
    color: {EEP_TEAL};
    font-size: 1.75rem;
    font-weight: 800;
    margin-bottom: 10px;
}}

/* Input Labels */
div[data-testid="stForm"] label p {{
    color: {EEP_TEAL} !important;
    font-weight: 600;
    font-size: 0.875rem;
}}

/* Input Fields */
div[data-testid="stTextInput"] > div > div > input {{
    border-radius: 0.75rem;
    border: 1px solid #ccc;
    padding: 0.6rem 1rem;
}}
div[data-testid="stTextInput"] > div > div > input:focus {{
    border-color: {ACCENT_BLUE} !important;
    box-shadow: 0 0 0 2px {ACCENT_BLUE} !important;
}}

/* Button */
div[data-testid="stForm"] .stButton > button {{
    width: 100%;
    background-color:{SUCCESS_COLOR} !important;
    color: white !important;
    font-weight: 700;
    border-radius: 0.75rem;
    border: none;
    padding: 0.7rem 0.5rem;
}}
div[data-testid="stForm"] .stButton > button:hover {{
    background-color: #256629 !important;
}}
</style>
"""


# --- Logos Row ---
col_logo1, col_logo2, col_logo3, col_logo4 = st.columns([1, 1, 1, 1])
with col_logo1:
    st.image("pages/Ministry_of_Planning_Development_&_Special_Initiatives_Logo.png", width=150) 
with col_logo2:
    st.image("pages/uran.jpeg", width=120) 
with col_logo3:
    st.image("pages/logo_2.jpeg", width=120) 
with col_logo4:
    st.image("pages/flower.png", width=80) 

st.markdown('<div style="border-bottom: 1px solid #eee; margin: 15px 0;"></div>', unsafe_allow_html=True)

# --- Login Card ---
st.markdown('<div class="login-card">', unsafe_allow_html=True)
st.markdown(f'<h1 class="main-header">Sehat AI Login Portal </h1>', unsafe_allow_html=True)

with st.form("login_form"):
    name = st.text_input("Username", placeholder="Enter your name")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    submitted = st.form_submit_button("Login")

# --- Login Handling ---
if submitted:
    if not name or not password:
        st.error("Please enter both username and password.", icon="🚨")
    else:
        st.success(f"Welcome, **{name}**!", icon="✅")
        # Redirect to home.py
        st.switch_page("pages/home.py")

st.markdown('</div>', unsafe_allow_html=True)
