
# import numpy as np
# import streamlit as st
# import cv2
# from OCR.image_ocr import process_ocr_from_image_bytes, get_medicine_info_and_translate

# # Title Bar
# st.markdown(
#     """
#     <div style="background-color:#16c2d5; padding:15px; border-radius:10px; text-align:center; margin-bottom:20px;">
#         <h2 style="color:white; margin:0;">Prescription Digitization and Validation System</h2>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# # File Upload
# st.header("Digitized Prescription")
# pres_file = st.file_uploader("Upload Prescription Image", type=['jpg', 'png', 'jpeg'])

# # Smarter Blur Detection
# def is_blurry(image_bytes, threshold_low=60, threshold_high=100):
#     np_img = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
#     laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

#     if laplacian_var < threshold_low:
#         return True, laplacian_var, "Too blurry"
#     elif laplacian_var < threshold_high:
#         return False, laplacian_var, "Borderline"
#     else:
#         return False, laplacian_var, "Clear"

# # Noise Detection Function (relaxed)
# def is_noisy(image_bytes, edge_thresh=0.45, freq_thresh=0.35):
#     np_img = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

#     edges = cv2.Canny(img, 100, 200)
#     edge_density = np.sum(edges > 0) / edges.size

#     f = np.fft.fft2(img)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
#     high_freq_score = np.mean(magnitude_spectrum > 150)

#     return edge_density > edge_thresh or high_freq_score > freq_thresh, edge_density, high_freq_score

# # OCR Trigger
# if pres_file:
#     st.image(pres_file, caption='Uploaded Prescription', width=350)

#     if st.button("Perform OCR"):
#         with st.spinner('Checking image quality...'):
#             image_bytes = pres_file.read()

#             blurry, blur_score, blur_status = is_blurry(image_bytes)
#             noisy, edge_score, freq_score = is_noisy(image_bytes)

#             # st.write(f"📊 Blur Score: {blur_score:.2f} ({blur_status})")
#             # st.write(f"📊 Edge Density: {edge_score:.3f}")
#             # st.write(f"📊 High-Frequency Score: {freq_score:.3f}")

#             if blurry or noisy:
#                 st.error("❌ Image quality is insufficient for OCR. Please upload a clearer image with less blur or noise.")
#                 if blurry:
#                     st.warning("🔍 Detected blur: Image is too blurry for accurate OCR.")
#                 if noisy:
#                     st.warning("🔊 Detected noise: Image has excessive edge or frequency artifacts.")
#             else:
#                 with st.spinner('Processing Prescription...'):
#                     ocr_text = process_ocr_from_image_bytes(image_bytes)

#                 st.success("✅ OCR completed!")
#                 st.subheader("Extracted Text")
#                 st.text_area(label="Result", value=ocr_text, height=150)
                
                
                
                
#                 # Medicine Information 
# st.header("Learn about Prescribed Medication")
# medicine_name = st.text_input("Enter a medicine name here (e.g., Paracetamol):")

# if st.button("Get Medicine Information"):
#     if medicine_name:
#         with st.spinner(f'Searching for information on {medicine_name}...'):
#             info = get_medicine_info_and_translate(medicine_name)
        
#         if "error" in info:
#             st.error(info["error"])
#         else:
#             st.markdown(f"**Information for: {medicine_name}**")
#             st.info(info["result"])
#     else:
#         st.warning("Please enter a medicine name to search.")


import streamlit as st
import numpy as np
import os, base64, time
from OCR.image_ocr import process_ocr_from_image_bytes, get_medicine_info_and_translate

# ------------------- Theme Colors -------------------
DEEP_TEAL   = "#214457"   # Primary
ACCENT_BLUE = "#4fa5d8"   # Accent / hover
SOFT_TEAL   = "#bdd9cd"   # Sidebar secondary
BACKGROUND  = "#E7F1FB"   # Page background
SUCCESS     = "#388e3c"
DANGER      = "#e64a19"

# ------------------- Logo Helper --------------------
def _get_logo_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

logo_path = "pages/logo.jpeg"   # adjust if your logo is elsewhere
logo_b64  = _get_logo_base64(logo_path)
sidebar_logo_img_tag = f'<img class="sidebar-logo" src="data:image/jpeg;base64,{logo_b64}" alt="Logo"/>' if logo_b64 else ""

# ------------------- Page Config --------------------
st.set_page_config(page_title="Prescription OCR", layout="wide")

# ------------------- CSS ---------------------------
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
    background-color: rgba(240,248,255,0.08);
    transition: background-color 0.2s ease;
}}
.sidebar-header:hover {{
    background-color: rgba(240,248,255,0.14);
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
.header-bar h2 {{
    color: var(--deep-teal);
    font-size: 36px;
    font-weight: 900;
    margin: 0;
}}
.stButton>button {{
    background: var(--deep-teal);
    color: #ffffff;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 22px;
    border: none;
    transition: background 0.25s;
}}
.stButton>button:hover {{
    background: var(--success);
}}
</style>
""", unsafe_allow_html=True)

# ------------------- Sidebar -----------------------
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
    st.markdown('<a class="sidebar-link" href="Diabetes_Risk" target="_self"><img src="https://img.icons8.com/ios-filled/50/ffffff/diabetes.png"/>Diabetes Risk Detection </a>', unsafe_allow_html=True)
    st.markdown('<a class="sidebar-link" href="Info" target="_self"><img src="https://img.icons8.com/ios-filled/50/ffffff/info.png"/> Info</a>', unsafe_allow_html=True)
    st.markdown('<a class="sidebar-link" href="About" target="_self"><img src="https://img.icons8.com/ios-filled/50/ffffff/about.png"/> About</a>', unsafe_allow_html=True)

# ------------------- Page Header -------------------
st.markdown("""
<div class="header-bar">
    <h2>Prescription Digitization and Validation System</h2>
</div>
""", unsafe_allow_html=True)

# ------------------- File Upload -------------------
st.header("Digitized Prescription")
pres_file = st.file_uploader("Upload Prescription Image", type=['jpg', 'png', 'jpeg'])

# ------------------- OCR Trigger -------------------
if pres_file:
    st.image(pres_file, caption='Uploaded Prescription', width=350)

    if st.button("Perform OCR"):
        with st.spinner('Processing Prescription...'):
            image_bytes = pres_file.read()
            ocr_text = process_ocr_from_image_bytes(image_bytes)

        st.success("✅ OCR completed!")
        st.subheader("Extracted Text")
        st.text_area(label="Result", value=ocr_text, height=150)

# ------------------- Medicine Info -----------------
st.header("Learn about Prescribed Medication")
medicine_name = st.text_input("Enter a medicine name here (e.g., Paracetamol):")

if st.button("Get Medicine Information"):
    if medicine_name:
        with st.spinner(f'Searching for information on {medicine_name}...'):
            info = get_medicine_info_and_translate(medicine_name)
        if "error" in info:
            st.error(info["error"])
        else:
            st.markdown(f"**Information for: {medicine_name}**")
            st.info(info["result"])
    else:
        st.warning("Please enter a medicine name to search.")
