

### Sehat AI ( Smarter Healthcare for All )
Sehat AI is an AI-powered healthcare solution developed for the Uraan Competition under the Healthcare Domain, designed to democratize medical diagnostics and prescription understanding. It combines deep learning, OCR, and machine learning to assist doctors, patients, and pharmacists in making informed decisions.
The project features three modular, beginner-friendly components:

- Pneumonia Detection using CNN
- OCR for Digital Prescriptions (English + Urdu)
- Diabetes Risk Detection using ML


Each module is fully reproducible and runs seamlessly via a simple setup using Miniconda and Streamlit, with pre-downloaded datasets and minimal configuration required
    

## ⚙️ Installation Guide

We recommend using **Miniconda** to manage environment.
 
### 🔧 Step-by-Step Setup

1. **Install Miniconda**  
   [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install it should be in your System.

2. **Create Environment & Install Dependencies**

   Open your terminal and run each command 

   ```bash
   conda create -n sehat_env python=3.10
   conda activate sehat_env
   pip install -r requirements.txt 
3. **Run the Project**
    
    Once libraries are installed, launch the app:

    ```bash
    streamlit run app.py
# 📁 Datasets (Already Included)

No need to download manually — all datasets are pre-downloaded and placed in the correct folders.

  - #### Pneumonia Detection Dataset  
    - Source: Kaggle - Chest X-ray Pneumonia

-  #### Diabetes Risk Detection Dataset
   -  Source: Kaggle - Pima Indians Diabetes


# Module Overview

## 1️⃣ Pneumonia Detection (CNN)

      - Uses chest X-ray images to detect pneumonia.
      - Built with Convolutional Neural Networks (CNN).
      - Dataset from Kaggle is already included.
      - Output: Prediction of pneumonia presence or absence.


## 2️⃣ OCR Digital Prescription (English + Urdu)


    - Input: Image of a medicine strip or prescription.
    - Output: Medicine name and its purpose in English and Urdu.
    - Powered by Mistral AI API.
    🔑 Mistral API Setup
    - API expires on 1st November by default.
    - To extend usage:
    - Visit Mistral Console
    - Log in and create your own API key.
    - Set expiration date as per your need.

## 3️⃣ Diabetes Risk Detection (ML)

    - Uses patient data to predict diabetes risk.
    - Built with Machine Learning algorithms.
    - Dataset from Kaggle is already included.
    - Output: Risk level (High / Medium / Low)


# 🏁 Final Notes
- This project was built for the Uraan Competition under the Healthcare Domain.
  

