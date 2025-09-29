import base64
import os
import streamlit as st
from mistralai import Mistral

# This function encapsulates the OCR logic to be called from the UI.
def process_ocr_from_image_bytes(image_bytes):
    API = "9qv7uViZZ9mQBNqoHHRkT108e1krnQHT"
    client = Mistral(api_key=API)

    # Encode the image to base64
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Call the Mistral OCR API
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": f"data:image/png;base64,{base64_image}"
        }
    )

    # Extract and return the markdown text
    if ocr_response.pages:
        return ocr_response.pages[0].markdown
    else:
        return "No text could be extracted from the image."
    
    
def get_medicine_info_and_translate(medicine_name):
    API = "9qv7uViZZ9mQBNqoHHRkT108e1krnQHT"
    client = Mistral(api_key=API)

    # Clear and strict prompt for consistent formatting
    prompt = f"""
    You are a medical assistant. Describe the medicine "{medicine_name}" in one short sentence in English, focusing on its primary use.
    Then translate that sentence into Urdu.
    Always start the English line with 'English:' and the Urdu line with 'Urdu:'.
    If the medicine is unknown, say clearly:
    English: This medicine is not recognized.
    Urdu: یہ دوا پہچانی نہیں گئی۔
    """

    # Call the chat model using the correct method
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    full_text = response.choices[0].message.content

    # Flexible parsing logic
    english_part = ""
    urdu_part = ""

    lines = full_text.split('\n')
    for line in lines:
        if "English:" in line:
            english_part = line.split("English:")[-1].strip()
        elif "Urdu:" in line:
            urdu_part = line.split("Urdu:")[-1].strip()

    if english_part and urdu_part:
        formatted_info = f"**English:** {english_part}\n\n**Urdu:** {urdu_part}"
        return {"result": formatted_info}
    else:
        return {"error": "Could not parse the response. Please try again."}