import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
import os
import gdown  # <-- For downloading from Google Drive

# Class labels
data_cat = ['Fin Whale', 'Gray Whale', 'Humpback Whale', 'Southern Right Whale']

# File path and Google Drive model link
MODEL_PATH = r"C:\Users\ASUS\OneDrive\Desktop\deploy\resnet50_whale_classification.pth"
GDRIVE_URL = "https://drive.google.com/file/d/1FNnocDlVS59JTZ7-7Lxuh2I3ScIzVR_p/view?usp=sharing"  # Replace with your actual FILE ID

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model file..."):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load model
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(data_cat))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Custom CSS for Blue Theme ---
st.markdown("""
    <style>
    body {
        background-color: #e0f7fa;
    }
    .reportview-container {
        background: linear-gradient(to right, #81d4fa, #b3e5fc);
        color: #003366;
    }
    .sidebar .sidebar-content {
        background: #0288d1;
    }
    h1, h2, h3, h4, h5 {
        color: #01579b;
        text-align: center;
    }
    .stButton>button {
        color: white;
        background-color: #0277bd;
    }
    </style>
""", unsafe_allow_html=True)

# --- Streamlit UI ---
st.markdown("<h1 style='text-align: center;'>🐋 Whale Species Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Dive deep into the world of whales!</h4>", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("🌊 Upload a whale image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.markdown("<h5 style='text-align: center;'>Uploaded Image</h5>", unsafe_allow_html=True)
    st.image(image, caption="Beautiful Whale", width=300, use_column_width=False)
    
    input_tensor = transform(image).unsqueeze(0)

    # --- Progress Bar ---
    progress_text = "🔎 Analyzing the whale species..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(0, 100, 10):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 10, text=progress_text)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0).numpy()

    predicted_label = data_cat[np.argmax(probs)]
    confidence = np.max(probs) * 100

    my_bar.empty()

    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center;'>🔎 Prediction Result</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; color: #01579b;'>Species: {predicted_label}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)
    st.markdown("---")
else:
    st.markdown("<p style='text-align: center;'>Please upload an image to classify the whale species!</p>", unsafe_allow_html=True)
