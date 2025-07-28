import streamlit as st
import os
import time
from PIL import Image
import numpy as np

# Import PyTorch with error handling
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
except ImportError as e:
    st.error(f"PyTorch import error: {e}")
    st.stop()

import gdown

# Configure Streamlit to avoid PyTorch conflicts
st.set_page_config(
    page_title="Whale Species Classifier",
    page_icon="🐋",
    layout="centered"
)

# Disable file watcher for PyTorch compatibility
if 'model' not in st.session_state:
    st.session_state.model = None

# Class labels
data_cat = ['Fin Whale', 'Gray Whale', 'Humpback Whale', 'Southern Right Whale']

# File path and Google Drive model link
MODEL_PATH = r"C:\Users\ASUS\OneDrive\Desktop\deploy\resnet50_whale_classification.pth"
# Fixed Google Drive URL format
GDRIVE_URL = "https://drive.google.com/uc?id=1FNnocDlVS59JTZ7-7Lxuh2I3ScIzVR_p"

# Load model function with session state
def load_model():
    if st.session_state.model is None:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(data_cat))
        
        model.load_state_dict(torch.load(
            MODEL_PATH,
            map_location=torch.device('cpu'),
            weights_only=False
        ))
        
        model.eval()
        st.session_state.model = model
    return st.session_state.model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Custom CSS for Blue Theme
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

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>🐋 Whale Species Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Dive deep into the world of whales!</h4>", unsafe_allow_html=True)

# IEEE GRSS Challenge Section
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h3 style='text-align: center; color: #01579b;'>🏆 IEEE GRSS Student Grand Challenge</h3>", unsafe_allow_html=True)
    
    # Display the IEEE GRSS logo
    # Check if logo exists and display it
    logo_path = "ieee_grss_logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=400)
    else:
        # If local file doesn't exist, try different common paths
        possible_paths = [
            "./ieee_grss_logo.png",
            "images/ieee_grss_logo.png", 
            "assets/ieee_grss_logo.png"
        ]
        
        logo_found = False
        for path in possible_paths:
            if os.path.exists(path):
                st.image(path, width=400)
                logo_found = True
                break
        
        if not logo_found:
            # Display a simple text placeholder
            st.markdown("<h4 style='text-align: center; color: #01579b;'>📋 IEEE GRSS Student Grand Challenge - 4th Edition</h4>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin: 20px 0;'>
        <p style='font-size: 16px; color: #003366;'>
            This whale species classification project is part of the 4th IEEE GRSS Student Grand Challenge.
        </p>
        <a href='https://www.grss-ieee.org/resources/news/fourth-grss-student-grand-challenge-sgc/' 
           target='_blank' 
           style='background-color: #0277bd; color: white; padding: 10px 20px; 
                  text-decoration: none; border-radius: 5px; font-weight: bold;'>
            📖 View Problem Statement
        </a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model file..."):
        try:
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            st.stop()

# Load the model after ensuring it exists
try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

uploaded_file = st.file_uploader("🌊 Upload a whale image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.markdown("<h5 style='text-align: center;'>Uploaded Image</h5>", unsafe_allow_html=True)
    st.image(image, caption="Beautiful Whale", width=300, use_column_width=False)
    
    input_tensor = transform(image).unsqueeze(0)

    # Progress Bar
    progress_text = "🔎 Analyzing the whale species..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(0, 100, 10):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 10, text=progress_text)
    
    # Make prediction
    try:
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0).numpy()

        predicted_label = data_cat[np.argmax(probs)]
        confidence = np.max(probs) * 100

        my_bar.empty()

        st.markdown("---")
        st.markdown(f"<h2 style='text-align: center;'>🔎 Prediction Result</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: #01579b;'>Species: {predicted_label}</h3>", unsafe_allow_html=True)
        st.markdown("---")
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        my_bar.empty()
else:
    st.markdown("<p style='text-align: center;'>Please upload an image to classify the whale species!</p>", unsafe_allow_html=True)