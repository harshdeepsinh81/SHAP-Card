import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# ------------------------------
# Load your trained YOLOv8 classification model
# ------------------------------
MODEL_PATH = "runs/classify/yolo26n-cls-4/weights/best.pt"
model = YOLO(MODEL_PATH)

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="Card Classifier", layout="centered")
st.title("Aadhar/Pan Card Classification")

st.write("Upload an image to classify it into one of your card classes.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)
    
    # Convert PIL image to numpy array for YOLO
    img_array = np.array(image)
    
    # Run prediction
    with st.spinner("Classifying..."):
        results = model.predict(img_array)
    
    # Access classification prediction correctly
    pred_class_idx = results[0].probs.top1          # index of predicted class
    pred_label = results[0].names[pred_class_idx]  # class name
    pred_conf = results[0].probs.top1conf.item()   # confidence score
    
    # Display results
    st.markdown(f"**Predicted Class:** {pred_label}")
    st.markdown(f"**Confidence:** {pred_conf*100:.2f}%")