import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import shap
from ultralytics import YOLO

model = YOLO("runs/classify/yolo26n-cls-4/weights/best.pt")
model.eval()

IMAGE_PATH = "images/test.jpg"
image = Image.open(IMAGE_PATH).convert("RGB")
image= image.resize((240, 240))  
img_array = np.array(image)

def model_predict(x):
    """
    x: numpy array [B,H,W,C] in [0,1]
    returns: numpy array [B,num_classes] with probabilities
    """
    x_uint8 = (x * 255).astype(np.uint8)
    all_probs = []
    
    for i in range(x_uint8.shape[0]):
        img = x_uint8[i]
        result = model.predict(img, verbose=False)[0]  # take first result
        probs = result.probs.data.cpu().numpy()
        all_probs.append(probs)
    
    return np.array(all_probs)
     

# SHAP explainer
masker = shap.maskers.Image("inpaint_telea", img_array.shape)  

explainer = shap.Explainer(model_predict, masker)

# Explain the image
input_img = np.array(img_array, dtype=np.float32)[np.newaxis, ...] / 255.0
shap_values = explainer(img_array[np.newaxis, ...], max_evals=100)


shap.image_plot([shap_values.values[0]], -img_array)  # negative for visualization
