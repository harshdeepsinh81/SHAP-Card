from ultralytics import YOLO
import numpy as np
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

model = YOLO("runs/classify/yolo26n-cls-4/weights/best.pt")

def predict_fn(images):
    preds = []

    for img in images:
        # Ensure 3 channels
        if img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        results = model.predict(img, imgsz=224, batch=1, verbose=False)

        probs = results[0].probs.data  

        if hasattr(probs, "numpy"):
            probs = probs.numpy()

        preds.append(probs)

    return np.stack(preds)

image_path = "images/ad1.jpg"
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(
    image=img,
    classifier_fn=predict_fn,
    top_labels=2,       
    hide_color=0,      
    num_samples=300    
)

top_class = explanation.top_labels[0]  

temp, mask = explanation.get_image_and_mask(
    top_class,
    positive_only=True,  
    num_features=10,     
    hide_rest=False
)

plt.figure(figsize=(8, 8))
plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.axis('off')
plt.title(f"LIME explanation for class {top_class}")
plt.show()

probs = predict_fn([img])[0]
print("Predicted probabilities:", probs)