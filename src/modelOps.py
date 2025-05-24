import streamlit as st
from torchvision import models, transforms
import torch
from torch import nn
from PIL import Image
from pathlib import Path

# 1. Load Model
@st.cache_resource
def load_model(MODEL_PATH: Path = "src/outputs/p2_e29_best_model.pth", device: str = "cpu"):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model

# 2. Preprocessing pipeline (match training)
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_class(input_tensor, model):
    """ Attempts to predict  """
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0, predicted_class].item()
    
    return predicted_class, confidence


