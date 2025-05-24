import streamlit as st
import torch
from PIL import Image
from modelOps import load_model, preprocess_image, predict_class

def main():
    st.set_page_config(page_title="Breaking Bone", page_icon="🦴")
    st.title("🦴 Breaking Bone")
    st.write("An X-Ray Broken Bone Classifier")
    st.caption("Prepared by: John Manuel Carado")
    st.write("Upload an X-ray image to classify potential fractures.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device)

    uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image")

        with col2:
            st.subheader("Classification Results")
            with st.spinner("Classifying..."):
                try:
                    input_tensor = preprocess_image(image).to(device)
                    pred, conf = predict_class(input_tensor, model)
                    st.success(f"Predicted Class: **{pred}**")
                    st.info(f"Confidence: **{conf:.2%}**")
                except Exception as e:
                    st.error(f"An error occurred during classification: {e}")
                    st.write("Please ensure the uploaded image is valid and the model is loaded correctly.")

if __name__ == "__main__":
    main()