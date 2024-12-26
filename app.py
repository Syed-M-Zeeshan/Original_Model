# app.py

import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Set the page configuration **before** any other Streamlit commands
st.set_page_config(
    page_title="Blood Cancer Detection Tool",
    layout="centered",
    # You can also add other parameters like page_icon if needed
)

# 2. Define paths
MODEL_PATH = "D:/my Data/OneDrive/Documents/Uni work 5/ANN Lab//ANN_Project/Original_Model/blood_cancer_detector.keras"  # Updated extension

# 3. Load the trained model with caching
@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()

# 4. Define class labels and analysis texts
CLASS_LABELS = ['Benign', 'Early', 'Pre', 'Pro']
ANALYSIS_TEXT = {
    'Benign': "No cancerous cells detected. Continue regular check-ups.",
    'Early': "Early stage cancer detected. Recommended to consult a hematologist for further procedures.",
    'Pre': "Pre-cancerous cells detected. Monitor closely and follow medical advice.",
    'Pro': "Advanced stage cancer detected. Immediate medical intervention required."
}

# 5. Define utility functions
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def annotate_image(img, prediction_label):
    """
    Annotate the image with the prediction label.

    Args:
        img (PIL.Image): Original image.
        prediction_label (str): Predicted class label.

    Returns:
        Annotated image as a matplotlib Figure object.
    """
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.set_title(f"Predicted: {prediction_label}", fontsize=16)
    ax.axis('off')
    return fig

# 6. Define the main function
def main():
    st.title("🩸 Blood Cancer Detection Tool")
    st.write("Upload a blood cell image to detect cancer and its stage.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load and display image
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption='Uploaded Image.', use_container_width=True)  # Updated parameter
            st.write("")
            st.write("🔍 Classifying...")

            # Preprocess image
            img_preprocessed = preprocess_image(img)

            # Make prediction
            predictions = model.predict(img_preprocessed)
            pred_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100
            predicted_label = CLASS_LABELS[pred_class]

            # Display prediction
            st.success(f"**Prediction:** {predicted_label} ({confidence:.2f}% confidence)")

            # Annotate image
            fig = annotate_image(img, predicted_label)
            st.pyplot(fig)
            plt.close(fig)  # Correctly closes the Figure object

            # Detailed Analysis
            st.header("📄 Detailed Analysis")
            st.write(ANALYSIS_TEXT.get(predicted_label, "No analysis available."))

            # Confidence Breakdown
            st.markdown("---")
            st.write("**Confidence Breakdown:**")
            confidence_dict = {label: f"{prob*100:.2f}%" for label, prob in zip(CLASS_LABELS, predictions[0])}
            st.write(confidence_dict)

        except Exception as e:
            st.error("❌ Error processing the image. Please upload a valid blood cell image.")
            st.error(f"**Error Details:** {e}")
    else:
        st.info("💡 Please upload an image to get started.")

    # Footer
    st.markdown("---")
    st.write("Developed as an educational project for blood cancer detection using deep learning.")

# 7. Execute the main function
if __name__ == "__main__":
    main()