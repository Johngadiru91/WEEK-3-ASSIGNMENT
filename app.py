# This content will be written to a file named 'app.py'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image # For image handling
import pandas as pd # For displaying probabilities nicely

# --- 1. Load the Trained MNIST Model ---
# @st.cache_resource is used to load the model only once, improving performance
@st.cache_resource
def load_mnist_model():
    # Make sure 'mnist_cnn_model.h5' is in the same directory as this script (or uploaded to Colab)
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    return model

model = load_mnist_model()

# --- 2. Streamlit App Interface ---
st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) and let the model predict it!")
st.markdown("---")

# --- 3. Input Mechanism: Image Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("L") # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # --- 4. Preprocess Input Image ---
    # Resize to 28x28 pixels (MNIST input size)
    image = image.resize((28, 28))
    # Convert to numpy array
    image_array = np.array(image)
    # Normalize pixel values to 0-1
    image_array = image_array / 255.0
    # Reshape for model input (add batch dimension and channel dimension for CNN: (1, 28, 28, 1))
    image_for_prediction = image_array.reshape(1, 28, 28, 1)

    # --- 5. Make Prediction ---
    predictions = model.predict(image_for_prediction)
    predicted_class = np.argmax(predictions) # Get the index of the highest probability
    confidence = np.max(predictions) * 100 # Get the highest probability as confidence

    # --- 6. Display Results ---
    st.success(f"**Predicted Digit:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    st.markdown("---")
    st.subheader("All Class Probabilities:")
    prob_df = pd.DataFrame({
        'Digit': range(10),
        'Probability': [f"{p*100:.2f}%" for p in predictions[0]]
    })
    st.dataframe(prob_df)
