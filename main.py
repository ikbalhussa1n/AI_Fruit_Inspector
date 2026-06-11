import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "fruit_classifier.keras"
FILE_ID = "1l-Vsm6tAAcg59NLIllVHR2PK7i2ilMqk"

# ----------------------------
# LOAD MODEL (from Drive if not present)
# ----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ----------------------------
# GET INPUT SHAPE SAFELY
# ----------------------------
input_shape = model.input_shape

# If model expects image input (CNN)
is_image_model = len(input_shape) == 4

st.title("🍎 Fruit Freshness Classifier")
st.write("Upload an image of a fruit to check if it is Fresh or Rotten.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# ----------------------------
# PREDICTION
# ----------------------------
if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Processing image...")

    # ----------------------------
    # IMAGE PREPROCESSING
    # ----------------------------
    if is_image_model:
        # CNN model (Conv2D expected)
        h, w = input_shape[1], input_shape[2]
        image_resized = image.resize((w, h))
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

    else:
        # Dense model (FLATTEN expected)
        image_resized = image.resize((64, 64))  # fallback size
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_batch = img_array.reshape(1, -1)

    st.write("Input shape:", img_batch.shape)

    # ----------------------------
    # PREDICT
    # ----------------------------
    prediction = model.predict(img_batch)

    # ----------------------------
    # HANDLE OUTPUT
    # ----------------------------
    if prediction.shape[-1] == 1:
        value = float(prediction[0][0])

        label = "Rotten" if value >= 0.5 else "Fresh"
        confidence = value if value >= 0.5 else 1 - value

    else:
        class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction))

        label = "Fresh" if class_idx == 0 else "Rotten"

    # ----------------------------
    # RESULT UI
    # ----------------------------
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")

    if label == "Fresh":
        st.success("🍏 The fruit looks fresh!")
    else:
        st.error("🍂 The fruit looks rotten!")