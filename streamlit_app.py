import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# --- Load TFLite model ---
interpreter = tf.lite.Interpreter(model_path="currency_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Class labels ---
class_labels = {0: '5 SR', 1: '10 SR', 2: '50 SR', 3: '100 SR', 4: '200 SR'}

# --- UI Setup ---
st.set_page_config(page_title="Currency Classifier", page_icon="💵", layout="centered")
st.title("💵 Saudi Currency Classifier")
st.write("Upload a Saudi banknote image and get its denomination prediction.")

uploaded_file = st.file_uploader("Select an Image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Show the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)
    # Preprocess
    image_resized = cv2.resize(image, (128, 128))
    image_normalized = image_resized / 255.0
    input_tensor = np.expand_dims(image_normalized, axis=0).astype(np.float32)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    predicted_label = np.argmax(output)
    confidence = np.max(output) * 100

    # Display result
    st.markdown(f"""
    ### ✅ Prediction:
    **{class_labels[predicted_label]}**

    **Confidence:** {confidence:.2f}%
    """)
