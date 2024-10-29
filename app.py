import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Customize page layout and title
st.set_page_config(page_title="🌿 Plant Disease Recognition System", layout="centered")
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.5em;
            color: #228B22;
            font-weight: bold;
            text-align: center;
        }
        .sub-header {
            font-size: 1.5em;
            color: #4F7942;
            margin-top: -10px;
            text-align: center;
        }
        .upload-box {
            border: 2px solid #d9d9d9;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            background-color: #f7f7f7;
        }
        .prediction-box {
            font-size: 1.2em;
            color: white;
            background-color: #4F7942;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True
)

# TensorFlow Model Prediction
def model_prediction(test_image):
    cnn = tf.keras.models.load_model("trained_model.keras")
    image_bytes = test_image.read()  # Read the uploaded image file once
    image = Image.open(io.BytesIO(image_bytes)).resize((128, 128))  # Open and resize the image
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Reshape for the model
    predictions = cnn.predict(input_arr)
    result_index = np.argmax(predictions)
    return result_index  # return index of max element

# Header
st.markdown('<h1 class="main-header">Plant Disease Recognition System 🌱</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect plant diseases accurately and swiftly</p>', unsafe_allow_html=True)

# Image uploader with styling
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
st.subheader("📸 Upload an Image:")
test_image = st.file_uploader("", type=['jpg', 'jpeg', 'png'])
st.markdown('</div>', unsafe_allow_html=True)

# Display uploaded image and prediction
if test_image is not None:
    st.image(test_image, caption="Uploaded Image", width=300)

    # Predict button with padding
    if st.button("🔍 Predict Disease"):
        result_index = model_prediction(test_image)
        
        # Labels for plant diseases
        class_name=['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

        # Display prediction with custom style
        st.markdown('<div class="prediction-box">🌿 Prediction: {}</div>'.format(class_name[result_index]), unsafe_allow_html=True)
