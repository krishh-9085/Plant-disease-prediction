import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import keras
import os

# Configure page layout and title
st.set_page_config(page_title="🌿 Plant Disease Recognition", layout="centered")
st.markdown(
    """
    <style>
        /* Main Header */
        .main-header {
            font-size: 3em;
            color: #4CAF50;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.2em;
        }
        /* Sub Header */
        .sub-header {
            font-size: 1.2em;
            color: #808080;
            text-align: center;
            margin-top: -15px;
        }
        /* Upload Box Styling */
        .upload-box {
            background-color: #f9fdf9;
            padding: 1.5%;
            border-radius: 15px;
            text-align: center;
            width: 80%;
            margin: 20px auto;
            transition: background-color 0.3s ease;
        }
        .upload-box:hover {
            background-color: #f0f8f0;
        }
        /* Prediction Box Styling */
        .prediction-box {
            font-size: 1.3em;
            color: white;
            background-color: #388E3C;
            padding: 1em;
            border-radius: 12px;
            text-align: center;
            font-weight: bold;
            width: 80%;
            margin: 20px auto;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease;
        }
        .prediction-box:hover {
            transform: scale(1.03);
        }
        /* Info Box Styling */
        .info-box {
            background-color: #f2f7f2;
            padding: 1em;
            border-radius: 10px;
            margin: 20px auto;
            width: 80%;
            font-size: 1em;
            color: #333;
        }
        /* Button Styling */
        .stButton > button {
            color: white;
            background-color: #4CAF50;
            border: none;
            border-radius: 5px;
            padding: 0.8em 1.5em;
            font-size: 1em;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #388E3C;
        }
    </style>
    """, unsafe_allow_html=True
)

# Load model for prediction
def model_prediction(test_image):
    model_path = os.path.join(os.getcwd(), 'trained_model.h5')
    cnn = keras.models.load_model(model_path)
    image_bytes = test_image.read()
    image = Image.open(io.BytesIO(image_bytes)).resize((128, 128))
    input_arr = np.array(image)[np.newaxis, ...]
    predictions = cnn.predict(input_arr)
    return np.argmax(predictions)

# Header
st.markdown('<h1 class="main-header">Plant Disease Recognition 🌱</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detect plant diseases accurately and swiftly</p>', unsafe_allow_html=True)

# Image uploader
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
st.subheader("📸 Upload an Image of Your Plant:")
test_image = st.file_uploader("", type=['jpg', 'jpeg', 'png'])
st.markdown('</div>', unsafe_allow_html=True)

# Display uploaded image and make predictions
if test_image is not None:
    st.image(test_image, caption="Uploaded Image", width=500)

    if st.button("🔍 Predict Disease"):
        with st.spinner('Analyzing... Please wait'):
            result_index = model_prediction(test_image)

        # Disease labels and descriptions
        class_name = [
            'Apple___Apple_scab',
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
            'Tomato___healthy'
        ]
        disease_info = {
    'Apple___Apple_scab': 'Apple scab is a fungal disease caused by *Venturia inaequalis*. Symptoms include olive-green to dark-brown spots on leaves and fruit, leading to distorted, cracked fruit. It thrives in cool, wet weather during the spring. Control methods include removing fallen leaves, pruning to increase airflow, and applying fungicides as a preventive measure.',
    
    'Apple___Black_rot': 'Black rot, caused by the fungus *Botryosphaeria obtusa*, manifests as dark lesions on leaves, fruit, and branches. Symptoms include fruit rot, leaf spot, and cankers on branches. It prefers warm, humid conditions. Control measures include pruning infected branches, removing fallen fruit, and using fungicides if needed.',
    
    'Apple___Cedar_apple_rust': 'Cedar apple rust is a fungal disease caused by *Gymnosporangium juniperi-virginianae*. Characterized by yellow-orange spots on leaves and sometimes fruit, it requires both cedar and apple trees to complete its lifecycle. Cool, moist conditions promote its spread. Control includes removing nearby cedar trees or applying fungicides on susceptible apple trees during spring.',
    
    'Apple___healthy': 'The apple tree is healthy, with no visible signs of disease or pest infestation. Maintaining a regular schedule of pruning, watering, and nutrient management helps keep the tree vigorous and resilient against diseases.',
    
    'Blueberry___healthy': 'The blueberry plant is healthy, with green leaves, abundant fruit, and no visible signs of disease. Proper pH level (4.5-5.5), mulching, and regular irrigation help ensure good health and productivity.',
    
    'Cherry_(including_sour)___Powdery_mildew': 'Powdery mildew, caused by *Podosphaera clandestina*, produces a white powdery coating on cherry leaves and fruit, weakening the plant and reducing yield. The disease thrives in warm, dry conditions with high humidity. Control includes pruning for airflow, using resistant varieties, and applying sulfur or other fungicides.',
    
    'Cherry_(including_sour)___healthy': 'The cherry tree is healthy, with dense foliage and no signs of disease. Regular pruning and balanced fertilization help maintain its vigor and resilience.',
    
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Cercospora leaf spot, caused by *Cercospora zeae-maydis*, appears as grayish lesions on corn leaves, reducing photosynthesis and yield. Warm, humid weather favors its spread. Crop rotation, resistant varieties, and fungicides are recommended control measures.',
    
    'Corn_(maize)___Common_rust_': 'Common rust, caused by *Puccinia sorghi*, manifests as reddish-brown pustules on corn leaves. The disease thrives in cool, wet weather. Planting resistant hybrids, crop rotation, and fungicides can help manage the disease.',
    
    'Corn_(maize)___Northern_Leaf_Blight': 'Northern leaf blight, caused by *Exserohilum turcicum*, causes elongated gray lesions on leaves. It is most severe in warm, humid conditions. Use of resistant varieties, crop rotation, and fungicides can help reduce its impact.',
    
    'Corn_(maize)___healthy': 'The corn plant is healthy, with green leaves and strong stalks. Proper fertilization, irrigation, and pest control help maintain its vigor and yield potential.',
    
    'Grape___Black_rot': 'Black rot in grapes is caused by *Guignardia bidwellii*. Symptoms include small, dark spots on leaves, and black, shriveled fruit. High humidity and warm temperatures favor the disease. Pruning, removing infected fruit, and applying fungicides can help manage black rot.',
    
    'Grape___Esca_(Black_Measles)': 'Esca, also known as black measles, is a complex fungal disease in grapes. Symptoms include leaf discoloration and fruit shriveling. It is particularly damaging in hot, dry climates. Control is challenging, but trunk renewal and limiting plant stress can help.',
    
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Leaf blight, caused by *Pseudocercospora vitis*, appears as dark spots on grape leaves, leading to premature leaf drop. It thrives in warm, wet weather. Regular pruning and fungicides help control it.',
    
    'Grape___healthy': 'The grapevine is healthy, showing vigorous growth and no signs of disease. Regular care, including pruning, irrigation, and disease monitoring, is essential to maintaining its health and productivity.',
    
    'Orange___Haunglongbing_(Citrus_greening)': 'Citrus greening, caused by *Candidatus Liberibacter* bacteria, is spread by the Asian citrus psyllid. Symptoms include yellowing leaves, stunted growth, and bitter, misshapen fruit. Management includes controlling psyllid populations and removing infected trees.',
    
    'Peach___Bacterial_spot': 'Bacterial spot, caused by *Xanthomonas arboricola*, causes dark, water-soaked spots on peach leaves and fruit, leading to premature leaf drop. The disease spreads in wet conditions. Control includes planting resistant varieties and copper-based sprays.',
    
    'Peach___healthy': 'The peach tree is healthy, showing vigorous growth and good fruit production. Regular care, including pruning and nutrient management, helps maintain its vitality and productivity.',
    
    'Pepper,_bell___Bacterial_spot': 'Bacterial spot in peppers, caused by *Xanthomonas campestris*, leads to water-soaked lesions on leaves and fruit. It thrives in warm, moist conditions. Preventative measures include crop rotation, resistant varieties, and copper-based bactericides.',
    
    'Pepper,_bell___healthy': 'The bell pepper plant is healthy, with vibrant leaves and strong fruiting. Regular watering, mulching, and pest management contribute to its health and yield.',
    
    'Potato___Early_blight': 'Early blight, caused by *Alternaria solani*, causes dark, concentric spots on potato leaves, weakening the plant and reducing yield. It spreads in warm, wet conditions. Crop rotation, resistant varieties, and fungicides are recommended controls.',
    
    'Potato___Late_blight': 'Late blight, caused by *Phytophthora infestans*, is a severe fungal disease causing water-soaked lesions on leaves and stems. It thrives in cool, wet weather and can devastate crops. Fungicides, crop rotation, and removing infected plants help control the spread.',
    
    'Potato___healthy': 'The potato plant is healthy, with strong foliage and good tuber development. Regular watering, nutrient management, and pest monitoring are key to its success.',
    
    'Raspberry___healthy': 'The raspberry plant is healthy, producing lush foliage and abundant fruit. Regular pruning and nutrient management are crucial for maintaining its vigor and productivity.',
    
    'Soybean___healthy': 'The soybean plant is healthy, showing good growth and development. Proper soil fertility, irrigation, and pest control contribute to optimal yield potential.',
    
    'Squash___Powdery_mildew': 'Powdery mildew, caused by *Erysiphe cichoracearum*, produces a white, powdery coating on squash leaves, reducing photosynthesis and yield. The disease thrives in dry, warm conditions. Control includes resistant varieties and applying sulfur-based fungicides.',
    
    'Strawberry___Leaf_scorch': 'Leaf scorch, caused by *Diplocarpon earlianum*, results in browning and drying of strawberry leaves, reducing fruit production. High humidity favors the disease. Management includes improving air circulation, mulching, and fungicide applications.',
    
    'Strawberry___healthy': 'The strawberry plant is healthy, producing vibrant leaves and abundant fruit. Regular watering, nutrient management, and disease monitoring are essential for successful fruit production.',
    
    'Tomato___Bacterial_spot': 'Bacterial spot in tomatoes, caused by *Xanthomonas spp.*, leads to dark, water-soaked lesions on leaves and fruit. It spreads rapidly in warm, humid conditions. Crop rotation and copper-based bactericides help manage it.',
    
    'Tomato___Early_blight': 'Early blight in tomatoes, caused by *Alternaria solani*, leads to dark spots with concentric rings on leaves. High humidity and warm temperatures favor the disease. Crop rotation and fungicides are effective control methods.',
    
    'Tomato___Late_blight': 'Late blight, caused by *Phytophthora infestans*, leads to water-soaked lesions and can rapidly destroy tomato plants. It spreads in cool, wet conditions. Fungicides and removing infected plants help limit its spread.',
    
    'Tomato___Leaf_Mold': 'Leaf mold, caused by *Passalora fulva*, causes yellow spots and moldy growth on the underside of leaves. It thrives in high humidity and low light. Pruning, improving ventilation, and using fungicides can help manage it.',
    
    'Tomato___Septoria_leaf_spot': 'Septoria leaf spot, caused by *Septoria lycopersici*, results in small, dark spots on tomato leaves, often causing defoliation. It spreads in warm, humid conditions. Crop rotation, pruning, and fungicides are effective controls.',
    
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spider mites are pests that feed on tomato plant sap, causing leaf yellowing and reduced fruit quality. They thrive in dry, hot conditions. Control includes regular watering, insecticidal soap, and predatory mites.',
    
    'Tomato___Target_Spot': 'Target spot, caused by *Corynespora cassiicola*, causes circular spots on leaves and fruit, leading to defoliation. It spreads in humid, warm conditions. Crop rotation and fungicides are effective measures.',
    
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato yellow leaf curl virus, spread by whiteflies, causes leaf yellowing, curling, and stunted growth, reducing fruit production. Control includes managing whitefly populations and using resistant varieties.',
    
    'Tomato___Tomato_mosaic_virus': 'Tomato mosaic virus causes leaf mottling and reduced plant vigor. It spreads via contaminated tools and infected plants. Proper sanitation and resistant varieties help manage it.',
    
    'Tomato___healthy': 'The tomato plant is healthy, producing vigorous growth and fruit. Proper watering, nutrient management, and pest control help maximize yield.'
}


        # Display prediction result
        st.markdown(f'<div class="prediction-box">🌿 Predicted Disease: {class_name[result_index]}</div>', unsafe_allow_html=True)

        # Show detailed information in an expander
        with st.expander("More Information"):
            st.write("Disease Information:")
            st.write(disease_info.get(class_name[result_index], "Description not available."))
