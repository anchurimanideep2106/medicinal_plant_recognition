import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import time



def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, encoder):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    plant_details = {
    "Alpinia Galanga (Rasna)": "Telugu name: Dumparaashtrakamu \nHindi name: Kulanjan \nSanskrit name: Mahabhari Vacha",
    "Amaranthus Viridis (Arive-Dantu)": "Telugu name: Thotakura \nHindi name: Jangli Chaulai \nSanskrit name: Tanduliya",
    "Artocarpus Heterophyllus (Jackfruit)": "Telugu Name: Panasa \nHindi Name: Kathal \nSanskrit Name: Panasam",
    "Azadirachta Indica (Neem)": "Telugu name: Vepa \nHindi name: Neem \nSanskrit name: Nimba \nTaxonomy - Kingdom: Plantae \nClade: Tracheophytes \nFamily: Meliaceae \nGenus: Azadirachta \nSpecies: A. indica",
    "Basella Alba (Basale)": "Telugu Name: Bachhali \nHindi Name: Poi \nSanskrit Name: Upodika",
    "Brassica Juncea (Indian Mustard)": "Telugu Name: Avalu \nHindi Name: Rai \nSanskrit Name: Rajika",
    "Carissa Carandas (Karanda)": "Telugu Name: Vakkai \nHindi Name: Karonda \nSanskrit Name: Karamarda",
    "Citrus Limon (Lemon)": "Telugu Name: Nimma \nHindi Name: Nimbu \nSanskrit Name: Nimbuka",
    "Ficus Auriculata (Roxburgh fig)": "Telugu Name: Racha bodda \nHindi Name: Tirmal \nSanskrit Name: Audumbara",
    "Ficus Religiosa (Peepal Tree)": "Telugu Name: Raavi Chettu \nHindi Name: Pipal \nSanskrit Name: Ashvattha",
    "Hibiscus Rosa-sinensis": "Telugu Name: Dasanamu \nHindi Name: Gudhal \nSanskrit Name: Japa",
    "Jasminum (Jasmine)": "Telugu Name: Malle Puvvu Chettu \nHindi Name: Juhi, Champa Bela \nSanskrit Name: Mallika \nTaxonomy - Kingdom: Plantae \nClade: Tracheophytes \nClade: Angiosperms \nClade: Eudicots \nClade: Asterids \nOrder: Lamiales \nFamily: Oleaceae \nTribe: Jasmineae \nGenus: Jasminum",
    "Mangifera Indica (Mango)": "Telugu Name: Mamidi \nHindi Name: Aam \nSanskrit Name: Aamra",
    "Mentha (Mint)": "Telugu Name: Podina \nHindi Name: Pudina \nSanskrit Name: Putiha",
    "Moringa Oleifera (Drumstick)": "Telugu name - Munaga1 \nHindi name - Sahijan \nSanskrit name - Shigru",
    "Muntingia Calabura (Jamaica Cherry-Gasagase)": "Telugu Name: Nakkaraegu \nHindi Name: Muntingia Calabura \nSanskrit Name: Vilvaphala",
    "Murraya Koenigii (Curry)": "Telugu Name: Karepeku \nHindi Name: Mitha Neem \nSanskrit Name: Surabhinimba",
    "Ocimum Tenuiflorum (Tulsi)": "Telugu Name: Tulasi \nHindi Name: Tulsi \nSanskrit Name: Tulasi \nTaxonomy - Kingdom: Plantae \nClade: Tracheophytes \nClade: Angiosperms \nClade: Eudicots \nClade: Asterids \nOrder: Lamiales \nFamily: Lamiaceae \nGenus: Ocimum \nSpecies: O. tenuiflorum"   
    }
    details=None
    classes_dict=plant_details.keys()
    start_time=time.time()
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = image_array / 255.0

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    end_time=time.time()
    prediction=prediction.reshape(-1,)
    if(np.max(prediction)>0.85):
        label=[encoder.inverse_transform([np.argmax(prediction)])]
        if(str(label[0][0]) in classes_dict):
            details=plant_details[label[0][0]]
        confidence=[prediction[np.argmax(prediction)]*100]
    else:
        top=np.argsort(prediction)[-3:][::-1]
        confidence=prediction[top]*100
        label=[encoder.inverse_transform([i]) for i in top]
    total_time=end_time-start_time
    return label,confidence,total_time,details