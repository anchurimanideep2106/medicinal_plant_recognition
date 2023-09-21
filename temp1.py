import streamlit as st
from PIL import Image
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
from util1 import classify

loaded_model=pickle.load(open("finalmodel.pkl",'rb'))


# load class names
encoder=joblib.load("label_encoder_1.pkl",'rb')
class_names=encoder.classes_
# load_model_button = st.button("Load Model")
# @st.cache_resource
# def load_resources():
#     loaded_model = load_model(r"E:\finalmodel.pkl")
#     encoder = joblib.load(r"E:\label_encoder_1.pkl")
#     return loaded_model, encoder

# # Load the model and other resources using the cached function
# loaded_model, encoder = load_resources()



# set title
st.title('Medicinal Plant Recognition')

# set header
st.header('Please upload a medicinal plant image')

# upload file
file = st.file_uploader('', type=['jpg'])

# display image

if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score,time_,details = classify(image, loaded_model, encoder)

    if(len(class_name)==1):
        st.write('## Plant Name: {}'.format(class_name[0]))
        st.write("### Plant details: ")
        for line in details.split('\n'):
            st.write("#### {}".format(line))

    else:
        st.write('##### This image might be subjected to noise or the image is blurred, please check your image.')
        st.write('#### Three nearest plant classes for the given image are :')
        st.write("")
        for j in range(len(class_name)):
            st.write('### Plane Name: {}'.format(class_name[j]))
            st.write('##### Confidence Score: {}%'.format(int(conf_score[j])),end='\n')
        st.write('Please upload a clear image.')
    st.write(f'##### Inference time: {time_:.2f} seconds')

else:
    st.write("Please upload an image.")
