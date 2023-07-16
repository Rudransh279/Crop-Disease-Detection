import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image


# predict function
def predict(model, img):


    img = Image.open(img)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    if model == potato_model:
        img_array_scaled = tf.keras.layers.Resizing(height=256, width=256)(img_array)
        class_names = ['Potato Early blight', 'Potato Late blight', 'Potato healthy']
    else:
        img_array_scaled = tf.keras.layers.Resizing(height=144, width=144)(img_array)
        class_names = ['Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
                       'Tomato Leaf Mold', 'Tomato Septoria leaf_spot', 'Tomato Spider mites Two spotted spider mite',
                       'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus',
                       'Tomato healthy']

    img_array = tf.expand_dims(img_array_scaled, 0)

    predictions = model.predict(img_array)

    for prediction in predictions:
        predicted_class = str(class_names[np.argmax(prediction)])
        confidence = str(round(100 * (np.max(prediction)), 2))
    return predicted_class, confidence


# loading the models
potato_model = tf.keras.models.load_model('potatoes.h5')
tomato_model_self = tf.keras.models.load_model('tomato.h5')
# tomato_model_transfer = tf.keras.models.load_model('transfer_learning_model.h5')


# sidebar for navigation

with st.sidebar:
    selected = option_menu('Multiple crop Disease Detection',

                           ['Potato Disease Predictor',
                            'Tomato Disease Predictor 1'],

                           icons=['flower1', 'flower2'],

                           default_index=0)

# Potato Disease Prediction page
if selected == 'Potato Disease Predictor':

    # page title
    st.title('Potato Disease Prediction using ML')

    # getting the input data from the user
    img = st.file_uploader("Please upload image of Potato Crop", type=["jpg", "png"])
    st.write("please upload image of single leaf for better result")


    # code for Prediction
    potato_diagnosis = ''
    confidence = ''

    # creating a button for Prediction

    if st.button('Diagnosis Test Result'):
        if img is None:
            st.write("please upload the image first")
        else:
            potato_diagnosis,confidence = predict(potato_model, img)
            result = f"diagnosis is: {potato_diagnosis}, \n confidence is: {confidence}"
            if potato_diagnosis == 'Potato healthy':
                img_display= Image.open("happy_potato.jpg")
                st.image(img_display)
            else:
                img_display = Image.open('sad_potato.jpg')
                st.image(img_display)

            st.success(result)

# Tomato Disease Prediction page
if selected == 'Tomato Disease Predictor 1':

    # page title
    st.title('Tomato Disease Prediction using ML')
    

    # getting the input data from the user
    img = st.file_uploader("Please upload image of Tomato crop", type=["jpg", "png"])
    st.write("please upload image of single leaf for better result")

    # code for Prediction
    tomato_diagnosis = ''
    confidence = ''
    result2 =''

    # creating a button for Prediction

    if st.button('Diagnosis Test Result'):
        if img is None:
            st.write("please upload the image first")
        else:
            tomato_diagnosis,confidence = predict(tomato_model_self, img)
            result2 = f"diagnosis is: {tomato_diagnosis} , \n confidence is: {confidence}"
            if tomato_diagnosis == 'Tomato healthy':
                img_display= Image.open("tomato_happy.jpg")
                st.image(img_display)
            else:
                img_display = Image.open('sad_tomato.jpg')
                st.image(img_display)

            st.success(result2)


# tomato Disease Prediction page
# if selected == 'Tomato Disease Predictor 2':

#     # page title
#     st.title('Tomato Disease Prediction using transfer learning')

#     # getting the input data from the user
#     img = st.file_uploader("Please upload image of potato leaf", type=["jpg", "png"])
#     st.write("please upload image of single leaf for better result")

#     # code for Prediction
#     tomato_diagnosis2 = ''
#     confidence = ''
#     result3 = ''

#     # creating a button for Prediction

#     if st.button('Diagnosis Test Result'):
#         if img is None:
#             st.write("please upload the image first")

#         else:
#             tomato_diagnosis2,confidence = predict(tomato_model_transfer, img)
#             result3 = f"diagnosis is: {tomato_diagnosis2}, \n confidence is: {confidence}"

#             if tomato_diagnosis2 == 'Tomato healthy':
#                 img_display= Image.open("tomato_happy.jpg")
#                 st.image(img_display)
#             else:
#                 img_display = Image.open('sad_tomato.jpg')
#                 st.image(img_display)

#             st.success(result3)