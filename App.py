import streamlit as st
from Pred import show_predict_page
from PIL import Image
import pandas
import numpy

image1 = Image.open('./images/imag1.jpg')
image1 = image1.resize((600,300))
image2 = Image.open('./images/imag2.jpg')


st.sidebar.image(image1)
st.sidebar.title("Select an option")
page = st.sidebar.selectbox("Explore or predict", ("Predict", "Explore")) 



if page =="Predict":
    show_predict_page()
else:
    show_explore_page()    
    
