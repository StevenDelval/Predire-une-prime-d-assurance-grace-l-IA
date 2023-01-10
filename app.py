import streamlit as st
import pandas as pd
import numpy as np
import pickle
from functions import *

pickle_in = open('modelLasso.pkl', 'rb') 
modelLasso = pickle.load(pickle_in)
pickle_in = open('modelElastic.pkl', 'rb') 
modelElastic = pickle.load(pickle_in)


titre = "Prédiction prix charges"
original_title = '<p style="font-family:Courier; color:Blue; font-size: 42px;">{} </p>'.format(titre)
st.markdown(original_title, unsafe_allow_html=True)

columns_model = ['age', 'sex','children', 'smoker', 'region','cat_bmi']

caracteristique_individue = [0 for _ in range(6)]

age = st.number_input(label="Age :",min_value=18,step=1, value=18, key="age")
caracteristique_individue[0] = age

sexe = st.radio(
     "Votre sexe :",
     ('Homme','Femme' ))
if sexe == 'Homme':
    caracteristique_individue[1] = 'male'
elif sexe == 'Femme':
    caracteristique_individue[1] = 'female'



children  = st.number_input(label="Nombre d'enfant :",value=0,min_value=0, key="children")
caracteristique_individue[2] = children
smoker = st.radio(
     "Vous fumez ? :",
     ('Oui','Non' ))
if smoker == 'Oui':
    caracteristique_individue[3] = 'yes'
elif smoker == 'Non':
    caracteristique_individue[3] = 'no'


liste_region= ['northeast', 'southeast', 'southwest', 'northwest']

region = st.selectbox("Region: ", 
                     liste_region)

caracteristique_individue[4] = region

bmi = st.number_input(label="Bmi :",value=30.5,min_value=10.,step=1.,format="%.2f", key="bmi")
caracteristique_individue[5] = cat_bmi(bmi)


if(st.button("Valider")):
    # st.dataframe(pd.DataFrame(np.array(caracteristique_individue).reshape(1, -1),columns=columns_model))
    predic_lasso = int(modelLasso.predict(pd.DataFrame(np.array(caracteristique_individue).reshape(1, -1),columns=columns_model)))
    predic_ElasticNet = int(modelElastic.predict(pd.DataFrame(np.array(caracteristique_individue).reshape(1, -1),columns=columns_model)))


    st.markdown("<h1 style='text-align: center;'>Modele Lasso</h1>", unsafe_allow_html=True)

    new_title = '<p style="font-family:sans-serif; color:Green;width:100%;text-align:center; font-size: 36px;">{} $</p>'.format(predic_lasso)
    st.markdown(new_title, unsafe_allow_html=True)

   
    st.markdown("<h1 style='text-align: center;'>Modele ElasticNet</h1>", unsafe_allow_html=True)
    new_title = '<p style="font-family:sans-serif; color:Green;width:100%;text-align:center; font-size: 36px;">{} $</p>'.format(predic_ElasticNet)
    st.markdown(new_title, unsafe_allow_html=True)


    st.markdown("<h1 style='text-align: center;'>Moyenne</h1>", unsafe_allow_html=True)
    new_title = '<p style="font-family:sans-serif; color:Green;width:100%;text-align:center; font-size: 36px;">{} $</p>'.format((predic_ElasticNet+predic_lasso)/2)
    st.markdown(new_title, unsafe_allow_html=True)

    