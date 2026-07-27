import streamlit as st
import pandas as pd
import joblib

model = joblib.load('LinearRegression.joblib')
ohe = joblib.load('OneHotEncoder.joblib')
scaler = joblib.load('StandardScaler.joblib')

st.title('House Prediction Aap')
date = st.date_input('date')
bedrooms = st.number_input('bedrooms')
bathrooms = st.number_input('bathrooms')
sqft_living = st.number_input('sqft living')
sqft_lot = st.number_input('sqft lot')
floors = st.number_input('floors')
waterfront = st.selectbox('water front', ['yes', 'no'])
view = st.number_input('view')
condition = st.number_input('condition')
sqft_above = st.number_input('sqft above')
sqft_basement = st.number_input('sqft basement')
yr_built = st.number_input('yr built')
yr_renovated = st.number_input('yr renovated')
street = st.text_input('street')
city = st.text_input('city')
statezip = st.text_input('statezip')
country = st.text_input('country')
data_user = pd.DataFrame({
    'date' : [date],
    'bedrooms' : [bedrooms],
    'bathrooms' : [bathrooms],
    'sqft_living' : [sqft_living],
    'sqft_lot' : [sqft_lot],
    'floors' : [floors],
    'waterfront' : [waterfront],
    'view' : [view],
    'condition' : [condition],
    'sqft_above' : [sqft_above],
    'sqft_basement' : [sqft_basement],
    'yr_built' : [yr_built],
    'yr_renovated' : [yr_renovated],
    'street' : [street],
    'city' : [city],
    'statezip' : [statezip],
    'country' : [country]
})
if st.button('predict'):
    
    encoder = ohe.transform(data_user)
    scaler = scaler.transform(encoder)
    prediction = model.predict(scaler)

    st.write("Predicted House Price:", prediction[0])