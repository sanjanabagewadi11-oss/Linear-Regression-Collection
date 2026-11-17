 
'''
Company :Crevavi
Author  :Sanjana Bagewadi
Date    : 25/09/2026
Purpose : Poc of Linear Regression ML Algorithm
'''
 
# Importing Modules which are required
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
 
st.write(F'###  House Rent Predictor')
 
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([" ", "Complete DataSet", "Top 5 Rows", \
                                        "Bottom 5 Row", "Stastics", "Graph", "Accuracy"])
 
# Load dataset
df = pd.read_csv('House_Rent_Dataset.csv')
 
# Converts string type to numeric code
df['City'] = df['City'].astype('category')
df['City_Code'] = df['City'] .cat.codes
 
df['Area Locality'] = df['Area Locality'].astype('category')
df['Area_Locality_Code'] = df['Area Locality'] .cat.codes
 
df['Area Type'] = df['Area Type'].astype('category')
df['Area_Type_Code'] = df['Area Type'] .cat.codes
 
# Display the entire dataset in dataframe
with tab1:
    st.write("### Full House Rent Dataset")
    st.write(df)
 
# To 5 Rows of House Rent Dataset will be displayed
with tab2:
    st.write("### Top 5 Rows of House Rent Dataset")
    st.dataframe(df.head())
 
# Botton 5 Rows of House Rent Dataset will be displayed
with tab3:
    st.write("### Bottom 5 Rows of House Rent Dataset")
    st.dataframe(df.tail())
 
with tab4:
    st.write("### Satstics of House Rent Dataset")
    st.dataframe(df.describe())
 
with tab5:
    st.write("### Scatter Graph")
    fig, ax = plt.subplots()
    ax.scatter(df['City'], df['Rent'], color='red')
    ax.set_title('Scatter Plot for Location Vs Rent')
    ax.set_xlabel('Location')
    ax.set_ylabel('Rent')
    st.pyplot(fig)
 
#data cleaning in case of None Value It will drop the row
df = df.dropna()
 
# Features        [Independ Variables]
X = df[['City_Code', 'Area_Locality_Code', 'Area_Type_Code', 'Size', 'BHK']]
 
# Target Variable [ Dependant Variable ]
Y = df[['Rent']]                
 
# Training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state=42)
# Selection of LinearRegression Model
model = LinearRegression()
model.fit(X_train, Y_train)
 
st.sidebar.header("Select Data for Prediction")
 
temp_city = st.sidebar.selectbox("Select a Location : ", df['City'].unique())
 
temp_area_locality = st.sidebar.selectbox("Select a Area Locality: ", df['Area Locality'].unique())
temp_area_type = st.sidebar.selectbox("Select a Area Type: ", df['Area Type'].unique())
 
temp_size = st.sidebar.selectbox("Select a Size : ", df['Size'].unique())
temp_bhk = st.sidebar.selectbox("Select BHK: ", df['BHK'].unique())
 
if temp_city in df['City'].cat.categories:
        temp_city_code = df['City'].cat.categories.get_loc(temp_city)
 
if temp_area_locality in df['Area Locality'].cat.categories:
        temp_area_locality_code = df['Area Locality'].cat.categories.get_loc(temp_area_locality)
 
if temp_area_type in df['Area Type'].cat.categories:
        temp_area_type_code = df['Area Type'].cat.categories.get_loc(temp_area_type)
 
# Data for Prediction      
ft_location = pd.DataFrame({'City_Code':[temp_city_code],\
                            'Area_Locality_Code':[temp_area_locality_code],\
                            'Area_Type_Code':[temp_area_type_code],\
                            'Size':[temp_size],'BHK':[temp_bhk]})
 
#Prediction for Given Data
rent_pred = model.predict(ft_location)
 
# To display Output
st.markdown(F"<h5> Entered Data for Prediction : <br><br>Location :  {temp_city} \
             <br>Area Locality: {temp_area_locality} <br>Area Type : {temp_area_type} \
             <br>Area: {temp_size} <br>Rooms : {temp_bhk} \
             <br><br>Rent Predicted : {rent_pred}</h5>", unsafe_allow_html=True )
 
with tab6:
    b = model.predict(X_test)
    c = Y_test.to_numpy()
    mse = mean_squared_error(c, b)
    st.write(mse)
    st.markdown(r2_score(c, b))