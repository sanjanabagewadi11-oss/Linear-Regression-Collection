'''
Company : Crevavi,Belgaum
Author  : Sanjana U Bagewadi
Date    : 26/09/2026
Purpose : Poc of Linear Regression ML Algorithm
        : Multiple Variables version 1.0
'''

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.write(F"Car price prdiction") 

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["Dataset","Top 5 rows","Bottom 5 rows","Statistics of Dataset","Scatter plot","Prediction"])

#loading dataset
df = pd.read_csv("C:/Users/Sanjana/Downloads/ToyotaCorolla - MLR.csv")


with tab1 :
    st.write("Car Dataset")
    st.write(df)

with tab2 :
       st.write("Top 5 Rows of Car Dataset")
       st.dataframe(df.head())

with tab3:
    st.write("Bottom 5 Rows of House Rent Dataset")
    st.dataframe(df.tail())

    
with tab4:
    st.write("Satstics of Car Dataset")
    st.dataframe(df.describe())

with tab5:
    st.write("Scatter Graph")
    fig, ax = plt.subplots()
    ax.scatter(df['Age_08_04'], df['Price'], color='red')
    ax.set_title('Scatter Plot for Age_08_04 Vs Price')
    ax.set_xlabel('Age_08_04')
    ax.set_ylabel('Price')
    st.pyplot(fig)


#independent variables 
X = df[['Age_08_04', 'KM', 'HP']]
#depedent variab les
Y = df['Price']

#training n testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,
                                                    random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)


st.sidebar.header("Select Data for Prediction")


Age = st.sidebar.selectbox("Select age : ", df['Age_08_04'].unique())
KM = st.sidebar.selectbox("Select KM driven :", df['KM'].unique())
HP = st.sidebar.selectbox("Select HP :", df['HP'].unique())

input_data = pd.DataFrame([[Age, KM, HP]], columns=['Age_08_04', 'KM', 'HP'])
prediction = model.predict(input_data)

st.write("Predicted Price:", round(prediction[0],2))

# Create and train model
model = LinearRegression()
model.fit(X_train, Y_train)

with tab6: 
    b = model.predict(X_test)
    c = Y_test.to_numpy()
    mse = mean_squared_error(c, b)
    st.write("Mean Squared Error")
    st.write(mse)
    st.write("R2 Score")
    st.markdown(r2_score(c, b))
    



