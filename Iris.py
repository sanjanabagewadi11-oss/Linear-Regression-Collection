'''
Problem : Prediction of Species of Flower
Algorithm :Logistic Prediction.
'''
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
 
from data_cleaning import clean_dataset
 
st.title("Iris Flower Prediction - Logistic Regression")
 
 
iris = load_iris()
 
df = pd.DataFrame(iris.data, columns=["Sepal Length","Sepal Width", "Petal Length","Petal Width"])
 
df["Species"] = iris.target
 
df["Species"] = df["Species"].map({0: "Setosa",1: "Versicolor",2: "Virginica"})
 
 
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([ "Complete Data", "Describe","Clean Data","Top 5 rows","Bottom 5 rows","Analysis of Data"])
 
with tab1:
    st.write("Complete Dataset")
    st.dataframe(df)
 
with tab2:
    st.write("Dataset Statistics")
    st.dataframe(df.describe())
 
with tab3:
    st.write("Clean Data")
    cleaned_df = clean_dataset(df.copy())
    st.dataframe(cleaned_df, use_container_width=True)
 
with tab4:
    st.write("Top 5 Rows")
    st.dataframe(df.head())
 
with tab5:
    st.write("Bottom 5 Rows")
    st.dataframe(df.tail())
 
with tab6:
    st.write("Species Count")
    st.bar_chart(df["Species"].value_counts())
 
ml_df = df.copy()
 
ml_df["Species"] = ml_df["Species"].map({"Setosa": 0,"Versicolor": 1,"Virginica": 2})
 
X = ml_df[["Sepal Length","Sepal Width","Petal Length","Petal Width"]]
 
Y = ml_df["Species"]
 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2,random_state=42)
 
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
 
pred = model.predict(X_test)
proba = model.predict_proba(X_test)
 
accuracy = accuracy_score(Y_test, pred)
 
st.subheader("Logistic Regression Results")
st.write("Accuracy:", round(accuracy * 100, 2), "%")
 
 
st.subheader("Predict Iris Flower")
sepal_length = st.sidebar.number_input("Sepal Length", 0.0, 10.0, 5.1)
 
sepal_width = st.sidebar.number_input("Sepal Width", 0.0, 10.0, 3.5)
petal_length = st.sidebar.number_input("Petal Length", 0.0, 10.0, 1.4)
petal_width = st.sidebar.number_input("Petal Width", 0.0, 10.0, 0.2)
 
if st.sidebar.button("Predict"):
    input_data = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]], columns=X.columns)
    result = model.predict(input_data)[0]
    species_names = {0: "Setosa",1: "Versicolor",2: "Virginica"}
    st.success("Predicted Species: " + species_names[result])