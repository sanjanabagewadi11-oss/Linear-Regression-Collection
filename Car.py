'''
Problem : Car Price Class Prediction
Algorithm : KNN
'''
 
import pandas as pd
import streamlit as st
 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
 
from data_cleaning import clean_dataset
 
st.title("Car Price Prediction - KNN")
 
upload_file = st.file_uploader("Upload Car Price CSV Dataset", type=["csv"])
 
if upload_file is not None:
 
    df = pd.read_csv(upload_file)
 
    # CREATE PRICE CLASS BEFORE CLEANING
    df["Price_Class"] = pd.cut(
        df["Selling_Price"],
        bins=[-float("inf"), 3, 6, float("inf")],
        labels=["Low", "Medium", "High"]
    )
 
    # TABS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Complete Data", "Describe", "Clean Data",
        "Top 5 Rows", "Bottom 5 Rows", "Analysis of Data"
    ])
 
    with tab1:
        st.dataframe(df)
 
    with tab2:
        st.dataframe(df.describe(include="all"))
 
    # USE YOUR CLEANING MODULE
    cleaned_df = clean_dataset(df.copy())
 
    with tab3:
        st.dataframe(cleaned_df)
 
    with tab4:
        st.dataframe(df.head())
 
    with tab5:
        st.dataframe(df.tail())
 
    with tab6:
        st.write("Selling Price Analysis")
        st.bar_chart(df["Selling_Price"])
 
    # PREPARE DATA
    ml_df = cleaned_df.dropna().copy()
 
    X = ml_df.drop(columns=["Selling_Price", "Price_Class", "Car_Name"])
    Y = ml_df["Price_Class"]
 
    # CONVERT TEXT INPUTS TO NUMBERS
    encoders = {}
 
    for col in X.select_dtypes(include=["object", "category"]).columns:
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col].astype(str))
 
    # ENCODE PRICE CLASS
    price_encoder = LabelEncoder()
    Y = price_encoder.fit_transform(Y.astype(str))
 
    # SPLIT DATA
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
 
    # SCALE DATA
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
 
    # KNN
    k = st.sidebar.number_input(
        "Enter K Value", min_value=1, max_value=len(X_train), value=3
    )
 
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, Y_train)
 
    pred = model.predict(X_test)
 
    # RESULTS
    st.subheader("KNN Results")
    st.write("Accuracy:", round(accuracy_score(Y_test, pred) * 100, 2), "%")
 
    # PREDICTION
    st.subheader("Predict Car Price Class")
    st.sidebar.header("Enter Car Details")
 
    input_data = []
 
    for col in X.columns:
        value = st.sidebar.number_input(
            col, value=float(ml_df[col].median())
        )
        input_data.append(value)
 
    if st.sidebar.button("Predict"):
        new_data = pd.DataFrame([input_data], columns=X.columns)
        new_data = scaler.transform(new_data)
 
        result = model.predict(new_data)[0]
        result = price_encoder.inverse_transform([result])[0]
 
        st.success("Predicted Price Class: " + result)