 
'''
Problem : EV Vehicle Sales Prediction
Algorithm : Linear Regression
'''
 
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
 
st.title("EV Vehicle Sales Prediction")
 
# Upload Dataset
file = st.file_uploader("Upload EV Sales CSV File", type=["csv"])
 
if file is not None:
 
    # Read dataset only once
    df = pd.read_csv(file)
 
    # Check required columns
    required_columns = [
        "Year",
        "Vehicle_Category",
        "Segment",
        "Sales_Quantity"
    ]
 
    if not all(col in df.columns for col in required_columns):
        st.error("The uploaded CSV does not contain the required columns.")
        st.write("Available columns:", df.columns.tolist())
 
    else:
        # Remove rows with missing values in required columns
        df = df.dropna(subset=required_columns).copy()
 
        if len(df) < 5:
            st.error("Dataset must contain at least 5 valid rows.")
 
        else:
            # Save category names before converting
            category_names = sorted(df["Vehicle_Category"].unique())
            segment_names = sorted(df["Segment"].unique())
 
            # Convert categorical columns into numbers
            category_codes = {
                name: code for code, name in enumerate(category_names)
            }
 
            segment_codes = {
                name: code for code, name in enumerate(segment_names)
            }
 
            df["Vehicle_Category"] = df["Vehicle_Category"].map(category_codes)
            df["Segment"] = df["Segment"].map(segment_codes)
 
            # Convert input and output columns to numeric
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            df["Sales_Quantity"] = pd.to_numeric(
                df["Sales_Quantity"], errors="coerce"
            )
 
            df = df.dropna(
                subset=["Year", "Vehicle_Category", "Segment", "Sales_Quantity"]
            )
 
            if len(df) < 5:
                st.error("Not enough valid data to train the model.")
 
            else:
                # Input and output
                X = df[["Year", "Vehicle_Category", "Segment"]]
                Y = df["Sales_Quantity"]
 
                # Split dataset
                X_train, X_test, Y_train, Y_test = train_test_split(
                    X, Y, test_size=0.2, random_state=42
                )
 
                # Train model
                model = LinearRegression()
                model.fit(X_train, Y_train)
 
                # Test prediction
                pred = model.predict(X_test)
 
                # Tabs
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "Complete Dataset",
                    "Top 5 Rows",
                    "Bottom 5 Rows",
                    "Statistics",
                    "Graph",
                    "Prediction"
                ])
 
                with tab1:
                    st.subheader("Complete Dataset")
                    st.dataframe(df)
 
                with tab2:
                    st.subheader("Top 5 Rows")
                    st.dataframe(df.head())
 
                with tab3:
                    st.subheader("Bottom 5 Rows")
                    st.dataframe(df.tail())
 
                with tab4:
                    st.subheader("Dataset Statistics")
                    st.dataframe(df.describe())
 
                with tab5:
                    st.subheader("EV Sales by Year")
 
                    chart_data = df.groupby("Year")[
                        "Sales_Quantity"
                    ].mean()
 
                    fig, ax = plt.subplots()
                    ax.plot(
                        chart_data.index,
                        chart_data.values,
                        marker="o",
                        color="blue"
                    )
 
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Sales Quantity")
                    ax.set_title("EV Sales by Year")
                    ax.grid(True)
 
                    st.pyplot(fig)
 
                with tab6:
                    st.subheader("Predict EV Sales")
 
                    year = st.number_input(
                        "Enter Year",
                        min_value=2015,
                        max_value=2035,
                        value=2025
                    )
 
                    category = st.selectbox(
                        "Select Vehicle Category",
                        category_names
                    )
 
                    segment = st.selectbox(
                        "Select Segment",
                        segment_names
                    )
 
                    if st.button("Predict Sales"):
 
                        # Prepare input using the same encoding
                        input_data = pd.DataFrame({
                            "Year": [year],
                            "Vehicle_Category": [
                                category_codes[category]
                            ],
                            "Segment": [
                                segment_codes[segment]
                            ]
                        })
 
                        result = model.predict(input_data)
 
                        st.success(
                            "Predicted EV Sales Quantity: "
                            + str(result[0])
                        )
 
                # Model evaluation
                st.subheader("Model Evaluation")
 
                st.write(
                    "Mean Squared Error:",
                    mean_squared_error(Y_test, pred)
                )
 
                st.write(
                    "R2 Score:",
                    r2_score(Y_test, pred)
                )
 
else:
    st.info("Please upload your Indian EV Sales CSV file.")