'''
 A company wants to predict its sales based on the amount spent on advertising. 
 The data collected for 5 months is as follows:
 The task is to find the Linear Regression Equation that models the relationship between Advertising Spend and Sales, and use it to predict sales if the advertising spend is 6.0 thousand. 
Month           Spend          Sales
1                1.0            2.0
2                2.0            2.5
3                3.0            3.5
4                4.0            4.0
5                5.0            5.0
'''
import numpy as np
import pandas as pd
 
 
df = pd.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [2, 2.5, 3.5, 4, 5]
})
print(df)
 
c1=df["x"].mean()
c2= df["y"].mean()
 
 
df["C3"] = df["x"] - c1
df["C4"] = df["y"] -c2
 
df["C5"] = df["C3"] * df["C4"]
df["C6"] = df["C3"] ** 2
 
m = np.sum(df["C5"]) / np.sum(df["C6"])
 
c = c2- (m * c1)
 
 
x1= 6
prediction = m * x1+ c
 
print(df)
 
print("Slope:", m)
print("Intercept:", c)
print("Prediction:", prediction)