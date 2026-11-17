'''
Predicting Monthly Electricity Bill Based on Temperature 

Month                              Sales
60                                  80
70                                  95
80                                  110
90                                  125
100                                 140
'''

import numpy as np
import pandas as pd
 
 
df = pd.DataFrame({
    "x": [60,70,80,90,100],
    "y": [80,95,110,125,140]
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
 
 
x1= 85
prediction = m * x1+ c
 
print(df)
 
print("Slope:", m)
print("Intercept:", c)
print("Prediction:", prediction)