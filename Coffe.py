'''
Predicting Coffee Sales Based on Temperature  

Month                              Sales
5                                   120
10                                  110
15                                  100
20                                  90
25                                  80
30                                  70
'''

import numpy as np
import pandas as pd
 
 
df = pd.DataFrame({
    "x": [5,10,15,20,25,30],
    "y": [120,110,100,90,80,70]
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
 
 
x1= 18
prediction = m * x1+ c
 
print(df)
 
print("Slope:", m)
print("Intercept:", c)
print("Prediction:", prediction)