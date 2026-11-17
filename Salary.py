''''
Problem 8: Predicting Employee Salary Based on Years of Experience
 
Dataset:
 
Years Experience (x)        Salary ($1000s) (y)
 
    0                             40
 
    2                             50
 
    4                             60
 
    6                             70
 
    8                             80
     
    5                             ?
 
Find: Salary for 5 years of experience
 
'''
 
 
import numpy as np
 
x = np.array([0,2,4,6,8,5])
y = np.array([40,50,60,70,80])
 
x1 = np.mean(x)
y1 = np.mean(y)
print(x1)
print(y1)
 
c3 = x - x1
c4 = y - y1
print(c3)
print(c4)
 
c5 = c3**c4
print(c5)
 
c6 = c3**2
print(c6)
 
m=np.sum(c5)/np.sum(c6)
print("Slope:",m)
 
c=y1-(m*x1)
print("Intercept:",c)
 
X=5
y=m*X+c
print("predicted Salary:",y)