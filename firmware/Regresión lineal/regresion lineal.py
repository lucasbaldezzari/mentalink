# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 13:15:43 2021

@author: enfil
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def recta(m, oy):
    yvalues = []
    x = list(range(0, 60))
    for i in x:
        y = m*x+oy
        yvalues.append(y)
    return x, yvalues
        

x = np.array([5,15,25,35,45,55]).reshape((-1,1))
y = np.array([5,20,14,32,22,38])
model = LinearRegression()
model.fit(x,y)

pendiente = model.coef_
ordenada = model.intercept_

print(f"pendiente {model.coef_}")
print(f"ordenada {model.intercept_}")


xval, yval = recta(pendiente, ordenada)


plt.plot(xval, yval[1])
plt.scatter(x, y, edgecolors = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regresión lineal')


prediccion = model.predict(x)
print(prediccion)