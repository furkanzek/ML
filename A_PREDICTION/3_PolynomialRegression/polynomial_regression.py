#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 16:02:37 2023

@author: zex
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# Öncelikle, gerekli kütüphaneleri ve veri setini içe aktarın.
data = pd.read_csv("maaslar.csv")

#%%
# Veri setinizi inceleyin ve özellikleri ve hedef değişkeni olarak ayırın.
x = data.iloc[:, 1:2]
y = data.iloc[:, 2:3] 
X = x.values
Y = y.values

#%%
# Özelliklerin polynomial özelliklerine dönüştürülmesi için PolynomialFeatures sınıfını kullanın.
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)

#%%
# Dönüştürülmüş polynomial özellikleri kullanarak regresyon modelini eğitin. 
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

#%%
# Eğitilmiş modeli kullanarak tahminler yapın.
y_pred =lin_reg.predict(poly_reg.fit_transform(X))

#%%
# Veri görselleştirme
plt.scatter(X, Y)
plt.plot(X, y_pred)
plt.show()

















