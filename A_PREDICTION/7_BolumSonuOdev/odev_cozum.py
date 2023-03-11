#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 23:09:56 2023

@author: zex
"""

#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


#veri yukleme
veriler = pd.read_csv('maas.csv')

x = veriler.iloc[:,2:5]
y = veriler.iloc[:,5:]
X = x.values
Y = y.values

print('CORRELATION')
print(veriler.corr())
print('*-------------------------------------------------*')

#linear regression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)


model=sm.OLS(lin_reg.predict(X),X)
print('\nLinear OLS')
print(model.fit().summary())

print('\nLinear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))

print('*-------------------------------------------------*')
#polynomial regression

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


print('\nPolynomial OLS')
model2=sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

print('\nPolynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

print('*-------------------------------------------------*')

#verilerin olceklenmesi


sc1=StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


#SVR Regresyon

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)


print('\nSVR OLS')
model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())


print('\nSVR R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

print('*-------------------------------------------------*')

#Decision Tree Regresyon

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


print('\nDecision Tree OLS')
model4=sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())

print('\nDecision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))

print('*-------------------------------------------------*')

#Random Forest Regresyonu

rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)
rf_reg.fit(X,Y.ravel())


print('\nRandom Forest OLS')
model5=sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())



print('\nRandom Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))


#Ozet R2 değerleri
print('\n\n*-------------------------------------------------*')
print('                       ÖZET                        ')
print('*-------------------------------------------------*')
print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))

print('\nPolynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

print('\nSVR R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


print('\nDecision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))

print('\nRandom Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))














