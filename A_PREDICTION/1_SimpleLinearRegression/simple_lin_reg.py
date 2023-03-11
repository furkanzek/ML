#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 23:34:51 2022

@author: zex
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# data import
data = pd.read_csv("satislar.csv")

#%%
# data preprocess
aylar = data[["Aylar"]]
print(aylar)

satislar = data[["Satislar"]]
print(satislar)

#%%
# dataset splitting - train/test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size= 0.33, random_state= 0)

#%%
# data scaling
"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)
Y_train = scaler.fit_transform(y_train)
Y_test = scaler.fit_transform(y_test)
"""
#%%
# model creating for simple linear regression
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(x_train, y_train)

#%%
# prediction
tahmin = linreg.predict(x_test)

#%%
# data visualization
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train, y_train)
plt.plot(x_test, tahmin)

#%% 
# R2 skorunu hesaplayalım.
from sklearn.metrics import r2_score

print(r2_score(y_test, linreg.predict(x_test)))

