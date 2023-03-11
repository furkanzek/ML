#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 02:44:27 2022

@author: zex
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# data import
data = pd.read_csv("veriler.csv")

#%%
# categoric data - encoding
from sklearn import preprocessing

# label encoding
country_lbl = preprocessing.LabelEncoder()
country = data.iloc[:, 0:1].values
country[:,0] = country_lbl.fit_transform(country[:, 0])
print(country)

# one hot encoding
country_oh = preprocessing.OneHotEncoder()
country_fit = country_oh.fit_transform(country).toarray()
print(country_fit)

#****************************************

# label encoding
g_lbl = preprocessing.LabelEncoder()
g = data.iloc[:, -1:].values
g[:, -1] = g_lbl.fit_transform(g[:, -1])
print(g)

# one hot encoding
g_oh = preprocessing.OneHotEncoder()
g_fit = g_oh.fit_transform(g).toarray()
print(g_fit)

#%%
# dataframe creating and assamble (concat())

numeric_vals = data.iloc[:, 1:4].values
print(numeric_vals)

df_country = pd.DataFrame(data= country_fit, index= range(22), columns= ["fr", "tr", "us"])
print(df_country)
df_numeric = pd.DataFrame(data= numeric_vals, index= range(22), columns= ["boy", "kilo", "yas"])
print(df_numeric)
df_gender = pd.DataFrame(data= g_fit[:, 0], index= range(22), columns= ["cinsiyet"])
print(df_gender)

conc1 = pd.concat([df_country, df_numeric], axis= 1)
print(conc1)

#%%
# dataset splitting - train/test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(conc1, df_gender, test_size= 0.33, random_state= 0)

#%%
# model creating for multiple linear regression
from sklearn.linear_model import LinearRegression

const = LinearRegression()
const.fit(x_train, y_train)
prediction = const.predict(x_test)

#%%
# backward elimination
import statsmodels.api as sm

'''
istatistiksel değerleri bulmak içindir

1- multiple linear regression fonksiyonundaki ß0 sabit değerini oluşturmak 
   için 1'lerden oluşan bir liste hazırlanıp datasetin sonuna ya da başına eklenir.

2- model oluşturulur.

3- model çıktısı alınır.
'''

# 1'lerden oluşmuş listenin ana listeye eklenmiş hali
X = np.append(arr = np.ones((22,1)).astype(int), values = conc1, axis = 1)

# eliminasyon listesi
Xlist = np.array(conc1.iloc[:,[0,1,2,3,4,5]].values, dtype = (float))

# Xlist içinde yer alan her kolonun df_gender kolonuna etkisi ölçülür
# modelin başarısı ölçülür
# df_gender bağımlı değişken iken Xlist bağımsız değişkendir
model = sm.OLS(df_gender, Xlist).fit()

print(model.summary())

# eliminasyon listesi
Xlist = np.array(conc1.iloc[:,[1,3,4]].values, dtype = (float))

# Xlist içinde yer alan her kolonun df_gender kolonuna etkisi ölçülür
# modelin başarısı ölçülür
# df_gender bağımlı değişken iken Xlist bağımsız değişkendir
model = sm.OLS(df_gender, Xlist).fit()

print(model.summary())



























