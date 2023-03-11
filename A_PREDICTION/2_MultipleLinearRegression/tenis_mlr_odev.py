#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 00:21:55 2022

@author: zex
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# data import
data = pd.read_csv("tenis.csv")

#%%
# categoric data - encoding
from sklearn import preprocessing

# label encoding
outlook_lbl = preprocessing.LabelEncoder()
outlook = data.iloc[:, 0:1].values
outlook[:,0] = outlook_lbl.fit_transform(outlook[:, 0])
print(outlook)

# one hot encoding
outlook_oh = preprocessing.OneHotEncoder()
outlook_fit = outlook_oh.fit_transform(outlook).toarray()
print(outlook_fit)

#****************************************

# label encoding
play_lbl = preprocessing.LabelEncoder()
play = data.iloc[:, -1:].values
play[:, -1] = play_lbl.fit_transform(play[:, -1])
print(play)

# one hot encoding
play_oh = preprocessing.OneHotEncoder()
play_fit = play_oh.fit_transform(play).toarray()
print(play_fit)

#****************************************

# label encoding
windy_lbl = preprocessing.LabelEncoder()
windy = data.iloc[:, -2:-1].values
windy[:, 0] = windy_lbl.fit_transform(windy[:, 0])
print(windy)

# one hot encoding
windy_oh = preprocessing.OneHotEncoder()
windy_fit = windy_oh.fit_transform(windy).toarray()
print(windy_fit)

#%%
# dataframe creating and assamble (concat())

numeric_vals = data.iloc[:, 1:3].values
print(numeric_vals)

df_outlook = pd.DataFrame(data= outlook_fit, index= range(14), columns= ["overcast", "rainy", "sunny"])
print(df_outlook)
df_numeric = pd.DataFrame(data= numeric_vals, index= range(14), columns= ["temperature", "humidity"])
print(df_numeric)
df_windy = pd.DataFrame(data= windy_fit[:, 1], index= range(14), columns= ["windy"])
print(df_windy)
df_play = pd.DataFrame(data= play_fit[:, 1], index= range(14), columns= ["play"])
print(df_play)

conc1 = pd.concat([df_outlook, df_numeric, df_windy], axis= 1)
print(conc1)

#%%
# dataset splitting - train/test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(conc1, df_play, test_size= 0.33, random_state= 0)

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
X = np.append(arr = np.ones((14,1)).astype(int), values = conc1, axis = 1)

# eliminasyon listesi
Xlist = np.array(conc1.iloc[:,[0,1]].values, dtype = (float))

# Xlist içinde yer alan her kolonun df_gender kolonuna etkisi ölçülür
# modelin başarısı ölçülür
# df_gender bağımlı değişken iken Xlist bağımsız değişkendir
model = sm.OLS(df_play, Xlist).fit()

print(model.summary())

x_train = x_train.iloc[:, :2]
x_test = x_test.iloc[:, :2]

const.fit(x_train, y_train)
prediction_optimized = const.predict(x_test)














