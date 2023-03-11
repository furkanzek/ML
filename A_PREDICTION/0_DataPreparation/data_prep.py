#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 00:22:37 2022

@author: zex
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# data import
data = pd.read_csv("veriler.csv")

#%%
# data preprocess
yas = data[["yas"]]
print(yas)

#%%
# missing data
from sklearn.impute import SimpleImputer

missing_data = pd.read_csv("eksikveriler.csv")
imputer = SimpleImputer(missing_values= np.nan, strategy= "mean")
numeric_vals = missing_data.iloc[:, 1:4].values
print(numeric_vals)
imputer_fit = imputer.fit(numeric_vals[:,1:4])
numeric_vals[:, 1:4] = imputer_fit.transform(numeric_vals[:,1:4])
print(numeric_vals)

#%%
# categoric data - encoding
from sklearn import preprocessing

# label encoding
lbl_encoder = preprocessing.LabelEncoder()
country = data.iloc[:, 0:1].values
country[:,0] = lbl_encoder.fit_transform(country[:, 0])
print(country)

# one hot encoding
oh_encoder = preprocessing.OneHotEncoder()
country_fit = oh_encoder.fit_transform(country).toarray()
print(country_fit)

#%%
# dataframe creating and assamble (concat())

df_country = pd.DataFrame(data= country_fit, index= range(22), columns= ["fr", "tr", "us"])
df_numeric = pd.DataFrame(data= numeric_vals, index= range(22), columns= ["boy", "kilo", "yas"])
gender = data.iloc[:, -1].values
df_gender = pd.DataFrame(data= gender, index= range(22), columns= ["cinsiyet"])

conc1 = pd.concat([df_country, df_numeric], axis= 1)
print(conc1)

#%%
# dataset splitting - train/test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(conc1, df_gender, test_size= 0.33, random_state= 0)

#%%
# data scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)














