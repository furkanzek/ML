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
# Random Forest modeli oluşturun.
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=0)
forest_reg.fit(X, Y.ravel())

print(forest_reg.predict([[2.5]]))

#%% 
# R2 skorunu hesaplayalım.
from sklearn.metrics import r2_score

print(r2_score(Y, forest_reg.predict(X)))











