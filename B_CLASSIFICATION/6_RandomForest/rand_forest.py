#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 19:57:33 2023

@author: zex
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# data import
data = pd.read_csv("veriler.csv")


x=data.iloc[:,1:4].values
y=data.iloc[:,4:].values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.33, random_state= 0)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)


from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=15, criterion="entropy")
rfc.fit(X_train, y_train)

y_pred=rfc.predict(X_test)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)






