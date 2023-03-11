#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 02:33:50 2023

@author: zex
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("musteriler.csv")

x = data.iloc[:, 3:].values

from sklearn.cluster import KMeans

sonuclar = []
for i in range(1,11):
    kmn = KMeans(n_clusters=i, init="k-means++", random_state=6666)
    kmn.fit(x)
    sonuclar.append(kmn.inertia_)

plt.plot(range(1,11), sonuclar)


kmn = KMeans(n_clusters=2, init="k-means++")
kmn.fit(x)
print(kmn.cluster_centers_)