#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 11:44:31 2023

@author: zex
"""

import pandas as pd
import numpy as np

data = pd.read_csv('sepet.csv', header=None)

cl = []
for row in range(0, len(data.axes[0])):
    cl.append([str(data.values[row, col]) for col in range(0, len(data.axes[1]))]) 
    
from apyori import apriori

assoc = apriori(cl, min_support = 0.01, min_confidence = 0.02, min_lift = 3, min_length = 2)
print(list(assoc))