# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 10:16:22 2019

@author: Erinc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 23:40:37 2019

@author: Erinc
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
# Importing the dataset


a = [[0  for i in range(1000)] for j in range(1000)]
ac = [[0  for i in range(1000)] for j in range(1000)]


dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 3:].values
import random
y=[[random.randint(0,2) for i in range(301)] for j in range(121)]   


for i in range(0,120):
    for j in range (0,300):
        
        #c=(X[i][b])
        a[i][j]=(X[j][y[i][j]])



from pandas import DataFrame
df=pd.DataFrame(a)
wrt=pd.ExcelWriter('scenario.xlsx',engine='xlsxwriter')
df.to_excel(wrt)
wrt.save()







