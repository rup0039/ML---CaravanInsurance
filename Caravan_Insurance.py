# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:40:13 2022

@author: Rupesh Roy
"""

#from sklearn.datasets import fetch_openml

# 1.0 Import libraries

import os
import pandas as pd

# 1.1 Load data

os.chdir("F:\\SPBT Training\\20200725-26\\Exercise")
df = pd.read_csv("caravan-insurance-challenge.csv")

# 1.2 Analyse the dataset

df.info() 
df.isnull().values.any() # There is no NULL value
df.isna().values.any()   # There is no NA  

# 2.0 Seperate train test 

df["ORIGIN"].value_counts()
X_train = df[df["ORIGIN"]=="train"]
X_test  = df[df["ORIGIN"]=="test"]

y_train = X_train.pop("CARAVAN")

X_train.shape
X_test.shape

# 2.1 Check data is balanced or not

y_train.value_counts()

#X_train,X_val,y_train,y_val = train_test_split(X_train, y_train,test_size=0.2, shuffle=True,stratify = y_train,random_state = 42)

# Feature Scaling
# All features are categorical variables out of which Except MOSTYPE and MOSHOOFD all are
# ordinal and represent correct order between categories, therefore we will encode only these
# two features. We will use Binary Encoding.

