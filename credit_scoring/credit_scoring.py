#%%
from ucimlrepo import fetch_ucirepo 

import numpy as np
import pandas as pd

#%%
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
df = pd.concat([X, y], axis=1)

# metadata 
print(statlog_german_credit_data.metadata) 
  
# variable information 
print(statlog_german_credit_data.variables) 

#%%
from collections import Counter
Counter(y['class'])

#%%
# 1. Dataset information
df.head()
#%%
print (f"Any NaN Value? {df.isna().any().any()}")
print (f"{df.shape=}")
df.info()

# %%
# Inspecting continous data distribution

#%%
# Inspecting categorical data distribution

#%%
# Handling multivariate features

#%%
# Target distibution

# Since we have an imbalanced dataset with few samples -> Oversampling
#%%
# Splitting data

#%%
# Oversampling

#%%
# Correlation matrix

#%%
# Feature selection (if needed)

#%%
# Dimensionality Visualization

#%%
# Classifier task

#%% 
# GridSearch - RandomSearch

#%%
# Post-tuning the decision threshold for cost-sensitive learning
