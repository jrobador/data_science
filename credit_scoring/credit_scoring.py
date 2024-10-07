#%%
import os
os.makedirs('./credit_scoring/plots', exist_ok=True)
os.makedirs('./credit_scoring/models', exist_ok=True)

from ucimlrepo import fetch_ucirepo 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, OneHotEncoder

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
#%%
# How many are continuous and how many are categorical?
continuous_features  = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

print(f"Number of continuous features: {len(continuous_features)}")
print(f"Number of categorical features: {len(categorical_features)}")

print("\nContinuous features:")
print(continuous_features)

print("\nCategorical features:")
print(categorical_features)
# %%
# Inspecting continous data distribution
std = {}
for feature in (continuous_features):
    plt.figure(figsize=(8, 5)) 
    sns.histplot(x=X[feature], color='b', kde=True, linewidth=1.2, alpha=1)
    
    plt.xlabel(feature.replace('-', ' ').capitalize(), fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim([min(X[feature].values), max(X[feature].values)])
    
    filename = "dist_" + feature + ".png"
    file_path = os.path.join('./credit_scoring/plots', filename)
    if not os.path.exists(file_path):
        plt.savefig(file_path)
    plt.show()

    std[feature] = X[feature].std()

#%%
# Standard deviation
print ("Standard deviation of features:")
for key, value in zip(std.keys(), std.values()):
    print (f"{key}={value:.2f}")
#%%
# Inspecting categorical data distribution
for feature in (categorical_features):
    plt.figure(figsize=(8, 5))
    sns.countplot(y=feature, data=df, order=df[feature].value_counts().index)
    plt.tight_layout()

    filename = "cplot_" + feature + ".png"
    file_path = os.path.join('./credit_scoring/plots', filename)
    if not os.path.exists(file_path):
        plt.savefig(file_path)
    plt.show()

#%%
# Handling multivariate features
X.loc[:, continuous_features] = RobustScaler().fit_transform(X[continuous_features])

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
