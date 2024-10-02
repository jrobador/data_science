#%%
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import os
os.makedirs('./plots', exist_ok=True)
os.makedirs('./models', exist_ok=True)

#%%
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# metadata 
print(adult.metadata) 
  
# variable information 
print(adult.variables) 

# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 

# %%
# Let's join our dataframe to a better analysis
df = pd.concat([X,y], axis=1)
  
# %%
# First sense of dataset
df.head()
#%%
df.describe()
# %%
print (f"{df.shape=}")

# %%
# Missing values Analysis
df = df.replace('?', np.nan)

print(f"Any NaN Value? {df.isna().any().any()}")
#%%
# 1. Detecting columns with missing values
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# 2. Calculating the percentage of missing values
missing_percentage = df.isnull().mean() * 100
print(missing_percentage[missing_percentage > 0])

# 3. Analizing pattern of missing values
#   a. Heatmap
sns.heatmap(df.isnull(), cbar=True, cmap="cividis")
if not os.path.exists('./plots/heatmap_missing_values.png'):
    plt.savefig('./plots/heatmap_missing_values.png')
plt.show()

#   b. Correlation between missing values
sns.set_theme(style="whitegrid")
plt.figure(figsize=(18, 14))
sns.heatmap(df.isnull().corr(), annot=True, cmap="cividis", fmt=".4f")
plt.title("Correlation Matrix of Missing Values")
if not os.path.exists('./plots/correlation_missing_values.png'):
    plt.savefig('./plots/correlation_missing_values.png')
plt.show()

#   c. Missing Pattern Analysis - #Ver para que me sirve!
missing_workclass_only = df[df['workclass'].isnull() & df['occupation'].notnull()]
missing_occupation_only = df[df['occupation'].isnull() & df['workclass'].notnull()]

print(missing_workclass_only)
print(missing_occupation_only)
# %%
# 4. Logic relation between correlated missing-values features
#   a. Cross-Analysis
subset = df.dropna(subset=['workclass', 'occupation'])
contingency_table = pd.crosstab(subset['workclass'], subset['occupation'])
print (contingency_table)

#   b. Countplot
plt.figure(figsize=(10, 6))
sns.countplot(data=subset, x='workclass', hue='occupation')
plt.xticks(rotation=90)
if not os.path.exists('./plots/countplot_missing_values.png'):
    plt.savefig('./plots/countplot_missing_values.png')
plt.show()

#   c. Conditional probability
workclass_given_occupation = contingency_table.div(contingency_table.sum(axis=0), axis=1)
print(workclass_given_occupation)

# %%
