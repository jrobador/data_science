#%%
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
print(f"Any NaN Value? {df.isna().any().any()}")

# 1. Detecting columns with missing values
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# 2. Calculating the percentage of missing values
missing_percentage = df.isnull().mean() * 100
print(missing_percentage[missing_percentage > 0])

# 3. Analizing pattern of missing values
#   a. Heatmap
sns.heatmap(df.isnull(), cbar=False, cmap="cividis")
plt.savefig('./plots/heatmap_missing_values.png')
plt.show()
#   b. Correlation between missing values
df.isnull().corr()
# %%
