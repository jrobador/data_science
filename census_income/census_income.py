#%%
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

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
if not os.path.exists('./census_income/plots/heatmap_missing_values.png'):
    plt.savefig('./census_income/plots/heatmap_missing_values.png')
plt.show()

#   b. Correlation between missing values
sns.set_theme(style="whitegrid")
plt.figure(figsize=(18, 14))
sns.heatmap(df.isnull().corr(), annot=True, cmap="cividis", fmt=".4f")
plt.title("Correlation Matrix of Missing Values")
if not os.path.exists('./census_income/plots/correlation_missing_values.png'):
    plt.savefig('./census_income/plots/correlation_missing_values.png')
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
if not os.path.exists('./census_income/plots/countplot_missing_values.png'):
    plt.savefig('./census_income/plots/countplot_missing_values.png')
plt.show()

#   c. Conditional probabilities
workclass_given_occupation = contingency_table.div(contingency_table.sum(axis=0), axis=1)
print(workclass_given_occupation)
print ("-"*60)
occupation_given_workclass = contingency_table.div(contingency_table.sum(axis=1), axis=0)
print(occupation_given_workclass)

# %%
# 5. Missing values imputation

# Imputing for 'occupation'
# Choosing from conditional probability after a certain threshold
threshold = 0.05
mode_workclass = df['workclass'].mode()[0]

for i, row in df.iterrows():
    if pd.isna(row['occupation']):
        workclass_value = row['workclass']
        
        if pd.isna(workclass_value):
            workclass_value = mode_workclass
            
        if workclass_value == "Never-worked":
            continue
        
        if workclass_value in occupation_given_workclass.index:
            occupation_probs = occupation_given_workclass.loc[workclass_value] #Loc for a specific row
            
            valid_occupations = occupation_probs[occupation_probs > threshold].index
            
            if len(valid_occupations) > 0:
                df.at[i, 'occupation'] = random.choice(valid_occupations)
            else: 
                raise ValueError(f"No existe un valor mayor al umbral.")
        else:
            raise ValueError(f"El valor de 'workclass' no es válido: {workclass_value}")
# %%
# Now we check if there is any NaN for occupation (excluding Never-worked)
print(df[df['workclass'] != 'Never-worked']['occupation'].isna().any())

# %%
# Imputing for 'workclass'
# Choosing from conditional probability after a certain threshold
for i, row in df.iterrows():
    if pd.isna(row['workclass']):
        occupation_value = row['occupation']

        if occupation_value in workclass_given_occupation.columns:
            workclass_probs = workclass_given_occupation[occupation_value]  #No need for loc = Looking for columns      
            
            valid_workclasses = workclass_probs[workclass_probs > threshold].index
            
            if len(valid_workclasses) > 0:
                df.at[i, 'workclass'] = random.choice(valid_workclasses)
            else: 
                raise ValueError(f"No existe un valor mayor al umbral.")
        else:
            raise ValueError(f"El valor de 'occupation' no es válido: {workclass_value}")

# %%
# Now we check if there is any NaN for workclass (excluding Never-worked)
print(df[df['workclass'] != 'Never-worked']['workclass'].isna().any())

# %%
# Finally, imputing 'native-country'.
# Random Imputing - to mantain variability of our dataframe.
categories = df['native-country'].dropna().unique()

df['native-country'].fillna(pd.Series(np.random.choice(categories, size=df['native-country'].isnull().sum())), inplace=True)
# %%
