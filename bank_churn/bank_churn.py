#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix

import plotly.express as ex
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
#%%
df = pd.read_csv('./data/BankChurners.csv')

y = df['Attrition_Flag']

cols_to_drop = [col for col in df.columns if col.startswith('Naive_Bayes_Classifier')] + ['Attrition_Flag']
df = df.drop(columns=cols_to_drop)
# %%
df.describe()
# %%
df.isna().any().any()
# %%
df.info()
#%%
# 1. Exploratoy Data Analysis (EDA)
def analyze_feature_types(df):
    feature_types = {
        'numerical_features': {
            'int': [],
            'float': []
        },
        'categorical_features': {
            'object': [],
            'potential_categorical': []  # numerical columns with few unique values
        }
    }
    
    for column in df.columns:
        # Get dtype and number of unique values
        dtype = df[column].dtype
        n_unique = df[column].nunique()
        
        # Check if numerical column might actually be categorical
        if dtype in ['int64', 'float64'] and n_unique <= 10:  # threshold of 10 unique values
            feature_types['categorical_features']['potential_categorical'].append(
                f"{column} ({n_unique} unique values)")
            
        # Categorize based on dtype
        if dtype in ['int64', 'int32']:
            feature_types['numerical_features']['int'].append(column)
        elif dtype in ['float64', 'float32']:
            feature_types['numerical_features']['float'].append(column)
        elif dtype == 'object':
            feature_types['categorical_features']['object'].append(column)
    
    return feature_types

feature_types = analyze_feature_types(df)

print("\nDetailed Feature Analysis")
print("\nNumerical Features:")
print("Integer columns:", feature_types['numerical_features']['int'])
print("Float columns:", feature_types['numerical_features']['float'])

print("\nCategorical Features:")
print("Object columns:", feature_types['categorical_features']['object'])
print("\nPotentially Categorical (numerical with ≤10 unique values):")
print(feature_types['categorical_features']['potential_categorical'])

# %%
# Checking data distrbution
# Customer Age
fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Customer_Age'],name='Age Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Customer_Age'],name='Age Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of Customer Ages")
fig.show()

#%%
# Dependent Count (Family Size)
fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Dependent_count'],name='Dependent count Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Dependent_count'],name='Dependent count Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of Dependent counts (close family size)")
fig.show()

#%%
# Education Level proportion
ex.pie(df,names='Education_Level',title='Propotion Of Education Levels',hole=0.33)

#%%
# Marriage Status proportion
ex.pie(df,names='Marital_Status',title='Propotion Of Different Marriage Statuses',hole=0.33)

#%% 
# Income levels proportion
ex.pie(df,names='Income_Category',title='Propotion Of Different Income Levels',hole=0.33)

#%%
# Card Categories proportion
ex.pie(df,names='Card_Category',title='Propotion Of Different Card Categories',hole=0.33)

#%%
# How longer the customer is part of the bank proportion
fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Months_on_book'],name='Months on book Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Months_on_book'],name='Months on book Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of months the customer is part of the bank")
fig.show()

#%%
# No of total prod/customer 

fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Total_Relationship_Count'],name='Total no. of products Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Total_Relationship_Count'],name='Total no. of products Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of Total no. of products held by the customer")
fig.show()

#%%
# No of months inactive
fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Months_Inactive_12_mon'],name='number of months inactive Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Months_Inactive_12_mon'],name='number of months inactive Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of the number of months inactive in the last 12 months")
fig.show()

#%% 
# Distribution of credit limit
fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Credit_Limit'],name='Credit_Limit Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Credit_Limit'],name='Credit_Limit Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of the Credit Limit")
fig.show()

# Distribution of total transaction amount
fig = make_subplots(rows=2, cols=1)

tr1=go.Box(x=df['Total_Trans_Amt'],name='Total_Trans_Amt Box Plot',boxmean=True)
tr2=go.Histogram(x=df['Total_Trans_Amt'],name='Total_Trans_Amt Histogram')

fig.add_trace(tr1,row=1,col=1)
fig.add_trace(tr2,row=2,col=1)

fig.update_layout(height=700, width=1200, title_text="Distribution of the Total Transaction Amount (Last 12 months)")
fig.show()

# %%
# Check target distribution
ex.pie(y,names='Attrition_Flag',title='Proportion of churn vs not churn customers',hole=0.33)
# %%
# Continuous normalization and categorical encoding

# %%
# Train - Test - Eval splitting

# %%
# 