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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.manifold import TSNE

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score


import time
from collections import Counter



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

encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' para evitar la multicolinealidad (one feature is a linear combination of others)
encoded_categorical = encoder.fit_transform(X[categorical_features])

encoded_X = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))

X = X.drop(columns=categorical_features).reset_index(drop=True)
encoded_X = encoded_X.reset_index(drop=True)
X = pd.concat([X, encoded_X], axis=1)

#%%
# Target distibution
tg_dist = sns.countplot(data=df, x='class', hue='class', palette='cividis', legend=False)
tg_dist.set_xticklabels(['Good', 'Bad'])

total = sum([p.get_height() for p in tg_dist.patches])
for p in tg_dist.patches:
    height = p.get_height()
    percentage = 100 * height / total
    tg_dist.annotate(f'{percentage:.0f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='bottom', fontsize=12)
    
plt.tight_layout()

if not os.path.exists('./credit_scoring/plots/y_dist_imbal'):
    plt.savefig('./credit_scoring/plots/y_dist_imbal')
plt.show()

# Since we have an imbalanced dataset with few samples -> Oversampling
#%%
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=37, stratify=y)
#%%
# Oversampling
X_oversampled, y_oversampled = SMOTE(random_state=37).fit_resample(X_train, y_train)
df_oversampled = pd.concat([X_oversampled,y_oversampled], axis=1)
#%%
# Data distribution after oversampling
tg_dist_2 = sns.countplot(data=df_oversampled, x='class', hue='class', palette='cividis', legend=False)
tg_dist_2.set_xticklabels(['Good','Bad'])

total = sum([p.get_height() for p in tg_dist_2.patches])
for p in tg_dist_2.patches:
    height = p.get_height()
    percentage = 100 * height / total
    tg_dist_2.annotate(f'{percentage:.0f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='bottom', fontsize=12)
    
plt.tight_layout()

if not os.path.exists('./credit_scoring/plots/y_dist_bal'):
    plt.savefig('./credit_scoring/plots/y_dist_bal')
plt.show()

#%%
# Correlation matrix
correlation_matrix_l = df_oversampled.corr(method='pearson', min_periods=1, numeric_only=False)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix_l, annot=False, square=True, cmap='cividis', cbar=True)
plt.title('Correlation Matrix between Features and Target')
if not os.path.exists('./credit_scoring/plots/correlation_matrix.png'):
    plt.savefig('./credit_scoring/plots/correlation_matrix.png')
plt.show()

#%%
print("Positive Correlation between features and class (Higher than 0.2):")
pos_cor_feat = []
for feature in correlation_matrix_l.columns:
    if feature != 'class' and correlation_matrix_l.loc[feature, 'class'] > 0.2:
        print(f"{feature}: {correlation_matrix_l.loc[feature, 'class']:.2f}")
        pos_cor_feat.append(feature)

neg_cor_feat = []
print("Negative Correlation between features and class (Lower than -0.2):")
for feature in correlation_matrix_l.columns:
    if feature != 'class' and correlation_matrix_l.loc[feature, 'class'] < -0.2:
        print(f"{feature}: {correlation_matrix_l.loc[feature, 'class']:.2f}")
        neg_cor_feat.append(feature)

#%%
# Correlation inspection

sns.boxplot(x="class", y=pos_cor_feat[0], data=df_oversampled)
plt.title(f'{pos_cor_feat[0]} vs Class Positive Correlation')
plt.tight_layout()
plt.show()

f, axes = plt.subplots(ncols=len(neg_cor_feat), figsize=(30,15))

for i, feature in enumerate(neg_cor_feat):
    sns.boxplot(x="class", y=f"{feature}", data=df_oversampled, ax=axes[i])
    axes[i].set_title(f'{feature} vs Class Negative Correlation')

plt.tight_layout()
plt.show()
#%%
#%%
# Dimensionality Visualization
#t-SNE
data_embedded_TSNE = TSNE(n_components=3, random_state=37).fit_transform(X_oversampled)

#%%
fig = plt.figure(facecolor="white", constrained_layout=True)
ax = fig.add_subplot(projection='3d')

y_oversampled_flat = y_oversampled.values.ravel()

ax.scatter(data_embedded_TSNE[(y_oversampled_flat == 1),0], data_embedded_TSNE[(y_oversampled_flat == 1),1], c='yellow',  label="Non Fraud")
ax.scatter(data_embedded_TSNE[(y_oversampled_flat == 2),0], data_embedded_TSNE[(y_oversampled_flat == 2),1], c='blue',    label="Fraud")

ax.legend()

ax.set_title('t-SNE')
ax.grid(True)

plt.show()
#%%
# Classifier task
XGB_model = XGBClassifier(random_state=37)

#1 was good, 2 was bad.
y_oversampled_target = y_oversampled['class'].map({1: 0, 2: 1})

score = cross_val_score(XGB_model, X_oversampled, y_oversampled_target,
                             cv=StratifiedKFold(n_splits=5, random_state=37, shuffle=True), scoring='accuracy')

print (f"Accuracy for CV: {score.mean()}")
#%% 
# GridSearchCV
y_test_target = y_test['class'].map({1: 0, 2: 1})

param_grid = {
    'n_estimators': [100, 150, 200],      
    'max_depth': [4, 5, 6],               
    'learning_rate': [0.1, 0.2],    
    'subsample': [0.75, 0.8, 0.85],       
    'colsample_bytree': [0.7, 0.75, 0.8], 
    'gamma': [0.15, 0.2, 0.25],           
    'min_child_weight': [2, 3, 4],        
    
}

XGB_model = XGBClassifier(random_state=37)

# GridSearchCV
grid_search = GridSearchCV(estimator=XGB_model, param_grid=param_grid, 
                           scoring='accuracy', cv=3, verbose=1)

start = time.time()
grid_search.fit(X_oversampled, y_oversampled_target, eval_set=[(X_test, y_test_target)], verbose=False)
print("CPU GridSearchCV Time: %s seconds" % (str(time.time() - start)))

# Print the best parameters and best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

#%%
# RandomSearchCV

param_dist = {
    'n_estimators': np.arange(50, 300, 50),  
    'max_depth': np.arange(3, 8),
    'learning_rate': np.linspace(0.01, 0.3, 10), 
    'subsample': np.linspace(0.6, 0.9, 4).tolist() + [1.0], 
    'colsample_bytree': np.linspace(0.6, 0.9, 4).tolist() + [1.0],
    'gamma': np.linspace(0, 0.5, 5),    
    'min_child_weight': np.arange(1, 6),
    'reg_alpha': np.logspace(-3, 0, 5), 
    'reg_lambda': np.logspace(-1, 1, 5) 
}

random_search = RandomizedSearchCV(estimator=XGB_model, param_distributions=param_dist, 
                                   n_iter=50, scoring='accuracy', cv=3, verbose=1, random_state=37)

start = time.time()
random_search.fit(X_oversampled, y_oversampled_target,  eval_set=[(X_test, y_test_target)], verbose=False)
print("CPU RandomizedSearchCV Time: %s seconds" % (str(time.time() - start)))

print(f"{random_search.best_params_=}")
print(f"{random_search.best_score_}")

#%%
# Train with best results
XGB_best_params = XGBClassifier(**random_search.best_params_)
XGB_best_params.fit(X_oversampled, y_oversampled_target)

#%%
y_train_predict = XGB_best_params.predict(X_oversampled)
y_test_predict  = XGB_best_params.predict(X_test)

acc_train = accuracy_score(y_oversampled_target, y_train_predict)
pre_train = precision_score(y_oversampled_target, y_train_predict)

acc_test = accuracy_score(y_test_target, y_test_predict) 
pre_test = precision_score(y_test_target,y_test_predict)

#%%
score = cross_val_score(XGB_best_params, X_oversampled, y_oversampled_target,
                             cv=StratifiedKFold(n_splits=5, random_state=37, shuffle=True), scoring='accuracy')
print(score)
#%%
# Plot results

#%%
# Post-tuning the decision threshold for cost-sensitive learning
