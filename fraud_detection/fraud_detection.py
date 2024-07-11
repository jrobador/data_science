#%%
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.patches as mpatches
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import norm


#%%
# Have first sense of dataset

df = pd.read_csv('data/creditcard.csv')
df.head()
#%%
df.describe()
# %%
# Check dataset information

print (df["Amount"].mean())
print(df.isna().any().any())


# Dataset is already categorized with fraudulent/non-fraudulent
print (len(df))


#How many are non-fraudulent?
print (df['Class'].value_counts()[0])

#And how many are fraudulent?
print (df['Class'].value_counts()[1])

print('No Frauds:', round(df['Class'].value_counts()[0]/len(df), 5))
print('Frauds:', round(df['Class'].value_counts()[1]/len(df), 5))

# %%
# Let's check amount vs time distribution 

amount_val = df['Amount'].values
time_val = df['Time'].values

amount_mean = np.mean(amount_val)
amount_median = np.median(amount_val)
time_mean = np.mean(time_val)
time_median = np.median(time_val)

fig, ax = plt.subplots(1, 2, figsize=(18, 4))

sns.histplot(amount_val, kde=True, color='r', ax=ax[0], bins=50, edgecolor='black', linewidth=1.2, alpha=0.6)
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlabel('Transaction Amount', fontsize=12)
ax[0].set_ylabel('Frequency', fontsize=12)
ax[0].set_xlim([min(amount_val), max(amount_val)])
ax[0].set_yscale('symlog')  # Used for better visibility of large transactions
ax[0].axvline(amount_mean, color='k', linestyle='--', linewidth=1.2, label=f'Mean: ${amount_mean:,.2f}')
ax[0].axvline(amount_median, color='purple', linestyle='--', linewidth=1.2, label=f'Median: ${amount_median:,.2f}')
ax[0].legend()

sns.histplot(time_val, kde=True, color='b', ax=ax[1], bins=50, edgecolor='black', linewidth=1.2, alpha=0.6)
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlabel('Transaction Time', fontsize=12)
ax[1].set_ylabel('Frequency', fontsize=12)
ax[1].set_xlim([min(time_val), max(time_val)])
ax[1].axvline(time_mean, color='k', linestyle='--', linewidth=1.2, label=f'Mean: {time_mean:,.0f} seconds')
ax[1].axvline(time_median, color='purple', linestyle='--', linewidth=1.2, label=f'Median: {time_median:,.0f} seconds')
ax[1].legend()

plt.suptitle('Transaction Amount and Time Distributions', fontsize=16, y=1.05)

ax[0].grid(True, linestyle='--', alpha=0.7)
ax[1].grid(True, linestyle='--', alpha=0.7)

plt.savefig("/home/jrobador/GITHUB/data_science/fraud_detection/plots/distribution_plots.png")
plt.clf()
#%%
# Plot the normal distribution for 'Amount' and 'Time'
fig, ax = plt.subplots(1, 2, figsize=(18, 4))
# Amount distribution
mu_amount, std_amount = norm.fit(amount_val)
x_amount = np.linspace(min(amount_val), max(amount_val), 100)
p_amount = norm.pdf(x_amount, mu_amount, std_amount)
ax[0].plot(x_amount, p_amount, 'r-', linewidth=2)
ax[0].set_title('Normal Distribution of Transaction Amount', fontsize=14)

# Time distribution
mu_time, std_time = norm.fit(time_val)
x_time = np.linspace(min(time_val), max(time_val), 100)
p_time = norm.pdf(x_time, mu_time, std_time)
ax[1].plot(x_time, p_time, 'b-', linewidth=2)
ax[1].set_title('Normal Distribution of Transaction Time', fontsize=14)

plt.savefig("/home/jrobador/GITHUB/data_science/fraud_detection/plots/normal_distributions.png")
plt.close()

# %%
# Scaling Time and amount (non-scaled yet)

# RobustScaler is less prone to outliers.
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)

# %%
# Splitting original DataFrame

X = df.drop('Class', axis=1)
y = df['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions:')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))


# Test distribution

#%%
# Subsampling
# Approach: Take randomly the same proportion of non-fraud transaction to avoid wrong correlations.



# %%
