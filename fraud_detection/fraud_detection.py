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
df_mean     = df["Amount"].mean()
df_median   = df["Amount"].median()
print (f"{df_mean=}")
print (f"{df_median=}")
print("Any NaN Value? {a}".format(a=df.isna().any().any()))

#Dataframe shape
print (f"{df.shape=}")

# Dataset is already categorized with fraudulent/non-fraudulent
#How many are non-fraudulent?
non_fraudulent = df['Class'].value_counts()[0]
print (f"{non_fraudulent = }")

#And how many are fraudulent?
fraudulent = df['Class'].value_counts()[1]
print (f"{fraudulent = }")

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
plt.show()
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
plt.show()
# %%
# Scaling Time and amount (non-scaled yet)

rob_scaler = RobustScaler()

df.insert(0, 'scaled_amount', rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1)))
df.insert(1, 'scaled_time',   rob_scaler.fit_transform(df['Time'].values.reshape(-1,1)))

df.drop(['Time','Amount'], axis=1, inplace=True)

print (df.columns)


# %%
# Splitting original DataFrame

X = df.drop('Class', axis=1)
y = df['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    print (i)
    print("Train:", len(train_index), "Test:", len(test_index))
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

# Converts the pandas DataFrame and Series objects to NumPy arrays. 
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

# Suppose original_ytrain is [0, 1, 0, 1, 0]:
# train_unique_label will be [0, 1] (the unique labels).
# train_counts_label will be [3, 2] (3 instances of label 0 and 2 instances of label 1).
print('-' * 100)

print('Label Distributions:')
print("For training labels (Fraud, no Fraud)" + str(train_counts_label/ len(original_ytrain)))
print("For testing labels (Fraud, no Fraud) " + str(test_counts_label / len(original_ytest )))

print (train_counts_label)

#%%
# Subsampling
# Approach: Take randomly the same proportion of non-fraud transaction to avoid wrong correlations.
# Why undersampling? Because our dataset is large enough and we can do it.
# Just taking the same amount forWith each class.

df = df.sample(frac=1)

nf = df['Class'].value_counts()[0]
f = df['Class'].value_counts()[1]
print ("Before subsampling:")
print ("Non-Fraud Length:{a}, Fraud Length:{b}".format(a=nf, b=f))

fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:len(fraud_df)]
print ("After subsampling:")
print ("Non-Fraud Length:{a}, Fraud Length:{b}".format(b=len(fraud_df), a=len(non_fraud_df)))


normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

# %%
plt.figure(figsize=(24,20))
sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20})
plt.title('SubSample Correlation Matrix', fontsize=14)
plt.savefig("/home/jrobador/GITHUB/data_science/fraud_detection/plots/corr_matrices.png")
plt.show()
# %%
print ("Features with + correlation with 'class' (Higher than 0.6:)")
print ([x for x in sub_sample_corr.columns if x != 'Class' and sub_sample_corr.loc[x, 'Class'] > 0.6])

print ("Features with - correlation with 'class' (Lower than 0.6:)")
print ([x for x in sub_sample_corr.columns if x != 'Class' and sub_sample_corr.loc[x, 'Class'] < (-0.6)])

# Print positive correlation features (Higher than 0.4)
print("Positive Correlation between features and class (Higher than 0.6):")
for feature in sub_sample_corr.columns:
    if feature != 'Class' and sub_sample_corr.loc[feature, 'Class'] > 0.6:
        print(f"{feature}: {sub_sample_corr.loc[feature, 'Class']:.2f}")

# Print negative correlation features (Lower than -0.4)
print("Negative Correlation between features and class (Lower than -0.6):")
for feature in sub_sample_corr.columns:
    if feature != 'Class' and sub_sample_corr.loc[feature, 'Class'] < -0.6:
        print(f"{feature}: {sub_sample_corr.loc[feature, 'Class']:.2f}")
# %%
