#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, LearningCurveDisplay

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from datetime import datetime

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced


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

plt.savefig("./plots/distribution_plots.png")
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
plt.savefig("./plots/normal_distributions.png")
plt.show()
# %%
# Scaling Time and amount
df.insert(0, 'scaled_amount', RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1)))
df.insert(1, 'scaled_time',   RobustScaler().fit_transform(df['Time'].values.reshape(-1,1)))

df.drop(['Time','Amount'], axis=1, inplace=True)

print (df.columns)

# %%
df.describe()
# %%
# Splitting original DataFrame

X = df.drop('Class', axis=1)
y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=37
)

# %%
X.min()
#%%
X.loc[274771]
#%%
# Subsampling
# Approach: Take randomly the same proportion of non-fraud transaction to avoid wrong correlations.
# Why undersampling? Because our dataset is large enough and we can do it.
# Just taking the same amount for each class.

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
new_df = normal_distributed_df.sample(frac=1, random_state=37)

# %%
plt.figure(figsize=(24,20))
sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20})
plt.title('SubSample Correlation Matrix', fontsize=14)
plt.savefig("./plots/corr_matrices.png")
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
f, axes = plt.subplots(ncols=3, figsize=(15,4))

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Class", y="V10", data=new_df, ax=axes[0])
axes[0].set_title('V10 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V12", data=new_df, ax=axes[1])
axes[1].set_title('V12 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=new_df, ax=axes[2])
axes[2].set_title('V14 vs Class Negative Correlation')

plt.savefig("./plots/neg_corr.png")
plt.show()

f, axes = plt.subplots(ncols=2, figsize=(10,4))

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)

sns.boxplot(x="Class", y="V4", data=new_df, ax=axes[0])
axes[0].set_title('V4 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V11", data=new_df, ax=axes[1])
axes[1].set_title('V11 vs Class Positive Correlation')

plt.savefig("./plots/pos_corr.png")
plt.show()

# %%
# Normal Distributions plots
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(30, 8))

# V14 Distribution (Fraud Transactions)
v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values
sns.histplot(v14_fraud_dist, ax=ax1, color='r', kde=True, stat="density", label='KDE')
x_vals = np.linspace(min(v14_fraud_dist), max(v14_fraud_dist), 100)
ax1.plot(x_vals, norm.pdf(x_vals, np.mean(v14_fraud_dist), np.std(v14_fraud_dist)), color='orange', linestyle='--', linewidth=1.5, label='Normal Distribution Fit')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)
ax1.legend()

# V12 Distribution (Fraud Transactions)
v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values
sns.histplot(v12_fraud_dist, ax=ax2, color='r', kde=True, stat="density", label='KDE')
x_vals = np.linspace(min(v12_fraud_dist), max(v12_fraud_dist), 100)
ax2.plot(x_vals, norm.pdf(x_vals, np.mean(v12_fraud_dist), np.std(v12_fraud_dist)), color='orange', linestyle='--', linewidth=1.5, label='Normal Distribution Fit')
ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)
ax2.legend()

# V10 Distribution (Fraud Transactions)
v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values
sns.histplot(v10_fraud_dist, ax=ax3, color='r', kde=True, stat="density", label='KDE')
x_vals = np.linspace(min(v10_fraud_dist), max(v10_fraud_dist), 100)
ax3.plot(x_vals, norm.pdf(x_vals, np.mean(v10_fraud_dist), np.std(v10_fraud_dist)), color='orange', linestyle='--', linewidth=1.5, label='Normal Distribution Fit')
ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)
ax3.legend()

plt.savefig("./plots/negft_distr.png")
plt.show()
# %%
# Anomaly elimination

# V14 removing outliers from fraud transactions
v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25 for V14: {} | Quartile 75 for V14: {}'.format(q25, q75))
v14_iqr = q75 - q25
print('iqr: {}'.format(v14_iqr))

v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('Cut Off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('V14 outliers:{}'.format(outliers))
print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))


print('----' * 10)

# V12 removing outliers from fraud transactions
v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print('V12 Lower: {}'.format(v12_lower))
print('V12 Upper: {}'.format(v12_upper))
outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
print('V12 outliers: {}'.format(outliers))
print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))
print('----' * 10)


# V10 removing outliers from fraud transactions
v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print('V10 Lower: {}'.format(v10_lower))
print('V10 Upper: {}'.format(v10_upper))
outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V10 outliers: {}'.format(outliers))

new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(new_df)))
# %%
new_df.head()

# %%
#Botplox without outliers
 
f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))

# Boxplots with outliers removed
# Feature V14
sns.boxplot(x="Class", y="V14", data=new_df,ax=ax1)
ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)

# Feature 12
sns.boxplot(x="Class", y="V12", data=new_df, ax=ax2)
ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)

# Feature V10
sns.boxplot(x="Class", y="V10", data=new_df, ax=ax3)
ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)


plt.savefig("./plots/neg_corr_nools.png")
plt.show() 
# %%
# T-SNE for 3D plot

data_embedded_TSNE = TSNE(n_components=3, random_state=37).fit_transform((new_df.drop('Class', axis=1)))
# %%
fig = plt.figure(facecolor="white", constrained_layout=True)
ax = fig.add_subplot(projection="3d")


sc0, sc1, sc2 = data_embedded_TSNE[:,0], data_embedded_TSNE[:,1], data_embedded_TSNE[:,2]
y = new_df['Class']

ax.scatter(data_embedded_TSNE[(y == 0),0], data_embedded_TSNE[(y == 0),1], data_embedded_TSNE[(y == 0),2], c='yellow', label="Non Fraud")
ax.scatter(data_embedded_TSNE[(y == 1),0], data_embedded_TSNE[(y == 1),1], data_embedded_TSNE[(y == 1),2], c='blue', label="Fraud")

ax.legend()

ax.set_title('t-SNE')
ax.grid(True)

plt.savefig("./plots/tsne_3d.png")
plt.show() 

# %%
# T-SNE for 2D plot

data_embedded_TSNE = TSNE(n_components=2, random_state=37).fit_transform((new_df.drop('Class', axis=1)))
# %%
fig = plt.figure(facecolor="white", constrained_layout=True)
ax = fig.add_subplot()


y = new_df['Class']

ax.scatter(data_embedded_TSNE[(y == 0),0], data_embedded_TSNE[(y == 0),1], c='yellow',  label="Non Fraud")
ax.scatter(data_embedded_TSNE[(y == 1),0], data_embedded_TSNE[(y == 1),1], c='blue',    label="Fraud")

ax.legend()

ax.set_title('t-SNE')
ax.grid(True)

plt.savefig("./plots/tsne_2d.png")
plt.show() 


#%%
pca_module_90 = PCA(n_components=0.9, random_state=37)
data_embedded_PCA = pca_module_90.fit_transform((new_df.drop('Class', axis=1)))

# %%
data_embedded_PCA.shape
pca_module_90.explained_variance_ratio_

pca_module_90.explained_variance_ratio_.sum()

# %%
pca_module_2 = PCA(n_components=2, random_state=37)
data_embedded_PCA = pca_module_2.fit_transform((new_df.drop('Class', axis=1)))
# %%
pca_module_2.explained_variance_ratio_.sum()

# %%

fig = plt.figure(facecolor="white", constrained_layout=True)
ax = fig.add_subplot()
plt.scatter(data_embedded_PCA[(y == 1),0], data_embedded_PCA[(y == 1),1],c='yellow', label="Fraud")
plt.scatter(data_embedded_PCA[(y == 0),0], data_embedded_PCA[(y == 0),1],c='blue', label="Non-Fraud")

plt.legend()
ax.grid(True)

plt.savefig("./plots/pca.png")
plt.show() 

# %%
### Defining classifiers

names = [
    "Nearest Neighbors", "SVC",
    "Random Forest", "Logistic Regression",
    "GaussianNB"
]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=37),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=37
    ),
    LogisticRegression(),
    GaussianNB()
]
clf_test = [
    KNeighborsClassifier(3),
    LogisticRegression(),
    GaussianNB()
]
#%%
# Evaluate the classifier. CPU Approach
def evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    start_time = datetime.now()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    end_time = datetime.now()
    duration = end_time - start_time
    return accuracy, precision, recall, f1, duration

#%%
# Evaluate each classifier
results = []
for name, clf in zip(names, classifiers):
    print(f"Running for {name}")
    accuracy, precision, recall, f1, duration = evaluate_classifier(clf, X_train, y_train, X_test, y_test)
    results.append((name, accuracy, precision, recall, f1, duration))

# Print and save results
with open('sk_clasf_metrics.txt', 'w') as f:
    f.write("Metrics for Sci-Kit Learn Classifiers\n")
    f.write("=" * 40 + "\n\n")
    for name, accuracy, precision, recall, f1, duration in results:
        output_str = (f"{name}:\n"
                      f"  Accuracy:  {accuracy:.4f}\n"
                      f"  Precision: {precision:.4f}\n"
                      f"  Recall:    {recall:.4f}\n"
                      f"  F1 Score:  {f1:.4f}\n"
                      f"  Duration:  {duration}\n"
                      f"{'-' * 30}\n")
        print(output_str)
        f.write(output_str)
#%%
# KFold implementation

sss = StratifiedKFold(n_splits=5, random_state=37, shuffle=True)

#%%
X = df.drop('Class', axis=1)
y = df['Class']

#%%
# Cross-Validation inspection

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


# %%
# Cross-Validation classifiers - from scratch code
results_cv = [[] for _ in range(len(classifiers))]
for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    for j, clf_cv in enumerate(classifiers):
        print (f"Classifier name {clf_cv.__class__.__name__}  CV number {i} J variable {j}")
        accuracy, precision, recall, f1, duration = evaluate_classifier(clf_cv, original_Xtrain, original_ytrain,
                                                              original_Xtest, original_ytest)
        results_cv[j].append([accuracy, precision, recall, f1, duration]) 

# %%
results_cv_np = np.array(results_cv)
# %%
## (Classifier, CV folds, Metrics)
results_cv_np.shape
# %%
results_cv_np[0,:]
# %%
# Take the mean for folds.
cv_metrics = np.mean(results_cv_np, axis=1, keepdims=True)

# %%
cv_metrics.shape
#%% 
with open('sk_clasf_CV_metrics.txt', 'w') as f:
    f.write("Metrics for Statisfied K-Fold CV - Sci-Kit Learn Classifiers\n")
    f.write("=" * 40 + "\n\n")
    for i, name in enumerate (names):
        accuracy, precision, recall, f1, duration = cv_metrics[i,0]
        output_str = (f"{name}:\n"          # Innecesario hacerlo asi...
                      f"  Accuracy:  {accuracy:.4f}\n"
                      f"  Precision: {precision:.4f}\n"
                      f"  Recall:    {recall:.4f}\n"
                      f"  F1 Score:  {f1:.4f}\n"
                      f"  Duration:  {duration}\n"
                      f"{'-' * 30}\n")
        f.write(output_str)

# %%
# Cross-Validation classifiers - built-in metric functions
precision_scorer = make_scorer(precision_score, average='weighted')
recall_scorer = make_scorer(recall_score, average='weighted')
f1_scorer = make_scorer(f1_score, average='weighted')
scoring_methods = {
    "accuracy": 'accuracy',
    "precision": precision_scorer,
    "recall": recall_scorer,
    "f1": f1_scorer
}

# %%
# Implementation
results_cv_sk = [[] for _ in range(len(classifiers))]

for clf in classifiers:
    print (clf.__class__.__name__)
    for scoring_name, scoring in scoring_methods.items():
        print(scoring_name)
        scores = cross_val_score(clf, X, y,
                                 cv=StratifiedKFold(n_splits=5, random_state=37, shuffle=True), scoring=scoring)
        results_cv_sk.append(scores)

results_cv_sk = np.array(results_cv_sk).T
print(results_cv_sk)

# %% 
# Learning curves
fig, ax = plt.subplots(nrows=1, ncols=len(clf_test), sharey=True, figsize=(30,10))

for i, clf in enumerate(clf_test):
    LearningCurveDisplay.from_estimator(clf, X, y, 
                                        cv=StratifiedKFold(n_splits=5, random_state=37, shuffle=True), 
                                        scoring=scoring_methods["accuracy"], ax=ax[i], n_jobs=1)
    ax[i].set_title(f"{clf.__class__.__name__}")

fig.text(0.04, 0.5, 'Score', va='center', rotation='vertical')
fig.suptitle('Learning Curves for Different Classifiers', fontsize=16)
fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust rect to make room for suptitle
plt.savefig("./plots/lc.png")
plt.show()

# %%
# Over-sampling technique
for clf in clf_test:
   pipeline = make_pipeline(SMOTE(sampling_strategy='auto', random_state=37, k_neighbors=5, n_jobs=-1), clf)
   pipeline.fit(X, y)

print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))

# %%
    