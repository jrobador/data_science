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
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve

from collections import Counter
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE 
from imblearn.metrics import classification_report_imbalanced

from joblib import dump, load
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
print(f"Any NaN Value? {df.isna().any().any()}")

#Dataframe shape
print (f"{df.shape=}")

# Dataset is already categorized with fraudulent/non-fraudulent
#How many are non-fraudulent?
non_fraudulent = df['Class'].value_counts()[0]
print (f"{non_fraudulent=}")

#And how many are fraudulent?
fraudulent = df['Class'].value_counts()[1]
print (f"{fraudulent=}")

print('No Frauds=', round(df['Class'].value_counts()[0]/len(df), 5))
print('Frauds=', round(df['Class'].value_counts()[1]/len(df), 5))

# %%
# Some useful plots for business practices

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
# Standarization of data

df.insert(0, 'scaled_amount', RobustScaler().fit_transform(df['Amount'].values.reshape(-1,1)))
df.insert(1, 'scaled_time',   RobustScaler().fit_transform(df['Time'].values.reshape(-1,1)))

df.drop(['Time','Amount'], axis=1, inplace=True)

df.describe()

# %%
# Splitting data

x_tmp = df.drop('Class', axis=1)
y_tmp = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    x_tmp, y_tmp, test_size=0.2, random_state=37, stratify=y_tmp)       #Stratify - Mantener relacion de datos

print (f"length of X_train:{len(X_train)}, length of X_test:{len(X_test)}")

#%%
# Sub-sampling technique

# Approach: Take randomly the same proportion of non-fraud transaction to avoid wrong correlations.
# Why undersampling? Because our dataset is large enough and we can do it.
# Just taking the same amount for each class.

print ("---Subsampling Approach---")

df_train = pd.concat([X_train, y_train], axis=1)

print ("Before subsampling:")
print (f"Non-Fraud Length:{df_train['Class'].value_counts()[0]}, Fraud Length:{df_train['Class'].value_counts()[1]}")

fraud_df_train = df_train.loc[df_train['Class'] == 1]
non_fraud_df_train = df_train.loc[df_train['Class'] == 0][:len(fraud_df_train)]

print ("After subsampling:")
print (f"Non-Fraud Length:{len(non_fraud_df_train)}, Fraud Length:{len(fraud_df_train)}")

new_df_train = pd.concat([fraud_df_train, non_fraud_df_train]).sample(frac=1, random_state=37)

# %%
plt.figure(figsize=(24,20))
sub_sample_corr = new_df_train.corr(method='pearson', min_periods=1, numeric_only=False)
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
sns.boxplot(x="Class", y="V10", data=new_df_train, ax=axes[0])
axes[0].set_title('V10 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V12", data=new_df_train, ax=axes[1])
axes[1].set_title('V12 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=new_df_train, ax=axes[2])
axes[2].set_title('V14 vs Class Negative Correlation')

plt.savefig("./plots/neg_corr.png")
plt.show()

f, axes = plt.subplots(ncols=2, figsize=(10,4))

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)

sns.boxplot(x="Class", y="V4", data=new_df_train, ax=axes[0])
axes[0].set_title('V4 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V11", data=new_df_train, ax=axes[1])
axes[1].set_title('V11 vs Class Positive Correlation')

plt.savefig("./plots/pos_corr.png")
plt.show()

# %%
# Normal Distributions plots
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(30, 8))

# V14 Distribution (Fraud Transactions)
v14_fraud_dist = new_df_train['V14'].loc[new_df_train['Class'] == 1].values
sns.histplot(v14_fraud_dist, ax=ax1, color='r', kde=True, stat="density", label='KDE')
x_vals = np.linspace(min(v14_fraud_dist), max(v14_fraud_dist), 100)
ax1.plot(x_vals, norm.pdf(x_vals, np.mean(v14_fraud_dist), np.std(v14_fraud_dist)), color='orange', linestyle='--', linewidth=1.5, label='Normal Distribution Fit')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)
ax1.legend()

# V12 Distribution (Fraud Transactions)
v12_fraud_dist = new_df_train['V12'].loc[new_df_train['Class'] == 1].values
sns.histplot(v12_fraud_dist, ax=ax2, color='r', kde=True, stat="density", label='KDE')
x_vals = np.linspace(min(v12_fraud_dist), max(v12_fraud_dist), 100)
ax2.plot(x_vals, norm.pdf(x_vals, np.mean(v12_fraud_dist), np.std(v12_fraud_dist)), color='orange', linestyle='--', linewidth=1.5, label='Normal Distribution Fit')
ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)
ax2.legend()

# V10 Distribution (Fraud Transactions)
v10_fraud_dist = new_df_train['V10'].loc[new_df_train['Class'] == 1].values
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
v14_fraud = new_df_train['V14'].loc[new_df_train['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25 for V14: {} | Quartile 75 for V14: {}'.format(q25, q75))
v14_iqr = q75 - q25
print(f"{v14_iqr=}")

v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('Cut Off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('V14 outliers:{}'.format(outliers))
print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df_train = new_df_train.drop(new_df_train[(new_df_train['V14'] > v14_upper) | (new_df_train['V14'] < v14_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df_train)))
print('----' * 10)

# V12 removing outliers from fraud transactions
v12_fraud = new_df_train['V12'].loc[new_df_train['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25
print(f"{v12_iqr=}")

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print('V12 Lower: {}'.format(v12_lower))
print('V12 Upper: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
print('V12 outliers: {}'.format(outliers))
print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df_train = new_df_train.drop(new_df_train[(new_df_train['V12'] > v12_upper) | (new_df_train['V12'] < v12_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df_train)))
print('----' * 10)


# V10 removing outliers from fraud transactions
v10_fraud = new_df_train['V10'].loc[new_df_train['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25
print(f"{v10_iqr=}")

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print('V10 Lower: {}'.format(v10_lower))
print('V10 Upper: {}'.format(v10_upper))
outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
print('V10 outliers: {}'.format(outliers))
print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))

new_df_train = new_df_train.drop(new_df_train[(new_df_train['V10'] > v10_upper) | (new_df_train['V10'] < v10_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df_train)))

# %%
#Botplox without outliers
 
f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))

# Boxplots with outliers removed
# Feature V14
sns.boxplot(x="Class", y="V14", data=new_df_train,ax=ax1)
ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)

# Feature 12
sns.boxplot(x="Class", y="V12", data=new_df_train, ax=ax2)
ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)

# Feature V10
sns.boxplot(x="Class", y="V10", data=new_df_train, ax=ax3)
ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)


plt.savefig("./plots/neg_corr_nools.png")
plt.show() 
# %%
# Dimensionality Reduction and Clustering - T-SNE for 3D plot

data_embedded_TSNE = TSNE(n_components=3, random_state=37).fit_transform((new_df_train.drop('Class', axis=1)))
# %%
fig = plt.figure(facecolor="white", constrained_layout=True)
ax = fig.add_subplot(projection="3d")

y_drc = new_df_train['Class']

ax.scatter(data_embedded_TSNE[(y_drc == 0),0], data_embedded_TSNE[(y_drc == 0),1], data_embedded_TSNE[(y_drc == 0),2], c='yellow', label="Non Fraud")
ax.scatter(data_embedded_TSNE[(y_drc == 1),0], data_embedded_TSNE[(y_drc == 1),1], data_embedded_TSNE[(y_drc == 1),2], c='blue', label="Fraud")

ax.legend()

ax.set_title('t-SNE')
ax.grid(True)

plt.savefig("./plots/tsne_3d.png")
plt.show() 

# %%
# Dimensionality Reduction and Clustering - T-SNE for 2D plot

data_embedded_TSNE = TSNE(n_components=2, random_state=37).fit_transform((new_df_train.drop('Class', axis=1)))
# %%
fig = plt.figure(facecolor="white", constrained_layout=True)
ax = fig.add_subplot()

ax.scatter(data_embedded_TSNE[(y_drc == 0),0], data_embedded_TSNE[(y_drc == 0),1], c='yellow',  label="Non Fraud")
ax.scatter(data_embedded_TSNE[(y_drc == 1),0], data_embedded_TSNE[(y_drc == 1),1], c='blue',    label="Fraud")

ax.legend()

ax.set_title('t-SNE')
ax.grid(True)

plt.savefig("./plots/tsne_2d.png")
plt.show() 


#%%
# Dimensionality Reduction and Clustering - PCA with 90% of Variability

pca_module_90 = PCA(n_components=0.9, random_state=37)
data_embedded_PCA_90 = pca_module_90.fit_transform((new_df_train.drop('Class', axis=1)))

fig = plt.figure(facecolor="white", constrained_layout=True)
ax = fig.add_subplot()
plt.scatter(data_embedded_PCA_90[(y_drc == 1),0], data_embedded_PCA_90[(y_drc == 1),1],c='yellow', label="Fraud")
plt.scatter(data_embedded_PCA_90[(y_drc == 0),0], data_embedded_PCA_90[(y_drc == 0),1],c='blue', label="Non-Fraud")

plt.legend()
ax.grid(True)
plt.show() 

# %%
# Dimensionality Reduction and Clustering - PCA with 2 components

pca_module_2 = PCA(n_components=2, random_state=37)
data_embedded_PCA_2 = pca_module_2.fit_transform((new_df_train.drop('Class', axis=1)))
# %%

fig = plt.figure(facecolor="white", constrained_layout=True)
ax = fig.add_subplot()
plt.scatter(data_embedded_PCA_2[(y_drc == 1),0], data_embedded_PCA_2[(y_drc == 1),1],c='yellow', label="Fraud")
plt.scatter(data_embedded_PCA_2[(y_drc == 0),0], data_embedded_PCA_2[(y_drc == 0),1],c='blue', label="Non-Fraud")

plt.legend()
ax.grid(True)

plt.savefig("./plots/pca.png")
plt.show() 

#--------------------------------
# %%
### Classification task (is it a fraud case or not?)
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
# %%
precision_scorer = make_scorer(precision_score, average='weighted')
recall_scorer = make_scorer(recall_score, average='weighted')
f1_scorer = make_scorer(f1_score, average='weighted')
f2_scorer = make_scorer(fbeta_score, beta=2)

scoring_methods = {
    "precision": precision_scorer,
    "recall": recall_scorer,
    "f1": f1_scorer,
    "f2": f2_scorer
}
# %%
# Training classifiers

X_subsampling = new_df_train.drop('Class', axis=1)
y_subsampling = new_df_train['Class']

results_cv_sk = [[] for _ in range(len(classifiers))]

for i, clf in enumerate(classifiers):
    for scoring_name, scoring in scoring_methods.items():
        # Cross-validation
        scores = cross_val_score(clf, X_subsampling, y_subsampling,
                                 cv=StratifiedKFold(n_splits=5, random_state=37, shuffle=True), scoring=scoring)
        results_cv_sk[i].append(scores)


#%%
# Classifiers performance
avg_scores = []

for i, clf_results in enumerate(results_cv_sk):
    clf_name = classifiers[i].__class__.__name__
    print(f"Results for classifier: {clf_name}")

    avg_clf_scores = {}
    for j, score_set in enumerate(clf_results):
        sc_np = np.array(score_set) 
        mean_score = np.mean(sc_np) 

        scoring_key = list(scoring_methods.keys())[j]
        avg_clf_scores[scoring_key] = mean_score

        print(f"Scoring method {list(scoring_methods.keys())[j]}: Mean = {mean_score}")
    avg_scores.append((clf_name, avg_clf_scores))

#%%
#  Classifier comparison
best_classifier = None
best_score = -np.inf

print("\nSummary of classifier performances:")
for clf_name, scores in avg_scores:
    avg_performance = np.mean(list(scores.values())) 
    print(f"{clf_name}: Average performance across metrics = {avg_performance}")

    if avg_performance > best_score:
        best_score = avg_performance
        best_classifier = clf_name

print(f"\nBest classifier for validation set: {best_classifier} with an average score of {best_score}")

#%%
# Training and saving best classifier with joblib
best_clf = None

for clf in classifiers:
    if clf.__class__.__name__ == best_classifier:
        best_clf = clf
        break

if best_clf is None:
    raise ValueError(f"Classifier {best_classifier} not found.")

best_clf.fit(X_subsampling, y_subsampling)

model_filename = f'./models/model_LR_undersampling.joblib'
dump(best_clf, model_filename)
print(f"Model {best_clf.__class__.__name__} saved as {model_filename}")

#%%
# Final test (with previous test dataset taken from original dataframe)

best_clf = load('./models/model_LR_undersampling.joblib')

predictions = best_clf.predict(X_test)

precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
f2 = fbeta_score(y_test, predictions, average='weighted', beta=2)

print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print(f"F2 Score (weighted): {f2:.4f}")


# Business Plots - Performance before finetuning
# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No frauds", "Frauds"])
disp.plot()

plt.title(f"Confusion Matrix for {best_clf}")
plt.savefig("./plots/undersmp_cm")
plt.show()

#Precision - Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_clf.predict_proba(X_test)[:, 1])

plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall curve for {best_clf}')
plt.savefig('./plots/undersmp_pr')
plt.show()

# %% 
# Learning curves
fig, ax = plt.subplots(nrows=1, ncols=len(classifiers), sharey=True, figsize=(30,10))

for i, clf in enumerate(classifiers):
    LearningCurveDisplay.from_estimator(clf, X_subsampling, y_subsampling, 
                                        cv=StratifiedKFold(n_splits=5, random_state=37, shuffle=True), 
                                        scoring="f1_weighted", ax=ax[i], n_jobs=1)
    ax[i].set_title(f"{clf.__class__.__name__}")

fig.text(0.04, 0.5, 'Score', va='center', rotation='vertical')
fig.suptitle('Learning Curves for Different Classifiers', fontsize=16)
fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.savefig("./plots/lc.png")
plt.show()

#%%
#Hyperparameter Tuning for best classifier (The metric improve?)

def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2)

f2_scorer = make_scorer(f2_score)

LogisticRegression_model = LogisticRegression(max_iter=7000, random_state=37)

param_grid = {
    'penalty': ['l2'],
    'C': [1.0, 0.5, 0.1, 0.3, 0.7, 0.2, 0.4],
    'class_weight': ['balanced', None],
    'solver': ['lbfgs', 'liblinear', 'newton-cholesky'],
}

grid_search = GridSearchCV(estimator=LogisticRegression_model,
                           param_grid=param_grid,
                           scoring=f2_scorer,
                           cv=5,
                           n_jobs=-1,
                           verbose=1)

grid_search.fit(X_subsampling, y_subsampling)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best model: {grid_search.best_estimator_}")

model_grid_search = grid_search.best_estimator_

model_grid_search.fit(X_subsampling, y_subsampling)
predictions_grid_search = best_clf.predict(X_test)

precision = precision_score(y_test, predictions_grid_search, average='weighted')
recall = recall_score(y_test, predictions_grid_search, average='weighted')
f1 = f1_score(y_test, predictions_grid_search, average='weighted')
f2 = fbeta_score(y_test, predictions_grid_search, average='weighted', beta=2)

print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print(f"F2 Score (weighted): {f2:.4f}")

cm = confusion_matrix(y_test, predictions_grid_search)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No frauds", "Frauds"])
disp.plot()

plt.title(f"Confusion Matrix for {model_grid_search}")
plt.savefig('./plots/grid_search_cm')
plt.show()

#Precision - Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_clf.predict_proba(X_test)[:, 1])

plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall curve for {best_clf}')
plt.savefig('./plots/grid_search_pr')
plt.show()

# %%
# Under and Over-sampling technique
#%%
pipeline = Pipeline(steps=[
    ('under', RandomUnderSampler()),  
    ('over', SMOTE()),            
    ('classifier', LogisticRegression(max_iter=1000, random_state=37))
])

param_grid = {
    'under__sampling_strategy': [0.003, 0.005, 0.1],  
    'over__sampling_strategy': [0.1, 0.3, 0.5, 0.6],   
    'classifier__penalty': ['l2'],
    'classifier__C': [0.5, 0.1, 0.3, 0.2, 0.4, 1],
    'classifier__class_weight': ['balanced', None],
    'classifier__solver': ['lbfgs', 'liblinear', 'newton-cholesky'],
}

grid_search = GridSearchCV(pipeline, param_grid, scoring='recall', cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

model_filename = f'./models/model_{grid_search.__class__.__name__}.joblib'
dump(best_clf, model_filename)
print(f"Model {grid_search.__class__.__name__} saved as {model_filename}")

# %%
# Classifier trained for a good performance (85% recall, 50% precision)
pipeline_training_data = Pipeline(steps=[
    ('under', RandomUnderSampler(sampling_strategy=0.005)),  
    ('over', SMOTE(sampling_strategy=0.1))
])

X_resampled, y_resampled = pipeline_training_data.fit_resample(X_train, y_train)

print(f"Original dataset shape: {Counter(y_train)}")
print(f"Resampled dataset shape: {Counter(y_resampled)}")

best_classifier_unov = LogisticRegression(penalty='l2', C=0.1, class_weight=None, solver='liblinear', random_state=37)
best_classifier_unov.fit(X_resampled,y_resampled)
predictions_unov = best_classifier_unov.predict(X_test)

#%%
# Confusion Matrix

cm = confusion_matrix(y_test, predictions_unov)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No frauds", "Frauds"])
disp.plot()

plt.title(f"Confusion Matrix for {best_classifier_unov}")
plt.savefig("plots/unov_cm_1")
plt.show()

#Precision - Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_classifier_unov.predict_proba(X_test)[:, 1])

plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall curve for {best_classifier_unov}')
plt.savefig("plots/unov_pr_1")
plt.show()

print(classification_report_imbalanced(y_test, predictions_unov))

#%%
model_filename = f'./models/model_LR_unovsampling_1.joblib'
dump(best_classifier_unov, model_filename)
print(f"Model {best_classifier_unov.__class__.__name__} saved as {model_filename}")

# %%
#Classifier trained for the best recall (90% recall)
pipeline_training_data = Pipeline(steps=[
    ('under', RandomUnderSampler(sampling_strategy=0.1, random_state=37)),  
    ('over', SMOTE(sampling_strategy=0.6, random_state=37))
])

X_resampled, y_resampled = pipeline_training_data.fit_resample(X_train, y_train)

print(f"Original dataset shape: {Counter(y_train)}")
print(f"Resampled dataset shape: {Counter(y_resampled)}")

best_classifier_unov = LogisticRegression(penalty='l2', C=0.1, class_weight='balanced', solver='newton-cholesky', random_state=37)
best_classifier_unov.fit(X_resampled,y_resampled)
predictions_unov = best_classifier_unov.predict(X_test)

# %%
# Confusion matrix
cm = confusion_matrix(y_test, predictions_unov)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No frauds", "Frauds"])
disp.plot()

plt.title(f"Confusion Matrix for {best_classifier_unov}")
plt.savefig("plots/unov_cm_2")
plt.show()

#Precision - Recall Curve
precision, recall, _ = precision_recall_curve(y_test, best_classifier_unov.predict_proba(X_test)[:, 1])

plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.title(f'Precision-Recall curve for {best_classifier_unov}')
plt.savefig("plots/unov_pr_2")
plt.show()

print(classification_report_imbalanced(y_test, predictions_unov))

# %%
model_filename = f'./models/model_LR_unovsampling_2.joblib'
dump(best_classifier_unov, model_filename)
print(f"Model {best_classifier_unov.__class__.__name__} saved as {model_filename}")