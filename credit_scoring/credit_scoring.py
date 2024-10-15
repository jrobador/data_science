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

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, PrecisionRecallDisplay, RocCurveDisplay, make_scorer

from sklearn.model_selection._classification_threshold import TunedThresholdClassifierCV

import time
from collections import Counter

from joblib import dump

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
# Inspecting continuous data distribution
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
X = X.drop(columns=['Attribute8', 'Attribute11', 'Attribute18', 'Attribute9', 'Attribute10', 'Attribute14', 'Attribute20'])

continuous_features = [feature for feature in continuous_features if feature not in ['Attribute8', 'Attribute11', 'Attribute18']]
categorical_features = [feature for feature in categorical_features if feature not in ['Attribute9', 'Attribute10', 'Attribute14', 'Attribute20']]

X.loc[:, continuous_features] = RobustScaler().fit_transform(X[continuous_features])

encoder = OneHotEncoder(sparse_output=False) #, drop='first')  # drop='first' para evitar la multicolinealidad (one feature is a linear combination of others)
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
#%%
# Splitting data
#1 was good, 2 was bad.
y_mapped = y['class'].map({1: 0, 2: 1})

X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.1, random_state=37, stratify=y)
#%%
# Data dsitribution after splitting
tg_f = sns.countplot(data=pd.concat([X_train, y_train], axis=1), x='class', hue='class', palette='cividis', legend=False)
tg_f.set_xticklabels(['Good', 'Bad'])
total = sum([p.get_height() for p in tg_f.patches])
for p in tg_f.patches:
    height = p.get_height()
    percentage = 100 * height / total
    tg_f.annotate(f'{percentage:.0f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='bottom', fontsize=12)

plt.tight_layout()

if not os.path.exists('./credit_scoring/plots/y_odist_imbal'):
    plt.savefig('./credit_scoring/plots/y_odist_imbal')    
plt.show()
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

if not os.path.exists('./credit_scoring/plots/y_odist_bal'):
    plt.savefig('./credit_scoring/plots/y_odist_bal')
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
# Dimensionality Visualization
#t-SNE
data_embedded_TSNE = TSNE(n_components=2, random_state=37).fit_transform(X_oversampled)

#%%
fig = plt.figure(facecolor="white", constrained_layout=True)
ax = fig.add_subplot()

y_oversampled_flat = y_oversampled.values.ravel()

ax.scatter(data_embedded_TSNE[(y_oversampled_flat == 0),0], data_embedded_TSNE[(y_oversampled_flat == 0),1], c='yellow',  label="Non Fraud")
ax.scatter(data_embedded_TSNE[(y_oversampled_flat == 1),0], data_embedded_TSNE[(y_oversampled_flat == 1),1], c='blue',    label="Fraud")

ax.legend()

ax.set_title('t-SNE')
ax.grid(True)

if not os.path.exists('./credit_scoring/plots/t-sne.png'):
    plt.savefig('./credit_scoring/plots/t-sne.png')   

plt.show()
#%%
# Classifier task
XGB_model = XGBClassifier(random_state=37)

cv_strategy = StratifiedKFold(n_splits=5, random_state=37, shuffle=True)

accuracy_scores = cross_val_score(XGB_model, X_oversampled, y_oversampled,
                                   cv=cv_strategy, scoring='accuracy')

precision_scorer = make_scorer(precision_score)
precision_scores = cross_val_score(XGB_model, X_oversampled, y_oversampled,
                                    cv=cv_strategy, scoring=precision_scorer)

recall_scorer = make_scorer(recall_score)
recall_scores = cross_val_score(XGB_model, X_oversampled, y_oversampled,
                                 cv=cv_strategy, scoring=recall_scorer)

scores = {
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores
}

plt.figure(figsize=(10, 6))
bp = plt.boxplot(scores.values(), labels=scores.keys())
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.grid(axis='y')
plt.ylim(0, 1)

for i, (metric, values) in enumerate(scores.items()):
    mean_score = np.mean(values)
    plt.text(i + 1, mean_score + 0.05, f'{mean_score:.2%}', ha='center', va='bottom')

if not os.path.exists('./credit_scoring/plots/xgbperf1.png'):
    plt.savefig('./credit_scoring/plots/xgbperf1.png')
plt.show()

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
random_search.fit(X_oversampled, y_oversampled,  eval_set=[(X_test, y_test)], verbose=False)
print("CPU RandomizedSearchCV Time: %s seconds" % (str(time.time() - start)))

print(f"{random_search.best_params_=}")
print(f"{random_search.best_score_}")

#%%
# CV for Random Search best model
XGB_random_search = XGBClassifier(**random_search.best_params_, random_state=37)

score = cross_val_score(XGB_random_search, X_oversampled, y_oversampled,
                             cv=StratifiedKFold(n_splits=5, random_state=37, shuffle=True), scoring='accuracy')

print (f"Accuracy for CV: {score.mean()}")

#Model didn't improve with either GS or RS.
#%%
# Train with best results
XGB_best_params = XGBClassifier(random_state=37)
XGB_best_params.fit(X_oversampled, y_oversampled)

if not os.path.exists('./credit_scoring/models/XGB_model.joblib'):
    dump(XGB_best_params, './credit_scoring/models/XGB_model.joblib')
#%%
y_train_predict = XGB_best_params.predict(X_oversampled)
y_test_predict  = XGB_best_params.predict(X_test)

acc_train = accuracy_score(y_oversampled, y_train_predict)
pre_train = precision_score(y_oversampled, y_train_predict)

print(f"{acc_train=}, {pre_train=}")

acc_test = accuracy_score(y_test, y_test_predict) 
pre_test = precision_score(y_test,y_test_predict)

print(f"{acc_test=}, {pre_test=}")
#%%
score = cross_val_score(XGB_best_params, X_oversampled, y_oversampled,
                             cv=StratifiedKFold(n_splits=5, random_state=37, shuffle=True), scoring='accuracy')
print(score.mean())

#%%
# Plot results
# Accuracy barplot
accuracies = [score.mean(), acc_test]
labels = ['CV Accuracy', 'Test Accuracy']

plt.figure(figsize=(8, 6))
bar_plot = sns.barplot(x=labels, y=accuracies, hue=accuracies, palette="cividis", edgecolor='black', legend=False)

plt.ylim(0, 1)
plt.title('XGBoost Accuracy Comparison', fontsize=16)
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Dataset', fontsize=12)

for p in bar_plot.patches:
    bar_plot.annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='bottom', fontsize=12)
    
plt.tight_layout()
if not os.path.exists('./credit_scoring/plots/results_model_accuracy.png'):
    plt.savefig('./credit_scoring/plots/results_model_accuracy.png')    
plt.show()

#%%
#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_predict)

conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2%", cmap="cividis", cbar=False,
            xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"], linewidths=1, linecolor='black')

plt.title('Confusion Matrix (Normalized)', fontsize=16)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

accuracy_text = f'Accuracy: {acc_test:.2%}'
plt.gcf().text(0.80, 0.88, accuracy_text, fontsize=11, bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()
if not os.path.exists('./credit_scoring/plots/results_cm.png'):
    plt.savefig('./credit_scoring/plots/results_cm.png')
plt.show()

#%%
#ROC - AUC
y_test_prob = XGB_best_params.predict_proba(X_test)[:, 1]
y_pred = XGB_best_params.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='darkkhaki', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc="lower right")
if not os.path.exists('./credit_scoring/plots/results_roc.png'):
    plt.savefig('./credit_scoring/plots/results_roc.png')
plt.show()

#%%
# Matrix cost in function of business problem
cm = confusion_matrix(y_test, y_test_predict)

gain_matrix = np.array(
    [
        [0, 1],  # -1 gain for false positives
        [5, 0],  # -5 gain for false negatives
    ]
)

cm_values = np.sum(cm * gain_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(cm * gain_matrix, annot=True, cmap="cividis", cbar=False,
            xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"], linewidths=1, linecolor='black')

plt.title('Confusion Matrix for Business Scoring', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

cm_values_text = f'Cost: {cm_values}'
plt.gcf().text(0.88, 0.88, cm_values_text, fontsize=11, bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))

plt.tight_layout()
if not os.path.exists('./credit_scoring/plots/results_cm_business.png'):
    plt.savefig('./credit_scoring/plots/results_cm_business.png')
plt.show()

#%%
# Post-tuning the decision threshold for cost-sensitive learning
display = PrecisionRecallDisplay.from_estimator(
    XGB_best_params, X_test, y_test
)

plt.plot(
    recall_score(y_test, y_pred),  # Recall at the threshold of 0.5
    precision_score(y_test, y_pred),  # Precision at the threshold of 0.5
    marker="o",
    markersize=10,
    color="tab:blue",
    label="Default cut-off point at 0.5"
)

plt.title("Precision-Recall curve")
plt.legend()
if not os.path.exists('./credit_scoring/plots/results_thr_05.png'):
    plt.savefig('./credit_scoring/plots/results_thr_05.png')
plt.show()

#%%
#Tuning the cut-off point
def cg_calc(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    # The rows of the confusion matrix hold the counts of observed classes
    # while the columns hold counts of predicted classes. Recall that here we
    # consider "bad" as the positive class (second row and column).
    # Scikit-learn model selection tools expect that we follow a convention
    # that "higher" means "better", hence the following gain matrix assigns
    # negative gains (costs) to the two kinds of prediction errors:
    # - a gain of -1 for each false positive ("good" credit labeled as "bad"),
    # - a gain of -5 for each false negative ("bad" credit labeled as "good"),
    # The true positives and true negatives are assigned null gains in this
    # metric.
    #
    # Note that theoretically, given that our model is calibrated and our data
    # set representative and large enough, we do not need to tune the
    # threshold, but can safely set it to the cost ration 1/5, as stated by Eq.
    # (2) in Elkan paper [2]_.
    gain_matrix = np.array(
        [
            [0, -1],  # -1 gain for false positives
            [-5, 0],  # -5 gain for false negatives
        ]
    )
    return np.sum(cm * gain_matrix)

credit_gain_score = make_scorer(cg_calc)

tuned_model = TunedThresholdClassifierCV(
    estimator=XGB_best_params,
    scoring=credit_gain_score,
    store_cv_results=True, 
)

tuned_model.fit(X_oversampled, y_oversampled)
print(f"{tuned_model.best_threshold_=:0.2f}")
#%%
def fpr_score(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    tn, fp, _, _ = cm.ravel()
    tnr = tn / (tn + fp)
    return 1 - tnr

scoring = {
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "fpr": make_scorer(fpr_score),
    "tpr": make_scorer(recall_score),
}

def plot_roc_pr_curves(XGB_best_params, tuned_model, *, title):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))

    linestyles = ("dashed", "dotted")
    markerstyles = ("o", ">")
    colors = ("tab:blue", "tab:orange")
    names = ("XGBoost Classifier", "Tuned Threshold")
    for idx, (est, linestyle, marker, color, name) in enumerate(
        zip((XGB_best_params, tuned_model), linestyles, markerstyles, colors, names)
    ):
        decision_threshold = getattr(est, "best_threshold_", 0.5)
        PrecisionRecallDisplay.from_estimator(
            est,
            X_test,
            y_test,
            linestyle=linestyle,
            color=color,
            ax=axs[0],
            name=name,
        )
        axs[0].plot(
            scoring["recall"](est, X_test, y_test),
            scoring["precision"](est, X_test, y_test),
            marker,
            markersize=10,
            color=color,
            label=f"Cut-off point at probability of {decision_threshold:.2f}",
        )
        RocCurveDisplay.from_estimator(
            est,
            X_test,
            y_test,
            linestyle=linestyle,
            color=color,
            ax=axs[1],
            name=name,
            plot_chance_level=idx == 1,
        )
        axs[1].plot(
            scoring["fpr"](est, X_test, y_test),
            scoring["tpr"](est, X_test, y_test),
            marker,
            markersize=10,
            color=color,
            label=f"Cut-off point at probability of {decision_threshold:.2f}",
        )

    axs[0].set_title("Precision-Recall curve")
    axs[0].legend()
    axs[1].set_title("ROC curve")
    axs[1].legend()

    axs[2].plot(
        tuned_model.cv_results_["thresholds"],
        tuned_model.cv_results_["scores"],
        color="tab:orange",
    )
    axs[2].plot(
        tuned_model.best_threshold_,
        tuned_model.best_score_,
        "o",
        markersize=10,
        color="tab:orange",
        label="Optimal cut-off point for the business metric",
    )
    axs[2].legend()
    axs[2].set_xlabel("Decision threshold (probability)")
    axs[2].set_ylabel("Objective score (using cost-matrix)")
    axs[2].set_title("Objective score as a function of the decision threshold")
    fig.suptitle(title)

    plt.tight_layout()
   
    if not os.path.exists('./credit_scoring/plots/results_thr_tuned.png'):
        plt.savefig('./credit_scoring/plots/results_thr_tuned.png')

    plt.show()

plot_roc_pr_curves(XGB_best_params, tuned_model, title="Comparison of the cut-off point")
#%%
# Wrapper with manual threshold for deployment.
class CustomThresholdXGBClassifier(XGBClassifier):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold 
        super().__init__(**kwargs) 

    def predict(self, X):
        y_pred_prob = self.predict_proba(X)[:, 1]
        return (y_pred_prob > self.threshold).astype(int)

xgb_tuned_thr = CustomThresholdXGBClassifier(threshold=tuned_model.best_threshold_, n_estimators=100)
xgb_tuned_thr.fit(X_oversampled, y_oversampled)

print(f"Business defined metric: {credit_gain_score(xgb_tuned_thr, X_test, y_test)}")