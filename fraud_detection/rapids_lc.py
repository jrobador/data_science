import matplotlib.pyplot as plt
import numpy as np

from cuml.model_selection import StratifiedKFold
from cuml.linear_model import LogisticRegression
from cuml.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import LearningCurveDisplay
import pandas as pd
from sklearn.preprocessing import RobustScaler

df = pd.read_csv('data/creditcard.csv')
df.insert(0, 'scaled_amount', RobustScaler().fit_transform(df['Amount'].values.reshape(-1, 1)))
df.insert(1, 'scaled_time', RobustScaler().fit_transform(df['Time'].values.reshape(-1, 1)))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

classifiers = [
    Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())]),
    Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier())]),
    Pipeline([('scaler', StandardScaler()), ('clf', XGBClassifier(tree_method='gpu_hist'))])
]
cv = StratifiedKFold(n_splits=5, random_state=37, shuffle=True)
scoring_methods = {"accuracy": make_scorer(accuracy_score)}

fig, ax = plt.subplots(nrows=1, ncols=len(classifiers), sharey=True, figsize=(30, 10))
for i, clf in enumerate(classifiers):
    LearningCurveDisplay.from_estimator(clf, X, y, 
                                        cv=cv, 
                                        scoring=scoring_methods["accuracy"], 
                                        ax=ax[i])
    ax[i].set_title(f"Algo")

fig.text(0.04, 0.5, 'Score', va='center', rotation='vertical')
fig.suptitle('Learning Curves for Different Classifiers (GPU)', fontsize=16)
fig.tight_layout(rect=[0.05, 0.05, 1, 0.95])

plt.show()



## It won't work because classifiers are not scikit learn like. This is an important thing to improve in Rapids! 
## To-Do: Code and modify classes to be able to run a learning_curve or code a learning_curve to rapids
## Create a pull-request