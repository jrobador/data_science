import pandas as pd

import cupy as cp
from cuml.neighbors import KNeighborsClassifier as cuKNeighborsClassifier
from cuml.svm import LinearSVC as cuLinearSVC
from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.naive_bayes import GaussianNB as cuGaussianNB

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

from datetime import datetime

# Load and preprocess the data
df = pd.read_csv('data/creditcard.csv')
df.insert(0, 'scaled_amount', RobustScaler().fit_transform(df['Amount'].values.reshape(-1, 1)))
df.insert(1, 'scaled_time', RobustScaler().fit_transform(df['Time'].values.reshape(-1, 1)))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

X_train = cp.asarray(X_train, dtype=cp.float32)
X_test = cp.asarray(X_test, dtype=cp.float32)

#  Evaluate the classifier. GPU(Rapids) Approach
names = [
    "Nearest Neighbors", "SVC",
    "Random Forest", "Logistic Regression",
    "GaussianNB"
]

classifiers = [
    cuKNeighborsClassifier(n_neighbors=3),
    cuLinearSVC(),
    cuRandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    cuLogisticRegression(),
    cuGaussianNB(),
]

def evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    start_time = datetime.now()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if hasattr(y_pred, 'to_numpy'):
        y_pred = y_pred.to_numpy()
    elif isinstance(y_pred, cp.ndarray):
        y_pred = y_pred.get()  # Convert CuPy array to NumPy array

    # Convert y_test to NumPy if it's a CuPy array
    if isinstance(y_test, cp.ndarray):
        y_test = y_test.get()  # Convert CuPy array to NumPy array

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    end_time = datetime.now()
    duration = end_time - start_time
    return accuracy, precision, recall, f1, duration

# Evaluate each classifier
results = []
for name, clf in zip(names, classifiers):
    print(f"Running for {name}")
    accuracy, precision, recall, f1, duration = evaluate_classifier(clf, X_train, y_train, X_test, y_test)
    results.append((name, accuracy, precision, recall, f1, duration))

# Print and save results
with open('rapids_clasf_metrics.txt', 'w') as f:
    f.write("Metrics for Rapids Classifiers\n")
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
