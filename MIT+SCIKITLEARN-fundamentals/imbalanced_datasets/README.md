# Imbalanced Datasets

Tutorial followed from: [Imbalanced Learn](https://imbalanced-learn.org/)

Credits to the whole team who worked on it.

---

## Understanding imbalanced datasets

In many real-world applications, data sets are often imbalanced, meaning that the number of instances of one class significantly outnumbers the instances of other classes. This imbalance can severely affect the performance of machine learning models and lead to biased predictions.

An imbalanced data set occurs when the distribution of classes within the data is not uniform. For example, in a binary classification problem, there might be a 90:10 ratio between the majority and minority classes. Common examples of imbalanced data sets include:

- Fraud Detection: Where fraudulent transactions are much rarer than legitimate transactions.
- Medical Diagnosis: Certain diseases (e.g., rare cancers) have far fewer instances compared to common illnesses.
- Spam Detection: Spam emails are less frequent compared to non-spam emails.

The challenges posed by imbalanced datasets includes:

- Biased Model Performance: Machine learning algorithms tend to be biased towards the majority class. This is because they are often optimized to maximize overall accuracy, which leads to high accuracy for the majority class but poor performance on the minority class.
- Misleading Accuracy: In imbalanced datasets, accuracy is not a reliable metric. For instance, in a dataset with 95% of instances belonging to the majority class, a model predicting the majority class for all instances would achieve 95% accuracy, yet fail to correctly predict any instances of the minority class.
- Poor Recall and Precision for Minority Class: Models trained on imbalanced data often exhibit poor recall (sensitivity) and precision for the minority class. This means that they fail to correctly identify a significant number of minority class instances and also generate many false positives.
- Lack of Generalization: Imbalanced datasets can cause models to overfit to the majority class, resulting in poor generalization to new, unseen data, particularly for the minority class.

Several techniques can be employed to address the challenges posed by imbalanced data sets. We are going to deep dive into this models with this framework.
