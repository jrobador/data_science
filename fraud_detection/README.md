# Fraud Detection in Financial Transactions

## Main idea

Create predictive models to accurately detect whether a transaction is normal or fraudulent. The objectives include understanding the data distribution, creating a balanced sub-dataframe of fraud and non-fraud transactions, determining and evaluating various classifiers for accuracy, developing a neural network to compare its accuracy against the best classifier, and understanding common mistakes associated with imbalanced datasets.

### Challenges and its solutions

1. Imbalanced datasets are those where there is a severe skew in the class distribution, such as 1:100 or 1:1000 examples in the minority class to the majority class. This bias in the training dataset can influence many machine learning algorithms, leading some to ignore the minority class entirely. This is a problem as it is typically the minority class on which predictions are most important. One approach to addressing the problem of class imbalance is to randomly resample the training dataset. The two main approaches to randomly resampling an imbalanced dataset are to delete examples from the majority class, called **undersampling**, and to duplicate examples from the minority class, called **oversampling**. Both techniques can be used for two-class (binary) classification problems and multi-class classification problems with one or more majority or minority classes. Importantly, the change to the class distribution is only applied to the training dataset. The intent is to influence the fit of the models. The resampling is not applied to the test or holdout dataset used to evaluate the performance of a model. (See more in the repository of MIT+SCIKITLEARN:fundamentals).

