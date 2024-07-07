# Model Selection and Evaluation

## Cross-validation: Evaluating estimator performance

Evaluating a prediction function on the same data it was trained on is a methodological mistake. A model that simply memorizes the labels of the training samples would achieve a perfect score but would fail to predict anything useful on new, unseen data.

The simplest and often-used method to train and evaluate a model is to randomly split the data into training and test sets. However, when tuning hyperparameters, there's a risk of overfitting to the test set because parameters might be adjusted until the model performs optimally on this specific set. This results in information about the test set leaking into the model, making the evaluation metrics unreliable for assessing generalization performance. To address this, another portion of the dataset can be reserved as a "validation set." The model is trained on the training set, evaluated on the validation set, and finally tested on the test set once the model appears successful.

However, splitting the data into three sets reduces the number of samples available for training, and the results can vary depending on the specific random partition of the training and validation sets. Cross-validation is a solution to this problem.

Cross-validation is a statistical method used to evaluate the performance of a machine learning model by dividing the data into multiple subsets and assessing the model's performance on different subsets. Its primary goal is to ensure that the model generalizes well to unseen data and to reduce the risk of overfitting. While a test set should still be reserved for final evaluation, the validation set becomes unnecessary when using cross-validation. In the basic approach, known as k-fold cross-validation, the training set is divided into k smaller sets. The model is trained on k-1 of these sets and validated on the remaining set, rotating through all k combinations. This method ensures that each data point is used for both training and validation, providing a more robust evaluation of the model's performance.

The workflow is the following one:
![grid_search_workflow](https://scikit-learn.org/stable/_images/grid_search_workflow.png)

### Keys of applying cross-validation

- **Data efficiency**: By splitting the data into training and test sets, a significant part of the data is reserved for testing and not used for training. This can be problematic if a limited amount of data is available. In cross-validation, especially in k-fold cross-validation, the data is divided into k subsets (folds). The model is trained k times, each time using k-1 of the subsets for training and the remaining subset for validation. This ensures that each sample of the data set is used for both training and validation, providing a more complete estimate of the model's performance.
- **Trustable evaluation**: By splitting the data into training and test sets, the evaluation of the model may depend to a large extent on how the data set is divided. Different divisions may produce different results, making the assessment less reliable and more subject to variability. Cross-validation reduces this variability because each sample in the dataset is used for both training and validation.
- **Reducing over-fitting risk**: If hyperparameters are adjusted using an additional validation set, there is still a risk of overfitting these parameters to the specific samples in the validation set. In cross-validation, the hyperparameter fit is based on multiple partitions of the dataset, which decreases the likelihood of the model overfitting to a specific partition. This ensures that the model generalises better to unseen data.

Cross-validation has the following scheme:
![grid_search_cross_validation](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)

### Different cross validation strategies

#### K-Fold

KFold divides all the samples in groups of samples, called folds (if k=n this is equivalent to the Leave One Out strategy), of equal sizes (if possible). The prediction function is learned using k-1 folds, and the fold left out is used for test.

#### Repeated K-Fold

RepeatedKFold repeats K-Fold n times. It can be used when one requires to run KFold n times, producing different splits in each repetition. It works if you want to compute standard deviation in the experiments.

#### Leave One Out (LOO)

LeaveOneOut (or LOO) is a simple cross-validation. Each learning set is created by taking all the samples except one, the test set being the sample left out. Thus, for n samples, we have n different training sets and n different tests set. This cross-validation procedure does not waste much data as only one sample is removed from the training set.

LOO cross-validation, although exhaustive, is more computationally expensive and tends to have high variance. In comparison, k-fold cross validation (with k=5 k=5 or k=10) is more efficient and provides more stable estimates of the generalization error. For these reasons, k-fold is generally the preferred choice for model selection and performance evaluation.

## Tuning hyper-parameters of an estimator

## Tuning the decision of threshold for class prediction

## Metrics and scoring: quantifying the quality of predictions

Well, there are a LOT of metrics for quantifying the quality of predictions, The sklearn.metrics module implements several loss, score, and utility functions but you can also define your custom scoring in case of need.

For more information: [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

### Regression metrics

#### Mean absolute error (MAE)

#### Mean squared error (MSE)

#### Mean absolute percentage error (MAPE)

#### Explained variance score (With R2 scoring)

### Classification metrics

Some metrics are essentially defined for binary classification tasks (e.g. f1_score, roc_auc_score). In these cases, by default only the positive label is evaluated

#### Accuracy score

#### Confusion matrix

#### Balanced accuracy score

#### Precision, recall and F-measures

## Validation curves
