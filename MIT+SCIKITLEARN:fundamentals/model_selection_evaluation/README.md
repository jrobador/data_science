# Model Selection and Evaluation

## Cross-validation: Evaluating estimator performance

Evaluating a prediction function on the same data it was trained on is a methodological mistake. A model that simply memorizes the labels of the training samples would achieve a perfect score but would fail to predict anything useful on new, unseen data.

The simplest and often-used method to train and evaluate a model is to randomly split the data into training and test sets. However, when tuning hyperparameters, there's a risk of overfitting to the test set because parameters might be adjusted until the model performs optimally on this specific set. This results in information about the test set leaking into the model, making the evaluation metrics unreliable for assessing generalization performance. To address this, another portion of the dataset can be reserved as a "validation set." The model is trained on the training set, evaluated on the validation set, and finally tested on the test set once the model appears successful.

However, splitting the data into three sets reduces the number of samples available for training, and the results can vary depending on the specific random partition of the training and validation sets. Cross-validation is a solution to this problem.

Cross-validation is a statistical method used to evaluate the performance of a machine learning model by dividing the data into multiple subsets and assessing the model's performance on different subsets. Its primary goal is to ensure that the model generalizes well to unseen data and to reduce the risk of overfitting. While a test set should still be reserved for final evaluation, the validation set becomes unnecessary when using cross-validation. In the basic approach, known as k-fold cross-validation, the training set is divided into k smaller sets. The model is trained on k-1 of these sets and validated on the remaining set, rotating through all k combinations. This method ensures that each data point is used for both training and validation, providing a more robust evaluation of the model's performance.

![grid_search_workflow](https://scikit-learn.org/stable/_images/grid_search_workflow.png)

## Tuning hyper-parameters of an estimator


## Tuning the decision of threshold for class prediction


## Metrics and scoring: quantifying the quality of predictions


## Validation curves