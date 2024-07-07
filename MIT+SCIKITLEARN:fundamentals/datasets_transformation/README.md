# Dataset transformations

## Pipelines and composite estimators

## Feature extraction

## Preprocessing data

You should scale the data if:

- Your data are measurements of different units.
- Your columns are of completely different scales (thus obviously one will dominate the variance).
- Your data are measurements of different sensors.

You should not scale the data if:

- Your data are different dimensions of the same measurement, such as 3d points - because you WANT the (for example) x axis to dominate the variance, if all axes are of the same scale.
- Your data are measurements of the same multi-dimensional sensor, such as an image.

## Imputation of missing values

## Unsupervised dimensionality reduction

If your number of features is high, it may be useful to reduce it with an unsupervised step prior to supervised steps.

### PCA: principal component analysis


## Random projection

## Kernel Approximation

## Pairwise metrics, Affinities and kernels

## Transforming the prediction targets (y)
