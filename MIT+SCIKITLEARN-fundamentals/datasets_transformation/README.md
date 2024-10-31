# Dataset transformations

## Pipelines and composite estimators

## Feature extraction

### Text Feature Extraction
#### The Bag of Words representation
Raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length. Most common ways to extract numerical features from text content:
- *Tokenizing* strings and giving an integer id for each possible token, for instance by using white-spaces and punctuation as token separators.
- *Counting* the occurrences of tokens in each document.
- *Normalizing* and weighting with diminishing importance tokens that occur in the majority of samples / documents.

In this scheme, features and samples are defined as follows:
- Each individual token occurrence frequency (normalized or not) is treated as a feature.
- The vector of all the token frequencies for a given document is considered a multivariate sample.

A corpus of documents can thus be represented by a matrix with one row per document and one column per token (e.g. word) occurring in the corpus.

We call vectorization the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the Bag of Words or “Bag of n-grams” representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.

#### Tf–idf term weighting
In a large text corpus, some words will be very present (e.g. “the”, “a”, “is” in English) hence carrying very little meaningful information about the actual contents of the document. If we were to feed the direct count data directly to a classifier those very frequent terms would shadow the frequencies of rarer yet more interesting terms.

In order to re-weight the count features into floating point values suitable for usage by a classifier it is very common to use the *tf–idf transform*.

Tf means **term-frequency** while tf–idf means **term-frequency times inverse document-frequency**:
$\text{tf-idf(t,d)}=\text{tf(t,d)} \times \text{idf(t)}$

Using the TfidfTransformer’s default settings, TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False) the term frequency, the number of times a term occurs in a given document, is multiplied with idf component, which is computed as
$\text{idf}(t) = \log{\frac{1 + n}{1+\text{df}(t)}} + 1$

where *n* is the total number of documents in the document set, and *df(t)* is the number of documents in the document set that contain term *t*. The resulting tf-idf vectors are then normalized by the Euclidean norm.

This was originally a term weighting scheme developed for **information retrieval**(as a ranking function for search engines results) that has also found good use in document classification and clustering.




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
