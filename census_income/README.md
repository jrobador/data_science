# Census Income

[Dataset Link](https://archive.ics.uci.edu/dataset/2/adult)

Prediction task is to determine whether a person's income is over $50,000 a year.

## Dataset information
- 48842 subjects, 14 multivariate categories: Categorical and Integer. Target is defined as '<=50K' and '>50K'.
Categories are:
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

- Has missing values! Let's start with this pre-processing step.

# Missing values analysis

The procedure is as follows:

1. Detecting columns with missing values: Using EDA (Exploratory Data Analysis) functions to detect what is the quantity of missing values.
2. Calculate percentage of missing values: It is useful because we can decide if we would remove rows, columns or impute the number with some technique.
3.  Analize pattern of missing values: Are they randomly missing or there is any kind of pattern that we can leverage it? For this point, we are going to use the following strategies:
    -  Heatmap of missing values: In the visualization, we can see that     there are some columns with missing values, which are highlighted in a different color (yellow) in contrast to the others. In particular, the columns that appear to have missing values ​​include:
        - workclass
        - occupation
        - native-country
    The rest of the columns do not appear to have missing values. This suggests that the missing data problem is concentrated in a limited number of features, which will allow us to apply specific strategies only on those columns.

    - Correlation between missing values: We can detect some key observations:
        - Strong correlation between workclass and occupation: There is a very high correlation between missing values ​​in the workclass and occupation columns (0.9984), indicating that when a value is missing in one, it is also likely to be missing in the other. This suggests that the missing values ​​in both columns might be related, perhaps because the absence of information on workclass also implies that information on occupation is missing.
        - Slight correlation between workclass, occupation, and native-country: The workclass and occupation columns also have a slight correlation with the missing values ​​in the native-country column (0.0268), indicating that, although weak, the missing values ​​in native-country could be related to the missing information in the other two columns. 
        - Without correlation between other columns.
    



