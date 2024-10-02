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

## Missing values analysis

The procedure is as follows:

1. **Detecting columns with missing values:** Using EDA (Exploratory Data Analysis) functions to detect what is the quantity of missing values.
2. **Calculate percentage of missing values:** It is useful because we can decide if we would remove rows, columns or impute the number with some technique.
3.  **Analize pattern of missing values:** Are they randomly missing or there is any kind of pattern that we can leverage it? For this point, we are going to use the following strategies:
    -  **Heatmap of missing values:** 
    ![HM MV](./plots/heatmap_missing_values.png)
    
    In the visualization, we can see that there are some columns with missing values, which are highlighted in a different color (yellow) in contrast to the others. In particular, the columns that appear to have missing values ​​include:

        - *workclass*
        - *occupation*
        - *native-country*

        The rest of the columns do not appear to have missing values. This suggests that the missing data problem is concentrated in a limited number of features, which will allow us to apply specific strategies only on those columns.
    - **Correlation between missing values:** 
    ![Correlation MV](./plots/correlation_missing_values.png)
    From the figure, we can detect some key observations:
        - **Strong correlation between workclass and occupation:** There is a very high correlation between missing values ​​in the workclass and occupation columns (0.9984), indicating that when a value is missing in one, it is also likely to be missing in the other. This suggests that the missing values ​​in both columns might be related, perhaps because the absence of information on workclass also implies that information on occupation is missing.
        - **Slight correlation between workclass, occupation, and native-country:** The workclass and occupation columns also have a slight correlation with the missing values ​​in the native-country column (0.0268), indicating that, although weak, the missing values ​​in native-country could be related to the missing information in the other two columns. 
        - **Without correlation between other columns.**
    - **Missing Pattern Analysis**:

### Is there any logic relation among these features?

We want to find if there is any hidden relation between *workclass* and *ocupation* features to infere the values to column in function of the other one. On the other hand, *native-country* shows a small correlation, so we can handle it as a independent feature.

For this reason, we are going to explore different techniques:
- **Relation between both columns (cross-analysis):** We are going to see how the *workclass* and *occupation* columns are related in the rows where there are no missing values. (During this analysis, I realized that the dataframe contains '?' values, so I needed to change them to NaN).
- **Relation visualization (countplot):** See if there are some plots that allows us to visualize clear patterns. 

![CountPlot](./plots/countplot_missing_values.png)

- **Conditional probability:** We are going to explore the conditional probability of a column given another column. For example, the probability that a person has a certain value in occupation given that we already know their workclass. This can help us decide how to fill in missing values ​​in one column when the other is present. Here, we present the probability of a person to have a certain workclass, given his occupation:

| workclass        | Adm-clerical | Armed-Forces | Craft-repair | Exec-managerial | Farming-fishing | Handlers-cleaners | Machine-op-inspct | Other-service | Priv-house-serv | Prof-specialty | Protective-serv | Sales   | Tech-support | Transport-moving |
|------------------|--------------|--------------|--------------|-----------------|-----------------|-------------------|-------------------|---------------|-----------------|----------------|-----------------|---------|--------------|------------------|
| Federal-gov      | 0.086794     | 1.0          | 0.015216     | 0.044035        | 0.006040        | 0.017375          | 0.006287          | 0.011172      | 0.0             | 0.040992       | 0.047813        | 0.003089 | 0.066390     | 0.015711          |
| Local-gov        | 0.075031     | 0.0          | 0.034522     | 0.054387        | 0.028859        | 0.031371          | 0.007942          | 0.060938      | 0.0             | 0.171905       | 0.457782        | 0.002907 | 0.040111     | 0.066242          |
| Private          | 0.749955     | 0.0          | 0.776832     | 0.656425        | 0.449664        | 0.928089          | 0.953673          | 0.824091      | 1.0             | 0.552333       | 0.304171        | 0.806504 | 0.798064     | 0.798301          |
| Self-emp-inc     | 0.008376     | 0.0          | 0.027323     | 0.101380        | 0.055034        | 0.002896          | 0.005625          | 0.008531      | 0.0             | 0.039695       | 0.005086        | 0.076308 | 0.006224     | 0.016136          |
| Self-emp-not-inc | 0.012475     | 0.0          | 0.130563     | 0.096451        | 0.438255        | 0.010135          | 0.019523          | 0.056063      | 0.0             | 0.093163       | 0.007121        | 0.107376 | 0.029046     | 0.077707          |
| State-gov        | 0.066833     | 0.0          | 0.015380     | 0.047157        | 0.016779        | 0.009170          | 0.006287          | 0.038797      | 0.0             | 0.101912       | 0.178026        | 0.003634 | 0.060166     | 0.025478          |
| Without-pay      | 0.000535     | 0.0          | 0.000164     | 0.000164        | 0.005369        | 0.000965          | 0.000662          | 0.000406      | 0.0             | 0.000000       | 0.000000        | 0.000000 | 0.000000     | 0.000000          |



**We can infere the following results for missing values anaylisis:**
- **Unbalanced distribution:** As seen in the graph, most occupations belong to the *Private* class. This suggests that the *Private* class is dominant, and there is a strong relationship between occupation and job class.
- **Conditional probabilities:** In the *Private* class, occupations such as *Craft-repair*, *Handlers-cleaners*, and *Machine-op-inspct* have a high probability (over 90%) of belonging to that class. Similarly, some occupations are exclusive to certain job classes, such as *Armed-Forces* in *Federal-gov*, or with *Priv-house-serv*, working as a *Private*

### Imputation based on the relationship between columns

