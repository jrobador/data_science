import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import os

def r2_scoring(model, X_train, Y_train, X_test, Y_test, categories, dir, name):

    print (f"X Train Shape: {X_train.shape}")
    print (f"Y Train Shape: {Y_train.shape}")
    

    y_predicted_from_x_train = model.module_.predict_y_from_x(X_train).cpu().numpy()
    y_predicted_from_x_test = model.module_.predict_y_from_x(X_test).cpu().numpy()

    print(f"y_predicted_from_x_train shape: {y_predicted_from_x_train.shape}")

    scores_ = np.full((len(categories),), np.nan)

    for c in range(len(categories)):
        r2 = r2_score(Y_train[:, c], y_predicted_from_x_train[:, c])
        scores_[c] = r2


    print (f"Scores={scores_}")

    df_pred = pd.DataFrame(scores_, index=categories, columns=['R2 Score'])

    fig, ax = plt.subplots()
    sns.boxplot(
        data=df_pred.T, width=0.3, color='skyblue', ax=ax, showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "5"
        }
    )
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='grey', linestyle='dashed')
    plt.ylabel('R2 Score for train features')
    plt.tight_layout()

    file_name = 'r2_' + name + '.png'
    plt.savefig(os.path.join(dir, file_name))
    plt.close()
