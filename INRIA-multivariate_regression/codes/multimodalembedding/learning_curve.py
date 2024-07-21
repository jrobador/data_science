#LearningCurveDisplay con mean_absolute_percentage_error
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skorch.scoring import loss_scoring
import numpy as np
import seaborn as sns
import pandas as pd
from aux import mape




def lcurv(model, X_train, Y_train, X_test, Y_test, name, n_repeats):
        test_dict = {'x': X_test, 'y': Y_test}
        train_sizes = np.linspace(0.3, 0.9, 3)
        train_losses = []
        test_losses = []

        train_losses_mean = []
        train_losses_std  = []
        test_losses_mean = []
        test_losses_std  = []

        train_mape = []
        test_mape = []

        for train_size in train_sizes:
            train_mape_subset = []
            test_mape_subset = []

            for _ in range(n_repeats):
                X_train_subset, _, Y_train_subset, _ = train_test_split(X_train, Y_train, train_size=train_size, random_state=37)
                fwd_dict = {'x': X_train_subset, 'y': Y_train_subset}
                model.fit(fwd_dict, Y_train_subset)
                train_loss = model.history[-1, 'train_loss']
                #print(loss_scoring(model, test_dict, Y_test))
                test_loss = loss_scoring(model, test_dict, Y_test)

                mape_y_train, mape_y_test = mape(model, X_train_subset, Y_train_subset, X_test, Y_test)

                train_mape_subset.append(mape_y_train)
                test_mape_subset.append(mape_y_test)
                train_losses.append(train_loss)
                test_losses.append(test_loss)


            train_mape.append(np.mean(train_mape_subset))
            test_mape.append(np.mean(test_mape_subset))
            
            train_losses_mean.append(np.mean(train_losses))
            train_losses_std.append(np.std(train_losses))
            test_losses_mean.append(np.mean(test_losses))
            test_losses_std.append(np.std(test_losses))

        
        train_losses_mean = np.array(train_losses_mean)
        train_losses_std  = np.array(train_losses_std )
        test_losses_mean = np.array(test_losses_mean)
        test_losses_std  = np.array(test_losses_std )

        print (train_mape)
        print (test_mape)

        plt.figure()
        plt.title("MAPE vs Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("MAPE")
        plt.grid()

        plt.plot(train_sizes * len(X_train), train_mape, 'o-', color="b", label="Train MAPE")
        plt.plot(train_sizes * len(X_train), test_mape, 'o-', color="r", label="Test MAPE")
        plt.legend(loc="best")

        file_name = 'mape_vs_batch_size_' + name + '.png'
        plt.savefig(file_name)


        plt.figure()
        plt.title("Learning Curve")
        plt.xlabel("Number of samples in the training set")
        plt.ylabel("Loss")

        plt.grid() 

        plt.plot(train_sizes * len(X_train), train_losses_mean, 'o-', color="b", label="Train Score")
        plt.fill_between(train_sizes * len(X_train), train_losses_mean - train_losses_std, train_losses_mean + train_losses_std, alpha=0.1, color="b")

        plt.plot(train_sizes * len(X_train), test_losses_mean, 'o-', color="r", label="Validation Score")
        plt.fill_between(train_sizes * len(X_train), test_losses_mean - test_losses_std, test_losses_mean + test_losses_std, alpha=0.1, color="b")


        plt.legend(loc="best")

        file_name = 'lc_repeats=' + str(n_repeats) + '_' + name + '.png'
        plt.savefig(file_name)  


