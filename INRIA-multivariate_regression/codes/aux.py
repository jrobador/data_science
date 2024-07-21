import torch
from sklearn.manifold import TSNE, MDS
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, PredictionErrorDisplay
import os


### Plot functions ###
def plot_latent(f_gamma_pred, g_beta_pred, dir, name='rep_lat'):
    f_gamma_pred_np = f_gamma_pred.numpy()
    g_beta_pred_np = g_beta_pred.numpy()
    latent_representation = np.vstack((f_gamma_pred_np, g_beta_pred_np))
    tsne = MDS(n_components=2, random_state=0, normalized_stress='auto')
    latent_2d = tsne.fit_transform(latent_representation)

    num_samples = f_gamma_pred.shape[0]
    labels = np.array(['Features'] * num_samples + ['Cognition'] * num_samples)

    plot_data = pd.DataFrame({
        'dimension 1': latent_2d[:, 0],
        'dimension 2': latent_2d[:, 1],
        'Data Type': labels
    })

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=plot_data,
        x='dimension 1', y='dimension 2',
        hue='Data Type',
        palette=sns.color_palette("hsv", 2),
        legend="full"
    )

    plt.title('Latent Representations')
    plt.legend(title='Data Type')

    file_name = 'latent_space_plot_' + name + '.png'
    plt.savefig(os.path.join(dir, file_name))
    plt.close() 

def plot_loss(model, dir, name, combined=False):

    train_losses = model.history[:, 'train_loss']
    # valid_loss = model.history[:, 'valid_loss']

    epochs = range(1, len(train_losses)+1)

    # Plotting Train Loss and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    # if 'valid_loss' in model.history:
    #     valid_losses = [epoch['valid_loss'] for epoch in model.history]
    # plt.plot(epochs, valid_loss, label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    file_name = f'total_loss_{name}.png'
    plt.savefig(os.path.join(dir, file_name))
    plt.close()  

    # Get the history of l2_norm, reconst_loss and mape
    l2_norm_history = model.history[:, 'l2_norm'] if 'l2_norm' in model.history[0] else None 
    reconst_loss_history = model.history[:, 'reconst_loss'] if 'reconst_loss' in model.history[0] else None
    y_true_loss_history = model.history[:, 'y_true_loss'] if 'y_true_loss' in model.history[0] else None
    y_from_x_loss_history = model.history[:, 'y_from_x_loss'] if 'y_from_x_loss' in model.history[0] else None
    mape_history_test = model.history[:,'score'] if 'score' in model.history[0] else None
    mape_history_train = model.history[:, 'neg_mean_absolute_percentage_error'] if 'neg_mean_absolute_percentage_error' in model.history[0] else None

    # Plot l2_norm loss
    if l2_norm_history is not None:
        plt.plot(l2_norm_history, label='l2_norm loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('l2_norm Loss Over Epochs')
        plt.legend()
        file_name = f'l2_norm_loss_{name}.png'
        plt.savefig(os.path.join(dir, file_name))
        plt.close()
    # Plot reconst_loss
    if reconst_loss_history is not None:
        plt.plot(reconst_loss_history, label='reconst_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reconstruction Loss Over Epochs')
        plt.legend()
        file_name = f'reconst_loss_{name}.png'
        plt.savefig(os.path.join(dir, file_name))
        plt.close()
    if y_true_loss_history is not None:
        plt.plot(y_true_loss_history, label='y_true_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Y autoencoder Loss Over Epochs')
        plt.legend()
        file_name = f'y_true_loss_{name}.png'
        plt.savefig(os.path.join(dir, file_name))
        plt.close()
    if y_from_x_loss_history is not None:
        plt.plot(y_from_x_loss_history, label='y_from_x_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Y from X Loss Over Epochs')
        plt.legend()
        file_name = f'y_from_x_loss_{name}.png'
        plt.savefig(os.path.join(dir, file_name))
        plt.close()    
    # Plot mape test
    if mape_history_test is not None:
        plt.plot(mape_history_test, label='mape')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.ylim(-5,0)
        plt.title('Neg MAPE Test Over Epochs')
        plt.legend()
        file_name = f'mape_test_{name}.png'
        plt.savefig(os.path.join(dir, file_name))
        plt.close() 
    # Plot mape train
    if mape_history_train is not None:
        plt.plot(mape_history_train, label='mape')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.ylim(-5, 0)
        plt.title('Neg MAPE Train Over Epochs')
        plt.legend()
        file_name = f'mape_train_{name}.png'
        plt.savefig(os.path.join(dir, file_name))
        plt.close()         



    # Plotting Combined Losses
    if combined:
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        if 'valid_loss' in model.history:
            valid_losses = [epoch['valid_loss'] for epoch in model.history]
            plt.plot(epochs, valid_losses, label='Validation Loss')
        plt.title('Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot l2_norm loss
        if l2_norm_history:
            plt.subplot(1, 4, 2)
            plt.plot(l2_norm_history, label='l2_norm loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('l2_norm Loss Over Epochs')
            plt.legend()

        # Plot reconst_loss
        if reconst_loss_history:
            plt.subplot(1, 4, 3)
            plt.plot(reconst_loss_history, label='reconst_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Reconstruction Loss Over Epochs')
            plt.legend()
        
        if y_true_loss_history:
            plt.subplot(1, 4, 2)
            plt.plot(y_true_loss_history, label='y_true_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Y autoencoder Loss Over Epochs')
            plt.legend()

        if y_from_x_loss_history:
            plt.subplot(1, 4, 4)
            plt.plot(y_from_x_loss_history, label='y_from_x_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Y from X Loss Over Epochs')
            plt.legend()

        plt.tight_layout()

        # Save the combined plot
        file_name = f'combined_plots_{name}.png'
        plt.savefig(os.path.join(dir, file_name))
        plt.close()

###
def mape(model, test_dict, plots_dir):
    #X_train = fwd_dict['x']
    X_test  = test_dict['x']
    #Y_train = fwd_dict['y']
    Y_test  = test_dict['y']

    #y_predicted_from_x_train = model.module_.predict_y_from_x(X_train).cpu().numpy()
    y_predicted_from_x_test = model.module_.predict_y_from_x(X_test).cpu().numpy()
    #mape_y_train = mean_absolute_percentage_error(y_predicted_from_x_train, Y_train)
    mape_y_test  = mean_absolute_percentage_error(Y_test, y_predicted_from_x_test)

    print (f"{mape_y_test=}")


    display = PredictionErrorDisplay(y_true=Y_test, y_pred=y_predicted_from_x_test)
    display.plot()
    plt.savefig(os.path.join(plots_dir,'PED.png'))
    plt.close()
    

def plot_metrics(net,output_dir):
    #mae_test_sc     = net.history[:, 'mae_test_score']
    mae_train_sc    = net.history[:, 'mae_train_score']
    #mape_test_sc    = net.history[:,'mape_test_score']
    mape_train_sc   = net.history[:,'mape_train_score']
    #r2_test_sc      = net.history[:,'r2_test_score'] 
    #r2_train_sc     = net.history[:,'r2_train_score']
    train_losses    = net.history[:,'train_loss']    
    #valid_losses    = net.history[:,'valid_loss']    

    epochs = range(1, len(mae_train_sc)+1)

    metrics = {
        # 'mae_test_score': mae_test_sc,
        'mae_train_score': mae_train_sc,
        # 'mape_test_score': mape_test_sc,
        'mape_train_score': mape_train_sc,
        # 'r2_test_score': r2_test_sc,
        # 'r2_train_score': r2_train_sc,
        'train_loss': train_losses,
        #'valid_loss': valid_losses
    }

    os.makedirs(output_dir, exist_ok=True)

    for metric_name, metric_values in metrics.items():
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, metric_values, label=f'{metric_name}')
        plt.title(f'{metric_name} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.legend()
        file_name = f'{metric_name}.png'
        plt.savefig(os.path.join(output_dir, file_name))
        plt.close()


    scores = {
        # "mae_test_score":   min(mae_test_sc  ),
        "mae_train_score":  min(mae_train_sc ),
        # "mape_test_score":  max(mape_test_sc ),
        "mape_train_score": max(mape_train_sc),
        # "r2_test_score":    max(r2_test_sc   ),
        # "r2_train_score":   max(r2_train_sc  ),
        "train_loss":       min(train_losses ),
        #"valid_loss":       min(valid_losses )
    }


    best_scores_file = "model_best_scores.txt"

    with open(os.path.join(output_dir, best_scores_file), 'w') as file:
        for key, value in scores.items():
            file.write(f"{key}: {value}\n")




def mape_per_category(model, x_true, y_true, n_scores, fancy_categories_names, output_dir, for_train, y_predicted=None):
    if for_train:
        file_name = "mape_per_category_train.png"
        title = "Train"
    else:
        file_name = "mape_per_category_test.png"
        title = "Test"

    if y_predicted is None:
        y_pred = model.predict(x_true)
    else: y_pred = y_predicted

    mape_scores = np.zeros(n_scores)
    for c in range(n_scores):
        mape = mean_absolute_percentage_error(y_true[:, c], y_pred[:, c])
        mape_scores[c] = mape

    print (f"{mape_scores.shape=}")
    print (f"{mape_scores=}")

    df_pred = pd.DataFrame({
            'MAPE': mape_scores,
            'Category': fancy_categories_names
        })

        # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
            x='Category', y='MAPE', data=df_pred, width=0.3, color='skyblue', ax=ax, showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": "5"
            }
        )

    mean_mape = df_pred['MAPE'].mean()

    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='grey', linestyle='dashed')
    plt.ylabel('MAPE')
    plt.title(f"MAPE {title} Scores by Category\nMean score: {mean_mape:.4f}")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()


    # y_true = y_true.numpy() 
    # display = PredictionErrorDisplay(y_true=y_true, y_pred=y_pred)
    # display.plot()
    # plt.savefig(os.path.join(output_dir,'PED.png'))
    # plt.close()



def pca_explained_variance(pca, output_dir):
    explained_variance_ratio = pca.explained_variance_ratio_
    # Print the explained variance ratio
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"Principal Component {i + 1}: {ratio:.4f} ({ratio * 100:.2f}%)")
    # Plot the explained variance
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Explained variance ratio by principal components')
    plt.savefig(os.path.join(output_dir, "explained_variance.png"))
    plt.close()
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    # Print cumulative explained variance
    for i, cum_var in enumerate(cumulative_explained_variance):
        print(f"Principal Components 1 to {i + 1}: {cum_var:.4f} ({cum_var * 100:.2f}%)")
    # Plot cumulative explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance by Principal Components')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "explained_variance_accumulative.png"))
    plt.close()