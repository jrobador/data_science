import pickle
import torch.utils as utils
import torch
from sklearn.manifold import TSNE, MDS
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, PredictionErrorDisplay
import os


### Data processing ###
def extract_data(cognitive_variables):
    data_real = pickle.load(
        open(
            '/data/parietal/store2/work/ggomezji/graph_dmri/data/subjects_2000_LH_non_neighbours_linear_interpolation_concatenated.pkl',
            'rb'
        )
    )
    attenuations, cognition = list(zip(*[
        (d.x.squeeze(), torch.Tensor([d[c] for c in cognitive_variables]))
        for d in data_real
    ]))
    attenuations = torch.nan_to_num(torch.stack(attenuations).to(torch.float32))
    cognition = torch.nan_to_num(torch.stack(cognition).to(torch.float32))

    return attenuations, cognition

def real_data():
    cognitive_variables = [
        'Age_in_Yrs', 
        'WM_Task_Acc', 
        'WM_Task_Median_RT',
        'Relational_Task_Acc',
        'Relational_Task_Median_RT', 
        'Gambling_Task_Perc_Larger',
        'Gambling_Task_Median_RT_Larger',
        'ListSort_AgeAdj', 
        'Flanker_AgeAdj',
        'CardSort_AgeAdj',
        'PicSeq_AgeAdj',
        'ProcSpeed_AgeAdj'
    ]
    attenuations, cognition = extract_data(cognitive_variables)
    subject_data = utils.data.TensorDataset(attenuations, cognition)
    train_set, validation_set = utils.data.random_split(
        subject_data, [.9, .1], generator=torch.Generator().manual_seed(37)
    )

    X_train = torch.stack([item[0] for item in train_set])
    Y_train = torch.stack([item[1] for item in train_set])
    X_test = torch.stack([item[0] for item in validation_set])
    Y_test = torch.stack([item[1] for item in validation_set])

    return X_train, Y_train, X_test, Y_test

def process_data(X_train, X_test, Y_train, Y_test, normalized):
    if normalized:
        X_train_normalized = StandardScaler().fit_transform(X_train).astype(np.float32)
        X_test_normalized  = StandardScaler().fit_transform(X_test).astype(np.float32)
        Y_train_normalized = StandardScaler().fit_transform(Y_train).astype(np.float32)
        Y_test_normalized  = StandardScaler().fit_transform(Y_test).astype(np.float32)

        fwd_dict = {'x': X_train_normalized, 'y': Y_train_normalized}
        test_dict = {'x':X_test_normalized, 'y': Y_test_normalized}
        return fwd_dict, test_dict, Y_train_normalized
    else:
        fwd_dict = {'x': X_train, 'y': Y_train}
        test_dict = {'x':X_test, 'y': Y_test}
        return fwd_dict, test_dict
###

### Inference function ###
def validation(model, test_dict):
    f_gamma_pred, g_beta_pred, h_eta_pred = model.forward(test_dict, training=False, device='cpu')
    np.set_printoptions(precision=8, suppress=False, linewidth=200)

    print (f"features space = {f_gamma_pred.numpy()}, shape = {f_gamma_pred.shape}")
    print (f"x_true space = {test_dict['x'].numpy()}, shape = {test_dict['x'].shape}")
    print (f"cognition space = {g_beta_pred.numpy()}, shape = {g_beta_pred.shape}")
    print (f"decoder space = {h_eta_pred.numpy()}, shape = {h_eta_pred.shape}")
    print (f"y_true space = {test_dict['y'].numpy()}, shape = {test_dict['y'].shape}")

    return f_gamma_pred, g_beta_pred, h_eta_pred 
###


### Plot functions ###
def plot_latent(f_gamma_pred, g_beta_pred, dir, name):
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

def plot_loss(model, dir, name):

    train_losses = [epoch['train_loss'] for epoch in model.history]

    epochs = range(1, len(train_losses)+1)

    # Plotting Train Loss and Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    if 'valid_loss' in model.history:
        valid_losses = [epoch['valid_loss'] for epoch in model.history]
        plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    #plt.ylim(0,10)
    plt.legend()
    file_name = f'total_loss_{name}.png'
    plt.savefig(os.path.join(dir, file_name))
    plt.close()  

    # Get the history of l2_norm and reconst_loss
    l2_norm_history = model.history[:, 'l2_norm']
    reconst_loss_history = model.history[:, 'reconst_loss']
    mape_history = model.history[:,'score']


    # Plot l2_norm loss
    if l2_norm_history:
        plt.plot(l2_norm_history, label='l2_norm loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('l2_norm Loss Over Epochs')
        plt.legend()
        file_name = f'l2_norm_loss_{name}.png'
        plt.savefig(os.path.join(dir, file_name))
        plt.close()


    # Plot reconst_loss
    if reconst_loss_history:
        plt.plot(reconst_loss_history, label='reconst_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reconstruction Loss Over Epochs')
        plt.legend()
        file_name = f'reconst_loss_{name}.png'
        plt.savefig(os.path.join(dir, file_name))
        plt.close()

    # Plot mape
    if mape_history:
        plt.plot(mape_history, label='mape')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.ylim(0, 5)
        plt.title('Mean Absolute Percentage Error Over Epochs')
        plt.legend()
        file_name = f'mape_{name}.png'
        plt.savefig(os.path.join(dir, file_name))
        plt.close() 


    # Plotting Train Loss and Validation Loss
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    if 'valid_loss' in model.history:
        valid_losses = [epoch['valid_loss'] for epoch in model.history]
        plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Get the history of l2_norm and reconst_loss
    l2_norm_history = model.history[:, 'l2_norm']
    reconst_loss_history = model.history[:, 'reconst_loss']
    mape_history = model.history[:,'score']

    # Plot l2_norm loss
    if l2_norm_history:
        plt.subplot(1, 3, 2)
        plt.plot(l2_norm_history, label='l2_norm loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('l2_norm Loss Over Epochs')
        plt.legend()

    # Plot reconst_loss
    if reconst_loss_history:
        plt.subplot(1, 3, 3)
        plt.plot(reconst_loss_history, label='reconst_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reconstruction Loss Over Epochs')
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
    