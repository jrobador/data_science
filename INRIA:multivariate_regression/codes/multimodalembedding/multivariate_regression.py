from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint, TensorBoard, EpochScoring
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from networks import MultiModalEmbedding, MultiModalEmbeddingNet
from aux import real_data, process_data, validation, plot_latent, plot_loss, mape   
from learning_curve import lcurv
from downstream import r2_scoring
import sys
import os
from skorch.helper import SkorchDoctor
# from multimodalembedding.dataset_inspection.regression import regression_task
import numpy
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import matplotlib.pyplot as plt


def main():


    plots_dir = sys.argv[1]
    

    numpy.random.seed(37)
    torch.manual_seed(37)

    normalized = True
    model_print_summary = False
    lcurve = False
    diag_prob = False
    plt_latent = True
    sv_model = False

    fancy_categories = ["Age In Years", 
                        "Working Memory Task Accuracy", 
                        "Working Memory Task Reaction Time",
                        "Relational Task Accuracy", 
                        "Relational Task Reaction Time", 
                        "Gambling Task Percentage Larger", 
                        "Gambling Task RT Larger",
                        "WM test: Age-Adjusted Scale Score", 
                        "Flanker test: Age-Adjusted Scale Score", 
                        "Card Sort test: Age-Adjusted Scale Score", 
                        "Episodic Memory", 
                        "Processing Speed",
                        ]

    #Dataset loader
    X_train, Y_train, X_test, Y_test = real_data()

    #Model parameters   
    f_input_dim = X_train.shape[1]
    c_input_dim = Y_train.shape[1]
    f_layers_dim = [14753, 13006, 12264, 11775, 10875, 9729, 9303, 8852, 8000, 8000, 7642, 7175, 6949, 6810, 6311, 6000, 5823, 5583, 5245, 4990, 3629, 3000, 2732, 1200, 1950, 1655, 1500, 1414, 1032, 827, 665, 301, 135, 50]
    c_layers_dim = [12, 11, 10, 9, 8, 7]
    latent_space_dim = 6
    alpha = 1 if normalized else 0.00001

    if normalized:
        fwd_dict, test_dict, Y_train_normalized = process_data(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, normalized=normalized)
    else: fwd_dict, test_dict = process_data(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, normalized=normalized)
    if model_print_summary:  
        model_summary = MultiModalEmbedding(f_input_dim=f_input_dim, c_input_dim=c_input_dim, f_layers_dim=f_layers_dim, c_layers_dim=c_layers_dim, latent_space_dim=latent_space_dim)
        print(model_summary)
    

    norm_string = "normalized" if normalized else "unnormalized"
    mode = f"{norm_string}_fencoder_depth={len(f_layers_dim)}_LTdim={latent_space_dim}_"
    name = mode + 'multivariate_regression_' + str(latent_space_dim) 

    print(f"Model settings:")
    print(f"Alpha = {alpha}, Features encoder depth = {len(f_layers_dim)}, Cognition encoder depth = {len(c_layers_dim)}, Latent space dimension = {latent_space_dim}, Data normalized = {normalized}")
    print(f"Features encoder dimensions = {f_layers_dim}")
    print(f"Cognition encoder dimensions = {c_layers_dim}")
    print(f"Learning curve mode = {lcurve}")
    print("*"*20)



    mode_train = mode + 'train'
    mode_test = mode + 'test'
    epochs = 20 if not diag_prob else 10
    model = MultiModalEmbeddingNet (
        module = MultiModalEmbedding,
        module__f_input_dim = f_input_dim,
        module__c_input_dim = c_input_dim,
        module__f_layers_dim = f_layers_dim,
        module__c_layers_dim = c_layers_dim,
        module__latent_space_dim = latent_space_dim,
        module__alpha = alpha,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.SGD,
        optimizer__lr=0.001,
        max_epochs=epochs,
        train_split=None,
        batch_size=-1,
        callbacks = [('lr_scheduler', LRScheduler(policy=ReduceLROnPlateau,
                                                  mode='min', 
                                                  factor=0.1, 
                                                  patience=75
                                                 )),
                      ('early_stopper', EarlyStopping(monitor='train_loss',
                                                     patience=200,
                                                     threshold=0.0001
                                                     )),
        #              ('best_model_saving', Checkpoint(
        #                                         monitor='valid_loss_best',
        #                                         f_pickle="best_model_{last_epoch[epoch]}.pt",
        #                                     )),     
                        ('tensorboard', TensorBoard(writer)),
                        ('mape_scoring', EpochScoring(scoring=None, lower_is_better=True, on_train=True))
                     ],
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
    )


    if lcurve:
        n_repeats = 1
        lcurv(model, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, name=name, n_repeats=n_repeats)
    if diag_prob:
        doctor = SkorchDoctor(model)
        fwd_dict_sample = {'x': fwd_dict['x'][:100], 'y': fwd_dict['y'][:100]}
        doctor.fit(fwd_dict_sample, fwd_dict['y'][:100])

        # Activations, gradients, and parameter updates of each module are recorded
        print(model.history)
        print (doctor.module_names_)
        print (doctor.get_layer_names())
        print (doctor.get_param_names())

        #Total Loss
        fig, ax = plt.subplots(figsize=(12, 6))
        doctor.plot_loss(ax=ax, marker='o')
        fig.savefig('train_loss.png')
        plt.close()  

    else:
        if normalized:
            model.fit(fwd_dict, Y_train_normalized)
        else:
            model.fit(fwd_dict, Y_train)

        r2_scoring(model, X_train, Y_train, X_test, Y_test, fancy_categories, plots_dir, name)

        mape(model, test_dict, plots_dir)

        

        plot_loss(model, plots_dir, name)
    

        if plt_latent: 
            f_gamma_train, g_beta_train, _ = validation(model, fwd_dict)
            plot_latent(f_gamma_pred=f_gamma_train, g_beta_pred=g_beta_train, dir=plots_dir, name=mode_train)
            f_gamma_test, g_beta_test, _  = validation(model, test_dict)
            plot_latent(f_gamma_pred=f_gamma_test, g_beta_pred=g_beta_test, dir=plots_dir, name=mode_test)

        if sv_model:
            save_path = "./pickle_model/"
            os.makedirs(save_path, exist_ok=True)

            file_path = os.path.join(save_path, name)
            model.save_params(f_params=file_path+ '.pkl')



if __name__ == '__main__':
    main()



