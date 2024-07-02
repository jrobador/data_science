# %%
# Imports and global variables
from typing import Optional, List, Union, Dict, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, GroupKFold
from sklearn.metrics import r2_score

from sklearn.model_selection import LearningCurveDisplay


import skorch
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint, TensorBoard, EpochScoring
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scipy.stats.mstats import pearsonr

from utils.config import PATH_SUBJECT_LIST

import sys
sys.path.append('/home/mind/jrobador/multimodalembedding')

from networks import RidgeRegression, RidgeRegressionNet, RidgeRegressionWDimRed, RidgeRegressionWDimRedNet, MultiModalEmbedding, MultiModalEmbeddingNet, NonlinearRidgeRegression, NonlinearRidgeRegressionNet, ResNet50Regression, ResNet50RegressionNet
from aux import plot_loss, plot_latent, plot_metrics, mape_per_category
from metrics import r2_test_score, r2_train_score, mape_train_score, mape_test_score, mae_train_score, mae_test_score, mape_train_mme


from sklearn.svm import NuSVR
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)


CATEGORIES_DEFAULT = [
    # === Kong categories ===
    "PicSeq_AgeAdj",
    "CardSort_AgeAdj",
    "Flanker_AgeAdj",
    "PMAT24_A_CR",
    "ReadEng_AgeAdj",
    "PicVocab_AgeAdj",
    "ProcSpeed_AgeAdj",
    "DDisc_AUC_40K",
    "VSPLOT_TC",
    "SCPT_SEN",
    "SCPT_SPEC",
    "IWRD_TOT",
    "ListSort_AgeAdj",
]

FANCY_NAME_DEFAULT = [
    # === Kong categories ===
    "Visual Episodic Memory",
    "Cognitive Flexibility (DCCS)",
    "Inhibition (Flamker Task)",
    "Fluid Intelligence (PMAT)",
    "Reading (Pronunciation)",
    "Vocabulary (Picture Matching)",
    "Processing Speed",
    "Delay Discounting",
    "Spatial Orientation",
    "Sustained Attention - Sens.",
    "Sustained Attention - Spec.",
    "Verbal Episodic Memory",
    "Working Memory (List Sorting)"
]

# %%


class ScoresPredictionTask:
    def __init__(
        self,
        path_run: str,
        path_scores: str,
        categories: Optional[List] = None,
        fancy_categories_names: Optional[List] = None,
        subjects: Optional[Union[Iterable, str]] = None,
        n_subjects: Optional[int] = None
    ) -> None:
        self.output_dir = "/home/mind/jrobador/ridge_regression_results/multimodalembedding/test_12"
        # Register the run path and create a dedicated folder inside if it does not exist
        # self.path_run = Path(path_run)
        # self.path_save = Path(path_run) / 'scores_prediction'
        # if not (self.path_save).exists():
        #     self.path_save.mkdir()
        
        # Register the categories and initialize the results
        self.categories = categories if categories is not None else CATEGORIES_DEFAULT
        if fancy_categories_names is not None:
            self.fancy_categories_names = fancy_categories_names
        else:
            self.fancy_categories_names = FANCY_NAME_DEFAULT
        self.results = None
        
        # Save the subjects ID
        if subjects is None:
            assert isinstance(n_subjects, int), 'A number of subjects (integer type) is needed'
            self.n_subjects = n_subjects
            with open(PATH_SUBJECT_LIST, "r") as file:
                subjects_list = file.read().split("\n")
            self.subjects_list = np.array(subjects_list[:self.n_subjects])
        elif isinstance(subjects, Iterable):
            self.subjects_list = np.array([str(s) for s in subjects])
        elif isinstance(subjects, str):
            self.subjects_list = np.load(subjects).astype('str')
       
        # Load scores
        scores = pd.read_csv(path_scores)
        self.scores, self.mask = self._preprocess_scores(scores)
    
    def _preprocess_scores(self, data):
        data.Subject = data.Subject.astype('str')
        
        # Discard subjects with Nan-valued scores or subjects not in the list
        mask = data[self.categories].isna().sum(axis=1) == 0
        id_subjects_ok = set(data[mask].Subject) & set(self.subjects_list)

        mask = data.Subject.isin(id_subjects_ok)
        data = data[mask][self.categories + ['Subject']]
        data = data.set_index('Subject')

        scaler = StandardScaler()
        data = pd.DataFrame(
            data=scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )

        mask = pd.Series(self.subjects_list).isin(id_subjects_ok).to_numpy()
        return data.loc[self.subjects_list[mask]], mask
    
    def _get_kfolds_rscores(
        self,
        X: np.array,
        y: np.array,
        model: BaseEstimator,
        name_model: str,
        n_pca_components: int = None,
        n_splits: int = 20,
        n_repeats: int = 50,
        n_jobs: int = -1,
        random_state=None,
    ):  
          

        n_scores = len(self.categories)    
        def inner_loop(train_idx, test_idx):
            scores_ = np.full((n_scores,), np.nan)
            mape_scores = np.full((n_scores,), np.nan)

            X_train = X[train_idx]
            X_test = X[test_idx]

            mean_X_train = np.mean(X_train)
            std_X_train = np.std(X_train)
            mean_X_test = np.mean(X_test)
            std_X_test = np.std(X_test)
            mean_y_train = np.mean(y[train_idx])
            std_y_train = np.std(y[train_idx])
            mean_y_test = np.mean(y[test_idx])
            std_y_test = np.std(y[test_idx])

            # Print mean and standard deviation
            print(f"Mean of X_train: {mean_X_train:.4f}")
            print(f"Standard deviation of X_train: {std_X_train:.4f}")

            print(f"Mean of y[train_idx]: {mean_y_train:.4f}")
            print(f"Standard deviation of y[train_idx]: {std_y_train:.4f}")

            print(f"Mean of X_test: {mean_X_test:.4f}")
            print(f"Standard deviation of X_test: {std_X_test:.4f}")

            print(f"Mean of y[test_idx]: {mean_y_test:.4f}")
            print(f"Standard deviation of y[test_idx]: {std_y_test:.4f}")
                
            if n_pca_components is not None:
                pca = PCA(n_components=n_pca_components)
                print(f"Train dimension before PCA:{X_train.shape}")
                X_train = pca.fit_transform(X_train)
                print(f"Train dimension after PCA:{X_train.shape}")
                X_test = pca.transform(X_test)



            if name_model == 'custom_ridge':
                input_dim = X_train.shape[1]                
                output_dim = y[train_idx].shape[0] if len(y[train_idx].shape) == 1 else y[train_idx].shape[1]
                print(f"{input_dim=}, {output_dim=}")
                alpha = 1



#                custom_model = RidgeRegressionNet(
#                                module=RidgeRegression,
#                                module__input_dim=input_dim,
#                                module__output_dim=output_dim,
#                                optimizer__weight_decay=alpha,
#                                criterion=nn.MSELoss,
#                                optimizer=torch.optim.AdamW,
#                                optimizer__lr=0.001,
#                                max_epochs=7500,
#                                device='cuda' if torch.cuda.is_available() else 'cpu',
#                                callbacks = [('mape_scoring_train', EpochScoring(scoring=mape_train_score, lower_is_better=False, on_train=True)),
#                                             ('mape_scoring_test',  EpochScoring(scoring=mape_test_score, lower_is_better=False, on_train=False)),
#                                             ('r2_score_train',     EpochScoring(scoring=r2_train_score, lower_is_better=True, on_train=True)),
#                                             ('r2_score_test',      EpochScoring(scoring=r2_test_score, lower_is_better=True, on_train=False)),
#                                             ('mse_score_train',    EpochScoring(scoring=mae_train_score, lower_is_better=True, on_train=True)),
#                                             ('mse_score_test',     EpochScoring(scoring=mae_test_score, lower_is_better=True, on_train=False)),
#                                             ('early_stopper',      EarlyStopping(monitor='train_loss', patience=100, threshold=0.0001)),
#                                             ('lr_scheduler',       LRScheduler(policy=ReduceLROnPlateau,mode='min', factor=0.1, patience=50))
#                                             ]  
#                             )
#                hidden_dim=[512]
#                custom_model = NonlinearRidgeRegressionNet(
#                                module=NonlinearRidgeRegression,
#                                module__input_dim=input_dim,
#                                module__hidden_dims=hidden_dim,
#                                module__output_dim=output_dim,
#                                optimizer__weight_decay=alpha,
#                                criterion=nn.MSELoss,
#                                optimizer=torch.optim.AdamW,
#                                optimizer__lr=0.01,
#                                max_epochs=7500,
#                                train_split=None,
#                                device='cuda' if torch.cuda.is_available() else 'cpu',
#                                callbacks = [('mape_scoring_train', EpochScoring(scoring=mape_train_score, lower_is_better=False, on_train=True)),
#                                             ('mape_scoring_test',  EpochScoring(scoring=mape_test_score, lower_is_better=False, on_train=False)),
#                                             ('r2_score_train',     EpochScoring(scoring=r2_train_score, lower_is_better=True, on_train=True)),
#                                             ('r2_score_test',      EpochScoring(scoring=r2_test_score, lower_is_better=True, on_train=False)),
#                                             ('mse_score_train',    EpochScoring(scoring=mae_train_score, lower_is_better=True, on_train=True)),
#                                             ('mse_score_test',     EpochScoring(scoring=mae_test_score, lower_is_better=True, on_train=False)),
#                                             ('early_stopper',      EarlyStopping(monitor='train_loss', patience=100, threshold=0.0001)),
#                                             ('lr_scheduler',       LRScheduler(policy=ReduceLROnPlateau,mode='min', factor=0.1, patience=50))
#                                             ]                              
#                            )
                f_input_dim = input_dim
                c_input_dim = output_dim
                f_layers_dim     = [1024, 1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 32, 32]
                c_layers_dim     = [1024, 512, 256, 128]
                latent_space_dim = 6
                custom_model = MultiModalEmbeddingNet (
                    module = MultiModalEmbedding,
                    module__f_input_dim = f_input_dim,
                    module__c_input_dim = c_input_dim,
                    module__f_layers_dim = f_layers_dim,
                    module__c_layers_dim = c_layers_dim,
                    module__latent_space_dim = latent_space_dim,
                    module__alpha = alpha,
                    criterion=nn.MSELoss,
                    optimizer=torch.optim.AdamW,
                    optimizer__lr=0.001,
                    max_epochs=10000,
                    train_split=None,
                    callbacks = [   ('lr_scheduler', LRScheduler(policy=ReduceLROnPlateau,
                                                              mode='min', 
                                                              factor=0.1, 
                                                              patience=30
                                                             )),
                                    ('early_stopper', EarlyStopping(monitor='train_loss',
                                                                 patience=250,
                                                                 threshold=0.0001
                                                                 )),
                                    ('tensorboard', TensorBoard(writer)),
                                 ],
                    device='cuda:0' if torch.cuda.is_available() else 'cpu',
                )
            
                print(torch.cuda.is_available())

                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_train_tensor = torch.tensor(y[train_idx], dtype=torch.float32)
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_test_tensor = torch.tensor(y[test_idx], dtype=torch.float32)

                print (f"{f_layers_dim=}")
                print (f"{c_layers_dim=}")
#
                fwd_dict = {'x':X_train_tensor,'y':y_train_tensor}
                test_dict = {'x':X_test_tensor,'y':y_test_tensor}
#
#
#                common_params = {
#                    "X": X_tensor,
#                    "y": y_tensor,
#                    "train_sizes": np.linspace(0.1, 1.0, 5),
#                    "scoring": "r2",
#                    "score_type": "both",  # training and test scores
#                    "n_jobs": -1,
#                    "line_kw": {"marker": "o"},  # Use 'o' marker for plot lines
#                    "std_display_style": "fill_between",  # Fill area for standard deviation
#                    "score_name": "Accuracy",  # Name for the score metric
#                }
#
#                fig, ax = plt.subplots(figsize=(8, 6))
#
#                LearningCurveDisplay.from_estimator(
#                    custom_model, **common_params, ax=ax
#                )
#                handles, label = ax.get_legend_handles_labels()
#                ax.legend(handles[:2], ["Training Score", "Test Score"])
#
#                # Add legend
#                ax.legend()
#
#                plt.savefig("/home/mind/jrobador/ridge_regression_results/resnet/test_5/learning_curve.png")
#                plt.close()
#                name = f"linear_ridge_regression"
#              

                custom_model.fit(fwd_dict, y_train_tensor)

                np.set_printoptions(precision=8, suppress=False, linewidth=200)

                f_gamma_pred_train, g_beta_pred_train, h_eta_pred_train, h_eta_f_gamma_train = custom_model.forward(fwd_dict, training=False, device='cpu')
                mape_train_autoencoder = mean_absolute_percentage_error(y_train_tensor,h_eta_pred_train)
                print (f"{mape_train_autoencoder=}")
                print (f"f_gamma_pred_train = {f_gamma_pred_train.numpy()}, shape = {f_gamma_pred_train.shape}")
                print (f"g_beta_pred_train = {g_beta_pred_train.numpy()}, shape = {g_beta_pred_train.shape}")
                print (f"decoder_train = {h_eta_pred_train.numpy()}, shape = {h_eta_pred_train.shape}")
                print (f"y_true_train = {fwd_dict['y'].numpy()}, shape = {fwd_dict['y'].shape}")

                f_gamma_pred_test, g_beta_pred_test, h_eta_pred_test, h_eta_f_gamma_test = custom_model.forward(test_dict, training=False, device='cpu')
                mape_test_autoencoder = mean_absolute_percentage_error(y_test_tensor,h_eta_pred_test)
                print (f"{mape_test_autoencoder=}")
                print (f"g_beta_pred_test = {g_beta_pred_test.numpy()}, shape = {g_beta_pred_test.shape}")
                print (f"decoder_orig_test = {h_eta_pred_test.numpy()}, shape = {h_eta_pred_test.shape}")
                print (f"decoder_fromx_test = {h_eta_f_gamma_test.numpy()}, shape = {h_eta_f_gamma_test.shape}")
                print (f"y_true_test = {test_dict['y'].numpy()}, shape = {test_dict['y'].shape}")
                #plot_latent(f_gamma_pred, g_beta_pred, self.output_dir)


                y_pred = h_eta_f_gamma_test
                #y_pred = custom_model.predict(X_test_tensor) 


                # Plotting the losses
                plot_loss(custom_model, self.output_dir, "losses", combined=True)
                
            else:        
                model.fit(X_train, y[train_idx])
                y_pred = model.predict(X_test)

            for c in range(n_scores):
                r, _ = pearsonr(y_pred[:, c], y[test_idx][:, c])
                scores_[c] = r

            for c in range(n_scores):
                mape = mean_absolute_percentage_error(y[test_idx][:, c], y_pred[:, c])
                mape_scores[c] = mape

            return scores_, mape_scores
        

        if len(set(self.subjects_list)) == len(self.subjects_list):
            kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
            scores_list = Parallel(n_jobs=n_jobs)(
                delayed(inner_loop)(train_idx, test_idx)
                for train_idx, test_idx in tqdm(kf.split(X))
            )
            scores, mape_scores = zip(*scores_list)
        else:
            print("Several samples per subject: processing group k-folds...")
            scores = []
            
            def get_random_labels(labels, random_state):
                labels_shuffled = np.unique(labels)
                # shuffle works in place
                random_state.shuffle(labels_shuffled)
                new_labels_mapping = {k: i for i, k in enumerate(labels_shuffled)}
                new_labels = np.array([new_labels_mapping[label] for label in labels])
                return new_labels

            if random_state is None:
                random_state = np.random.RandomState(42)
            elif isinstance(random_state, int):
                random_state = np.random.RandomState(random_state)
                
            for _ in tqdm(range(n_repeats)):
                group_kf = GroupKFold(n_splits=n_splits)
                new_labels = get_random_labels(self.subjects_list[self.mask], random_state)
                temp = Parallel(n_jobs=n_jobs)(
                    delayed(inner_loop)(train_idx, test_idx)
                    for train_idx, test_idx in group_kf.split(X, groups=new_labels)
                )
                scores += temp
            
        return scores, mape_scores
    
    def plot_scores_prediction(
        self,
        y: pd.DataFrame,
        title: str = '',
        name_file: str = ''
    ):
        mean = y.to_numpy().mean()
        std = y.to_numpy().mean(axis=1).std()

        fig, ax = plt.subplots()
        sns.boxplot(
            data=y, width=0.3, color='skyblue', ax=ax, showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": "5"
            }
        )
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='grey', linestyle='dashed')
        # plt.ylim(-0.5, 0.5)
        plt.ylabel('r-Pearson correlation')
        plt.title(title + f"\n Mean score: {mean:.4f} +/- {std:.4f}")
        plt.tight_layout()

        plt.savefig(os.path.join(self.output_dir, name_file + '.png'))
        plt.close()
    
    def plot_mape_prediction(self,df_pred):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
                data=df_pred, width=0.3, color='skyblue', ax=ax, showmeans=True,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markersize": "5"
                }
            )
        mean_mape = df_pred.to_numpy().mean()
        std = df_pred.to_numpy().mean(axis=1).std()
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='grey', linestyle='dashed')
        plt.ylabel('MAPE')
        plt.title(f"MAPE Scores by Category\nMean score: {mean_mape:.4f} +/- {std:.4f}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "MAPE.png"))
        plt.close()

    def predict(
        self,
        X: np.array,
        models: Dict[str, BaseEstimator],
        name_X: str = '',
        n_pca_components: int = None,
        n_splits: int = 20,
        n_repeats: int = 50,
        n_jobs: int = -1,
        random_state=None,
        plot: bool = True,
    ):
        y = self.scores.to_numpy()
        X = np.array([x.cpu().numpy() for x in X])[self.mask]

        for name_model, model in models.items():
            print(f'Processing regression with model: {name_model} and features: {name_X}')
            predicted_scores, mape_scores = self._get_kfolds_rscores(
                X, y, model, name_model, n_pca_components, n_splits, n_repeats, n_jobs, random_state
            )
            df_pred = pd.DataFrame(predicted_scores, columns=self.fancy_categories_names)
            df_pred = df_pred.reset_index()
            df_pred['index'] = df_pred['index'] // n_splits
            df_pred = df_pred.groupby('index').mean()
            
            print("Average score:", df_pred.to_numpy().mean())
            print("Std score:", df_pred.to_numpy().mean(axis=1).std())
            print("-----------------")
            

            if plot:
                title = f"Model {name_model} on rv {name_X}"
                name_file = f'{name_model}_{name_X}'
                self.plot_scores_prediction(df_pred, title, name_file)
            
            print (f"{mape_scores=}")
            df_pred_mape = pd.DataFrame(mape_scores, columns=self.fancy_categories_names)
            df_pred_mape = df_pred_mape.reset_index()
            df_pred_mape['index'] = df_pred_mape['index'] // n_splits
            df_pred_mape = df_pred_mape.groupby('index').mean()
            self.plot_mape_prediction(df_pred_mape)

            df_pred['model'] = name_model
            df_pred['rv'] = name_X
            if self.results is None:
                self.results = df_pred
            else:
                self.results = pd.concat((self.results, df_pred), axis=0)
