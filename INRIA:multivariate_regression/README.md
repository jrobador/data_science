# Multivariate Regression

This repository contains the results of my work at INRIA on the following topics:

- Understanding and implementation of the paper "Improving Individual-Specific Functional Parcellation Through Transfer Learning."
- Mathematical formulation of loss functions and definition of various neural network architectures based on the problem to be solved.
- Implementation of linear regression models: Ridge Regression and Support Vector Regression (SVR).
- Implementation of nonlinear regression models: Various kernels with synthetic data.
- Enhancement of neural network training processes: Application of callbacks such as LR Scheduler, Early Stopping, experiment control with Tensorboard, and Best Model saver.
- Implementation of nonlinear regression models for features from the paper: MLP, Multimodal Embeddings, Resnet50 (custom implementation in PyTorch with skorch wrapper).
- Study and implementation of spherical harmonics.
- Application and understanding of metrics during training and testing: R2, MAPE, MSE, MAE. Latent space representation graph in MME.
- Environment tools applied:

    - **Sci-kit Learn**:
        - Model Selection with RepeatedKFold, GroupKFold.
        - Preprocessing with StandardScaler.
        - Decomposition with PCA.
        - Metrics like R2 Score, MAPE.
        - Models like Ridge and KernelRidge.
        - Manifold with T-SNE and MDS.
    - **Sci-py**: Correlation pearsonr.  
    - **Torch**: For NN Architectures
    - **Numpy**: For preparation, processing and handling data.
    - **Seaborn**: For metric plots.
    - **Pyplot**: For learning curves, scoring and loss plots.
    - **Pandas**: For CSV files and DataFrame handling.
    - **Joblib**: For Parallel computation.
- Computational tools applied: SSH connection, GPU with CUDA, SLURM, and GIT.
