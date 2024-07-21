# Regression Ridge Model

## Mathematical formulations
### Optimization Function:
\begin{equation}
\gamma^* = argmin_\gamma||y-f_\gamma(g_{\beta^*}(x))||_2^2+\alpha||\gamma||_2
\end{equation}
\begin{cases}
g: \mathbb{R}^{L \times 1010004} &\rightarrow \mathbb{R}^{100} \ \
f: \mathbb{R}^{100} &\rightarrow \mathbb{R}^{13}
\end{cases}
Let's break this equation in parts:
**Data Reconstruction Error:** $||y - f_\gamma(g_\beta(x))||_2^2$: This term measures the squared L2 distance between the true labels $y$ and the reconstructed labels obtained by applying the function $f_\gamma$ to the transformed data $g_{\beta^*}(x)$. Minimizing this term encourages $f_\gamma$ and $g_{\beta^*}$ to learn a good representation that captures the essential information for predicting the labels.
**L2 Regularization:** $α||γ||_2^2$: This term penalizes the L2 norm of the parameters γ in function $f_\gamma$. A higher α value increases the penalty for complex models, promoting simpler solutions that might generalize better. This is also called Ridge regression.

In the other hand, we have:
\begin{equation}
\beta^*= argmin [\mathcal{L}_{PCA}(g_{\beta}(x))]
\end{equation}




This equation seeks the optimal value of the hyperparameter $\beta$, denoted by $\beta^*$, which minimizes a loss function related to Principal Component Analysis (PCA) applied to the data transformed by $G_\beta(x)$. Notably, unlike $\gamma^*$ (which might depend on the target labels $y$), $\beta^*$ is independent of $y$. In simpler terms, the function $g$ doesn't consider the labels $y$ (target values) when transforming the data. Its main goal is to reduce the data's dimensionality while keeping as much of the  variance in the information as possible.

Our new goal is to find a new optimization function, that minimizes two terms with respect to parameters α and β:
\begin{equation}
\mathcal{L}= argmin_{\gamma, \beta}||y-f_\gamma(g_{\beta}(x))||_2^2+\alpha||\gamma||_2
\end{equation}
Here, $g_\beta(x)$ represents the output of a function $g$, which is parameterized by $\beta$, and it's adjusted to minimize the disparity between its output and the labels $y$. To achieve this, we've replaced the PCA step with a linear layer that transforms the input space from 1010004 dimensions to 100 dimensions. Although PCA is inherently a linear operation, it's important to note that its dimensionality reduction process involves rotations.

In this new function, we don't put any constraints on it. We are using a solution space larger than the PCA but it is learned based on the objective. Compared to PCA, this approach allow us for a broader range of possible transformations. This flexibility lets the model potentially learn more complex relationships in the data.

### Orthogonal regularization:
Orthogonal regularization is used to keep the weights of a layer as orthogonal as possible. The mathematical expression is:

\begin{equation}
\lambda = \sum_{i,j} \left( \left( W W^T \right)_{ij} - I_{ij} \right)^2
\end{equation}

where:
* $W$ are the weights of the first layer.
* $WW^T$ is the product of the weights and their transpose.
* $I$ is the identity matrix.
* $(WW^T)_{ij}$ is the element in position (i, j) of the product $WW^T$.
* $I_{ij}$ is the element in position (i, j) of the identity matrix.


This formulation penalizes the deviation of $WW^T$ from the identity matrix, encouraging $W$ to be an orthogonal matrix.

## Contrastive Loss for Dimensionality Reduction

The core idea:

1. We have a set of high-dimensional data points.
2. We define a method to identify "similar" data points for each point. Data points belonging to the same class are considered positive pairs, representing inherent similarity.
3. We use a contrastive loss function during training that penalizes the model when similar points are mapped far apart in the lower-dimensional space and vice versa.

This training process encourages the model to learn a mapping that captures the underlying structure of the data, where similar points are close together even in the lower-dimensional representation.






## Results

### BaseLine
Model with PCA and Ridge from Sci-kit Learning:
![](https://notes.inria.fr/uploads/upload_3944ff3d84af2163b27c01aacbec69cf.png)

### PCA + Custom Ridge Regression
Model with PCA from Sci-kit Learning and custom ridge regression:
![](https://notes.inria.fr/uploads/upload_0bed8085c22de8ab9628ef7d35253a52.png)


### Orthogonal regularization + Custom Ridge Regression
Model with custom DimRed and ridge regression: 

#### First test
##### Parameters
- splits = 10
- repeats = 1
- lr = 0.0001
- epochs = 150
- alpha = 1
- hidden_layers_dim = 1000
![](https://notes.inria.fr/uploads/upload_1cd3ce936b094bab2fb3de9c5fd97b4a.png)


It is important to note that the fitting of this model produced a significantly high loss of values:
![](https://notes.inria.fr/uploads/upload_fb9c912cad79d67e38e689985a90a98e.png)
#### Second test
##### Parameters
- splits = 5
- repeats = 1
- lr = 0.0001
- epochs = 800
- alpha = 1

![](https://notes.inria.fr/uploads/upload_ed6b5fd750fd2378540343ebc300a8cd.png)

![](https://notes.inria.fr/uploads/upload_c657b856b83871a30df5fd2ac1fb00a6.png)
The losses in epoch 164 are train_loss= 2.8724,  valid_loss= 8.3362.
The losses in epoch 718 are train_loss= 16.9247, valid_loss= 3.4614.

#### Third test
Reducing the number of epochs:
##### Parameters
- splits = 10
- repeats = 1
- lr = 0.0001
- epochs = 200
- alpha = 1
![](https://notes.inria.fr/uploads/upload_b2a97005243dc543965da2afa8ae3008.png)

![](https://notes.inria.fr/uploads/upload_9f75933475d6f96684dee0b28dc1be65.png)
The losses in epoch 173 are train_loss =2.6487, valid_loss=4.8250.


#### Fourth test
What if we reduce the number of splits and keep a lower number of epochs?
##### Parameters
- splits = 5
- repeats = 1
- lr = 0.0001
- epochs = 200
- alpha = 1

![](https://notes.inria.fr/uploads/upload_c4e8d8bc057ccf48f0ad513f8ead53e2.png)

![](https://notes.inria.fr/uploads/upload_4912314dbfcfd64c16c726ad2ca31e03.png)
The losses in epoch 155 are train_loss= 4.0171, valid_loss=4.5598.


#### Fifth test
Adding more repeat steps:
##### Parameters
- splits = 5
- repeats = 3
- lr = 0.0001
- epochs = 200
- alpha = 1


![](https://notes.inria.fr/uploads/upload_2565224327251cc3b02a246808b67a40.png)
![](https://notes.inria.fr/uploads/upload_19af340c3cdd1e91b33d3a16b17b4097.png)
The losses in epoch 127 are train_loss= 4.7057, valid_loss= 13.2694.

#### Sixth test
Reducing hidden layers, adding more epochs and splits
##### Parameters
- splits = 10
- repeats = 1
- lr = 0.0001
- epochs = 1000
- alpha = 1
- hidden_layers = 100
![](https://notes.inria.fr/uploads/upload_385fd3ff3ae15a1f4217af5183ec58a5.png)

![](https://notes.inria.fr/uploads/upload_9498a9841b52909bb2ba5cdc35161194.png)

The losses in epoch 970 are train_loss= 1.9851, valid_loss= 3.0833.

#### Seventh test
Reducing hidden layers

##### Parameters
- splits = 10
- repeats = 1
- lr = 0.0001
- epochs = 1000
- alpha = 1
- hidden_layers = 50

![](https://notes.inria.fr/uploads/upload_846f826779b8df03a0a3c8b5a828ba23.png)

![](https://notes.inria.fr/uploads/upload_48947619e5849dfc24890cedcf3dd64b.png)

#### Eighth test
We use a LR scheduler with Early Stopping to improve learning times.
##### Parameters
- splits = 10
- repeats = 5
- lr = 0.0001
- epochs = 1000
- alpha = 1
- hidden_layers = 1500

![](https://notes.inria.fr/uploads/upload_fac6cb36753a3e1a36ef20cf14f64277.png)

![](https://notes.inria.fr/uploads/upload_ed73bd8f0c42c12648fc6e03b30906ac.png)



# Multivariate Regression with multi-modal embedding.

$X$ are the features $\theta$ and $Y$ are those from cognition.

Goal, find to parametric functions $f_\gamma:\mathcal X\mapsto \mathcal M$, $g_\beta:\mathcal Y \mapsto \mathcal M$, and $h_\eta: \mathcal M \mapsto \mathcal Y$ such our goal is

$$
\arg\min \mathcal L(\gamma, \beta, \eta) = \sum_i d(f_\gamma(X_i), g_\beta(Y_i)) + \alpha\|h_\eta(g_\beta(Y_i))-Y_i\|^2_2 
$$
for each subject $i$, where $d$ can be the $L_2$ norm. This means that $h_\eta = g^{-1}_\beta$


Metrica de evaluacion
$$
1/n \sum_i |h_\eta(f_\gamma(X_i))-Y_i|/|(Y_i)+\epsilon|
$$
Para train_set, test_set y features cognitivas.
Correlacion entre h(f(x)) e y para cada sujeto, para cada variable = varianza explicada

Para saber si nuestra prediccion es buena. -> downstream task = predicir la cognicion en base de X.

Gran plot -> MAPE y varianza

## 2nd moment idea

If $S_1, \ldots, S_N$ are our subjects, the previous loss is
$$
\mathcal L(S_1, \ldots. S_N)=\sum_{S_i} f(S_i)
$$
then
$$
\partial \mathcal L/ (\partial S_i\partial S_j) =0 \leftrightarrow i \neq j
$$
a contrastive loss is of the form
$$
\mathcal L(S_1, \ldots. S_N)=\sum_{S_i, S_j} f(S_i, S_j)
$$
then
$$
\partial \mathcal L/ (\partial S_i\partial S_j) =\sum_{S_k, S_l} \partial f/(\partial S_i\partial S_j)(S_k, S_l) = \frac{\partial f}{(\partial S_i\partial S_j)}(S_i, S_j)
$$





## Results with PAVI + PCA

### Test 1


#### Model settings:
- Alpha = 1
- Features encoder depth = 17
- Cognition encoder depth = 6
- Latent space dimension = 6
- Features encoder dimensions = [180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20]
- Cognition encoder dimensions = [12, 11, 10, 9, 8, 7]
- n_pca_components=195
- n_splits=10
- n_repeats=1
![](https://notes.inria.fr/uploads/upload_83344ac1ba2a018e1cc560e01bfb41b2.png)

![](https://notes.inria.fr/uploads/upload_127fbfd3f1d9678c82adbeac6d12ee84.png)

![](https://notes.inria.fr/uploads/upload_80502f2621542a3dea07cdb259b2a685.png)
![](https://notes.inria.fr/uploads/upload_70ae55536fc2b3de5d976525b0e4ce4f.png)
![](https://notes.inria.fr/uploads/upload_8b3174575dd0b8ae3c130e8b712149b6.png)
![](https://notes.inria.fr/uploads/upload_40e66bb748d3687a5d8e46abd531d6bc.png)
![](https://notes.inria.fr/uploads/upload_2d4bfdca75c0bd98178335ee7d3e1f7c.png)
![](https://notes.inria.fr/uploads/upload_37a485178154bdc41f345aabff91d75b.png)

(regression_2446557_44)
### Test 2
#### Model settings:

- Alpha = 1
- Features encoder depth = 17
- Cognition encoder depth = 6
-  Latent space dimension = 4
-  Features encoder dimensions = [180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20]
-  Cognition encoder dimensions = [12, 11, 10, 9, 8, 7]
-  n_pca_components=195
-   n_splits=10
-   n_repeats=1





![](https://notes.inria.fr/uploads/upload_efc018b27900b3a2f6d4591407db1c29.png)
![](https://notes.inria.fr/uploads/upload_2dbd5e4c677fd607dbaf58103681b6f6.png)
![](https://notes.inria.fr/uploads/upload_0510001efe4f1091f81112457247295e.png)
![](https://notes.inria.fr/uploads/upload_f446a8b445449414e3a67d5445cd6279.png)
![](https://notes.inria.fr/uploads/upload_045447ed9ccb079d721d38d703c4fd1d.png)
![](https://notes.inria.fr/uploads/upload_9a5b806f8b356b43b5ab8045570c0acb.png)
![](https://notes.inria.fr/uploads/upload_8ad69952261ed2f754fa2732a0dde249.png)
![](https://notes.inria.fr/uploads/upload_e1ebc7d9328659158d0d9bb8f959b38c.png)
(regression_2446559_44)

### Test 3



#### Model settings:
- Alpha = 1
-  Features encoder depth = 17
-  Cognition encoder depth = 6
-  Latent space dimension = 6
-  Features encoder dimensions = [180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20]
-  Cognition encoder dimensions = [12, 11, 10, 9, 8, 7]
-  n_pca_components=195
-   n_splits=10
-   n_repeats=5

![](https://notes.inria.fr/uploads/upload_df9d5ce5e2c4483abd413c86a48143a0.png)
![](https://notes.inria.fr/uploads/upload_4258f70042ef9733af652b26f48b3d3a.png)
![](https://notes.inria.fr/uploads/upload_2cf4f5a209f0e095b58ab0a75452db37.png)
![](https://notes.inria.fr/uploads/upload_b24811a487e69990e3faa83ad6c90428.png)
![](https://notes.inria.fr/uploads/upload_1355e88acd1f00379495231d1eba34e6.png)
![](https://notes.inria.fr/uploads/upload_9653e947a17bc38caaac1613870e943e.png)
![](https://notes.inria.fr/uploads/upload_83690702c732d26ceec5557c10a5054c.png)
![](https://notes.inria.fr/uploads/upload_9a44459da38b9de35d0bc4f6d0170a73.png)
(regression_2446560_44)

### Test 4



#### Model settings:
- Alpha = 1
- Features encoder depth = 9
-  Cognition encoder depth = 6
-   Latent space dimension = 6
-   Features encoder dimensions = [100, 90, 80, 70, 60, 50, 40, 30, 20]
-   Cognition encoder dimensions = [12, 11, 10, 9, 8, 7]
-   n_pca_components=100
-   n_splits=10
-   n_repeats=5


![](https://notes.inria.fr/uploads/upload_82a10c82a822d8bb0651d9a7b87d18d3.png)
![](https://notes.inria.fr/uploads/upload_75a080807850568b56c2e747f18abbf0.png)
![](https://notes.inria.fr/uploads/upload_40c44be7f598ab69e9ce0acc7cd5c2b0.png)
![](https://notes.inria.fr/uploads/upload_dfc371494039d1b2f83bf1d9bfb3b28a.png)
![](https://notes.inria.fr/uploads/upload_c197bf746e3578c1659c77139984451a.png)
![](https://notes.inria.fr/uploads/upload_4f7ee13498f6753d021a8bda8e9e99c1.png)
![](https://notes.inria.fr/uploads/upload_94b30779df04933f1535d3cf64df12ca.png)
![](https://notes.inria.fr/uploads/upload_3a626048c46ab66fc8ed5906cca6d5da.png)
(regression_2446571_44)

### Test 5 



#### Model settings:
- Alpha = 0.1
- Features encoder depth = 18
-  Cognition encoder depth = 6
-   Latent space dimension = 6
-   Features encoder dimensions = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10]
-   Cognition encoder dimensions = [12, 11, 10, 9, 8, 7]
-   n_pca_components=100
-   n_splits=107
-   n_repeats=5

![](https://notes.inria.fr/uploads/upload_840978d4c79b878db6fdbcd363123e80.png)
![](https://notes.inria.fr/uploads/upload_532672ea6247478eb292b6bc30f773b0.png)
![](https://notes.inria.fr/uploads/upload_f8c9a96d3e7e6f6b3695bfe6007f0b60.png)
![](https://notes.inria.fr/uploads/upload_be247ca984eff652885c9105be329128.png)
![](https://notes.inria.fr/uploads/upload_5c2862949e37c415dedfb8b7f4e392d5.png)
![](https://notes.inria.fr/uploads/upload_c14d30f2c1032db6b4aea5944f287f21.png)
![](https://notes.inria.fr/uploads/upload_808c0ad86a8d2f1fb290cf026f894c22.png)
![](https://notes.inria.fr/uploads/upload_4d396dfd80fc9ebec36049d22a19cee4.png)
(regression_2446680_44)

### Test 6 

#### Model Settings:
- Alpha = 0.1
- n_pca_components=100
- Features encoder depth = 9
- Cognition encoder depth = 6
- Latent space dimension = 6
- Features encoder dimensions = [100, 90, 80, 70, 60, 50, 40, 30, 20]
- Cognition encoder dimensions = [12, 11, 10, 9, 8, 7]
- n_splits=10
- n_repeats=5

![](https://notes.inria.fr/uploads/upload_8e5861298b49482061d24ffdb7439ac8.png)
![](https://notes.inria.fr/uploads/upload_7b2ac7b8e0bd45cd1cb199fc6e384fdd.png)
![](https://notes.inria.fr/uploads/upload_1a0335740cce80c3f673809f8ccfed51.png)
![](https://notes.inria.fr/uploads/upload_870d159ca5da98eb359448d93861c3e7.png)
![](https://notes.inria.fr/uploads/upload_8d68a0eb4a0e722a7afc8bd507ac55f5.png)
![](https://notes.inria.fr/uploads/upload_8a5a312fe52d5a03358787dca51ab4fd.png)
![](https://notes.inria.fr/uploads/upload_89dbfc18960f130dac410f60f3896b01.png)
![](https://notes.inria.fr/uploads/upload_ae8d375624887b8da66a3f64fc96c8b9.png)

(regression_2446573_44)


# Next Step

Using Ridge regression, which is capable of performing linear regressions, measure its metrics and use it as a base to develop a more complex model capable of performing non-linear regressions. 

### Baseline - Simple Linear Ridge Regression
```
class RidgeRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x)
```
![](https://notes.inria.fr/uploads/upload_bf4eb45d64aef14f8caa4f2e78ff744a.png)
![](https://notes.inria.fr/uploads/upload_e28fe80b26ba798e1107c5e262be4273.png)
![](https://notes.inria.fr/uploads/upload_8b6b194582f93802e853851086f28441.png)
![](https://notes.inria.fr/uploads/upload_4568e51a8d2ed18202c70266c618dedd.png)
![](https://notes.inria.fr/uploads/upload_3e379a5880aeebfb40cdb07716881cea.png)

### Code snap for Multi-Layer Perceptron (MLP) regression

#### NN Architecture
```
class NonlinearRidgeRegression(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(NonlinearRidgeRegression, self).__init__()
        self.mlp = nn.Sequential(
            (nn.Linear(input_dim, hidden_dims[0])),
            (nn.ReLU()),
        )
        for i in range(1, len(hidden_dims)):
            self.mlp.add_module(f'linear_{i}', nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.mlp.add_module(f'relu_{i}', nn.ReLU())
        self.mlp.add_module(f'output_dim', nn.Linear(hidden_dims[-1], output_dim))
    def forward(self, x):
        x = self.mlp(x)
        return x

```

#### Skorch implementation

```
custom_model = NonlinearRidgeRegressionNet(
                module=NonlinearRidgeRegression,
                module__input_dim=input_dim,
                module__hidden_dims=hidden_dim,
                module__output_dim=output_dim,
                optimizer__weight_decay=alpha,
                criterion=nn.MSELoss,
                optimizer=torch.optim.AdamW,
                optimizer__lr=0.01,
                max_epochs=7500,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                callbacks = [('lr_scheduler', LRScheduler(policy=ReduceLROnPlateau,mode='min', factor=0.1, patience=50)),
                             ('mape_scoring_train', EpochScoring(scoring='neg_mean_absolute_percentage_error', lower_is_better=True, on_train=True)),
                             ('mape_scoring_test',  EpochScoring(scoring=None, lower_is_better=True, on_train=False)),
                             ('early_stopper', EarlyStopping(monitor='train_loss',
                                                patience=1000,
                                                threshold=0.0001
                                                )),
                             ]
            )
```


Note: Skorch implements the L2 regularization with a parameter defined as "optimizer__weight_decay" so there is no need to reimplement it.



### Tests
Note: over these tests, we apply PCA to work with 100 components.

#### First test

Two layers with an activation function (ReLU)

Hidden dimension = 512

![](https://notes.inria.fr/uploads/upload_410b7e4a1d320f039890a6f58a2a0871.png)
![](https://notes.inria.fr/uploads/upload_f23d416226873582abb4aea95ff51b31.png)
![](https://notes.inria.fr/uploads/upload_f86b0fe5d634ba35fad7e224be700ac0.png)
![](https://notes.inria.fr/uploads/upload_e32db9d3cf4017457dfd7aeb79f2734f.png)
![](https://notes.inria.fr/uploads/upload_bfc364915c39aba82b52f6222996c220.png)



#### Second test

Two layers with an activation function (ReLU)
Hidden dimension = 8192
![](https://notes.inria.fr/uploads/upload_9bb3d59e5bf15432cbcd50eeacddf818.png)

![](https://notes.inria.fr/uploads/upload_3a3dbc9b2ea395a2efa795f943ebe062.png)
![](https://notes.inria.fr/uploads/upload_6cc140afa4f76e941b5b8d0e47cec34d.png)
![](https://notes.inria.fr/uploads/upload_e14346984018cad86311614dd6e1eadd.png)

Adding more dimensionality to the hidden layer doesn't improve the results.

#### Third test
Three layers with ReLU
Hidden dimensions = [64, 64, 64]

![](https://notes.inria.fr/uploads/upload_dd07d9f97ebade89fd066ab51422b6b5.png)
![](https://notes.inria.fr/uploads/upload_1562cf44bd2c3f9fb50b9032353e5ec0.png)
![](https://notes.inria.fr/uploads/upload_387d9774d96de1ba9a7dec192510f53b.png)
![](https://notes.inria.fr/uploads/upload_3b610e192e6ba777e20b638168c91e83.png)

#### Fourth test
Eight layers with ReLU
Hidden dimensions = [90, 80, 70, 60, 50, 40, 30, 20]

![](https://notes.inria.fr/uploads/upload_71136d0feb51f76059e0c00d75a23480.png)
![](https://notes.inria.fr/uploads/upload_31460a96ef900633d595c80e16d3420a.png)
![](https://notes.inria.fr/uploads/upload_731834ddc60fa6c1c8e2223ca279152e.png)
![](https://notes.inria.fr/uploads/upload_7443e2fa94bdf283341e25b4cf8a373d.png)

#### Fifth test
Eight layers with ReLU
Hidden dimensions = [512, 512, 512, 512, 512, 512, 512, 512]
![](https://notes.inria.fr/uploads/upload_a770ede1562e6be38c2283b75a33976b.png)
![](https://notes.inria.fr/uploads/upload_8dfb796ed8f39788081953cd79db24f8.png)
![](https://notes.inria.fr/uploads/upload_e9b7fc1972f54cc73177395531341ed9.png)
![](https://notes.inria.fr/uploads/upload_8d2e009db887719df43b6b1a58f6f880.png)



#### Sixth test
Seventeen layers with ReLU
Hidden dimensions = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15]

![](https://notes.inria.fr/uploads/upload_1cd3efc7b637eae06370bc527a3a607f.png)
![](https://notes.inria.fr/uploads/upload_0e391d46e70112050ef9bd19c655de0c.png)
![](https://notes.inria.fr/uploads/upload_e3d5775d67b92e48a2ffdb4ea0906952.png)
![](https://notes.inria.fr/uploads/upload_b66dca5200edb7f1c3294ac52acde0dd.png)


### Conclusions

1. Adding a ReLU and a hidden layer of dimension 512 significantly decreases the losses. The loss in training reaches almost 0, while in testing it ranges between 1 and 2.
2. Increasing the dimension to 8192 helps decrease the test loss even further, but it becomes quite unstable at a certain point. However, the R2 metric does not improve.
3. Adding more layers (3) of dimension 64 causes the training loss to have slightly more oscillatory values, while the test loss does not decrease.
4. Adding more layers (8) does not result in significant changes, only greater stability in MAPE. If these layers have a higher dimensionality (512), it worsens the situation: the R2 Score is much lower.
5. Adding even more layers (17) only makes the model more stable, but maintains excessively high loss values, so it does not improve the regression performance.

Overall, simply adding a single hidden layer with ReLU improved the result, whereas adding more complexity did not.


## Deep Residual Learning

Inspired on [Deep Residual Learning for Nonlinear Regression](https://www.researchgate.net/publication/339109948_Deep_Residual_Learning_for_Nonlinear_Regression) to develop this architecture and by [A comparative study of 11 non-linear regression models](https://www.nature.com/articles/s41598-024-55243-x) to measure the different metrics, I coded and implemented the following architecture:

### NN Architecture

```
class IdentityBlock(nn.Module):
    def __init__(self, input_dim, units):
        super(IdentityBlock, self).__init__()
        self.units = units
        self.block = nn.Sequential(
            nn.Linear(input_dim, units),
            nn.BatchNorm1d(units),
            nn.ReLU(),

            nn.Linear(units, units),
            nn.BatchNorm1d(units),
            nn.ReLU(),

            nn.Linear(units, units),
            nn.BatchNorm1d(units)
        )

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += identity 
        out = F.relu(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, input_dim, units):
        super(DenseBlock, self).__init__()
        self.units = units
        self.block = nn.Sequential(
            nn.Linear(input_dim, units),
            nn.BatchNorm1d(units),
            nn.ReLU(),

            nn.Linear(units, units),
            nn.BatchNorm1d(units),
            nn.ReLU(),

            nn.Linear(units, units),
            nn.BatchNorm1d(units)
        )
        self.shortcut = nn.Sequential(
            nn.Linear(input_dim, units),
            nn.BatchNorm1d(units)
        )

    def forward(self, x):
        identity = x
        out = self.block(x)
        shortcut = self.shortcut(identity)
        out += shortcut
        out = F.relu(out)
        return out

class ResNet50Regression(nn.Module):
    def __init__(self, input_dim, output_dim, width=512):
        super(ResNet50Regression, self).__init__()
        self.width = width
        # Sequential blocks
        self.dense_blocks = nn.Sequential(
            DenseBlock(input_dim, width),
            IdentityBlock(width, width),
            IdentityBlock(width, width),

            DenseBlock(width, width),
            IdentityBlock(width, width),
            IdentityBlock(width, width),

            DenseBlock(width, width),
            IdentityBlock(width, width),
            IdentityBlock(width, width)
        )
        # Final layers
        self.final_layers = nn.Sequential(
            nn.BatchNorm1d(width),
            nn.Linear(width, output_dim)
        )
    def forward(self, x):
        x = self.dense_blocks(x)
        x = self.final_layers(x)
        return x
```

### Skorch Implementation

```
custom_model = ResNet50RegressionNet(
        module=ResNet50Regression,
        module__input_dim=input_dim,
        module__output_dim=output_dim,
        optimizer__weight_decay=1,
        criterion=nn.MSELoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=0.001,
        max_epochs=7500,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks = [('mape_scoring_train', EpochScoring(scoring=mape_train_score, lower_is_better=False, on_train=True)),
                     ('mape_scoring_test',  EpochScoring(scoring=mape_test_score, lower_is_better=False, on_train=False)),      
                     ('mse_score_train',    EpochScoring(scoring=mae_train_score, lower_is_better=True, on_train=True)),
                     ('mse_score_test',     EpochScoring(scoring=mae_test_score, lower_is_better=True, on_train=False)),
                     ('early_stopper',      EarlyStopping(monitor='train_loss', patience=100, threshold=0.0001)),
                     ('lr_scheduler',       LRScheduler(policy=ReduceLROnPlateau,mode='min', factor=0.1, patience=50))
                     ] )
```

### Test with PCA to 100 components

![](https://notes.inria.fr/uploads/upload_79fa0d829316d241a678cf53c9c59351.png)
![](https://notes.inria.fr/uploads/upload_e9b081ba056bbf5cae59f17c86cff82d.png)
![](https://notes.inria.fr/uploads/upload_0b14bd5ebc184a68f936143309961577.png)
![](https://notes.inria.fr/uploads/upload_65c496c507326f76f2361ac705c726a4.png)
![](https://notes.inria.fr/uploads/upload_3a6c30c814404fc3441ee078a4fb8bf9.png)
![](https://notes.inria.fr/uploads/upload_26af5117906a39338054f93194afc0bc.png)
![](https://notes.inria.fr/uploads/upload_cb33c71338d7b1317b6cf2c59e074c99.png)

### Test without PCA

![](https://notes.inria.fr/uploads/upload_11d1a8fb1ccf722e19f131586d39aff5.png)
![](https://notes.inria.fr/uploads/upload_b43f76f443a671d1b692ab99437d0d2a.png)
![](https://notes.inria.fr/uploads/upload_294c056f8a6cbf8367e3beabae2acfdd.png)
![](https://notes.inria.fr/uploads/upload_9fd08bf7e2e27517802a490bc9233d4d.png)
![](https://notes.inria.fr/uploads/upload_93959dbbd2677e46b52e3175697da359.png)
![](https://notes.inria.fr/uploads/upload_80b69c1379af59e18f8caa4470686b3e.png)
![](https://notes.inria.fr/uploads/upload_464c8ea9c1f26a60cda333aec2932f24.png)

![](https://notes.inria.fr/uploads/upload_b9f21b9ff5ffdcf6a1156c933a0eabf9.png)

### Conclusions

The training metrics show improvement, but the test metrics do not. This pattern is evident in the MAPE and the valid_loss for each model and parameter adjustment. While the train loss decreases to zero, the valid loss remains consistently high, no matter how "complex" is the model.


-----
### Test with Support Vector Regression

 ``` 
 regr = MultiOutputRegressor(make_pipeline(StandardScaler(), NuSVR(C=1.0, nu=0.1)))
regr.fit(X_train, y[train_idx])
y_pred = regr.predict(X_test)
```

![](https://notes.inria.fr/uploads/upload_5f9f3e3c59ccb7a23023792e0b5dd81b.png)


## New plots and Metrics

In this section we are going to take only an MLP with one layer with dimension 512. (First Test)

- Dataset is standardized with the mean removed and scaled to unit variance. (Features and cognition variables)

### PCA with 100 components
![](https://notes.inria.fr/uploads/upload_454c90f2f0fd4c044f8b46b95381d4ab.png)
![](https://notes.inria.fr/uploads/upload_f3cf4a2efda6dedaa43b003de19d3ce9.png)
![](https://notes.inria.fr/uploads/upload_509a282613ab57e8597b7d8eb7657e5d.png)
![](https://notes.inria.fr/uploads/upload_5bdc343189b0e49d98d691d42114c16c.png)
![](https://notes.inria.fr/uploads/upload_1d6ec851f6b4a068e36dc0986b1fc13c.png)
![](https://notes.inria.fr/uploads/upload_2d5405e33262865b941def6e66a0a627.png)


### PCA with 190 components
![](https://notes.inria.fr/uploads/upload_add8b10b46680f132ea8b8387f04041c.png)
![](https://notes.inria.fr/uploads/upload_d1a21571cf6e02e60f0a69c5452c0d69.png)

![](https://notes.inria.fr/uploads/upload_9262d3c0cc4dc1012c53e1b25da69556.png)
![](https://notes.inria.fr/uploads/upload_42814043c19484804bf9b5c7a381e98d.png)
![](https://notes.inria.fr/uploads/upload_17bbc5782bbba65c80c0cf20f5a40baa.png)
![](https://notes.inria.fr/uploads/upload_f49e40316f6c35f06db9eba1b0dcd1ad.png)

### Conclusions

- With 100 principal components, PCA captured only 54% of the explained variance. However, increasing the number of components to 190, it captures 97% of the explained variance.
- Using 190 components in PCA, the R2 score was improved and MAPE on the testing set was reduced, indicating a better performance of the model.
- Normalization contributed to slightly better results.



## Returning to Multimodal embedding idea...

Recall:
> The main idea was:
> $$
> \arg\min \mathcal L(\gamma, \beta, \eta) = \sum_i d(f_\gamma(X_i), g_\beta(Y_i)) + \alpha\|h_\eta(g_\beta(Y_i))-Y_i\|^2_2 
> $$
> for each subject $i$, where $d$ can be the $L_2$ norm. This means that $h_\eta = g^{-1}_\beta$

Now, if we think the new loss as:
$$
\arg\min \mathcal L(\gamma, \beta, \eta, \delta) = \|h_\eta(g_\beta(Y_i))-h_\delta(f_\gamma(X_i))\|^2_2 + \|h_\eta(g_\beta(Y_i))-Y_i\|^2_2 + \|h_\delta(f_\gamma(X_i))-Y_i\|^2_2
$$

We use three terms in our new loss function:
1. The distance between decoder outputs of the latent space mappings of $Y_i$ and $X_i$.
2. The distance between the decoder output of $Y_i$ and the actual label $Y_i$.
3. The distance between the decoder output of $X_i$ and the actual label $Y_i$.


### Code snap

#### NN Architecture
```
class MultiModalEmbedding(nn.Module):
    def __init__(self, f_input_dim, c_input_dim, f_layers_dim, c_layers_dim, latent_space_dim, alpha=1):
        super(MultiModalEmbedding, self).__init__()
        self.alpha = alpha

        # Features Encoder
        self.f_encoder = nn.Sequential(
            nn.Linear(f_input_dim, f_layers_dim[0]),
            nn.GELU(),
        )
        for i in range(len(f_layers_dim) - 1):
            self.f_encoder.add_module(f'linear_{i}', nn.Linear(f_layers_dim[i], f_layers_dim[i+1]))
            self.f_encoder.add_module(f'relu_{i}', nn.GELU())
        self.f_encoder.add_module(f'latent_space', nn.Linear(f_layers_dim[-1], latent_space_dim))

        # Cognition Encoder
        self.c_encoder = nn.Sequential(
            nn.Linear(c_input_dim, c_layers_dim[0]),
            nn.GELU(),
        )
        for i in range(len(c_layers_dim) - 1):
            self.c_encoder.add_module(f'linear_{i}', nn.Linear(c_layers_dim[i], c_layers_dim[i+1]))
            self.c_encoder.add_module(f'relu_{i}', nn.GELU())
        self.c_encoder.add_module(f'latent_space', nn.Linear(c_layers_dim[-1], latent_space_dim))

        # Cognition Decoder
        self.c_decoder = nn.Sequential(
            nn.Linear(latent_space_dim, c_layers_dim[-1]),
            nn.GELU(),
        )
        for i in range(len(c_layers_dim) - 1, 0, -1):
            self.c_decoder.add_module(f'linear_{i}', nn.Linear(c_layers_dim[i], c_layers_dim[i-1]))
            self.c_decoder.add_module(f'relu_{i}', nn.GELU())
        self.c_decoder.add_module(f'input_space', nn.Linear(c_layers_dim[0], c_input_dim))


    def forward (self, x, y):
        f_gamma = self.f_encoder(x)
        g_beta  = self.c_encoder(y)
        h_eta   = self.c_decoder(g_beta)
        h_eta_f_gamma = self.c_decoder(f_gamma)

        return f_gamma, g_beta, h_eta, h_eta_f_gamma
    

class MultiModalEmbeddingNet(skorch.NeuralNet):    
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        f_gamma, g_beta, h_eta, h_eta_f_gamma = y_pred
        
        reconst_loss = super().get_loss(h_eta, h_eta_f_gamma, *args, **kwargs)
        self.history.record('reconst_loss', reconst_loss.item())
        y_true_loss  = super().get_loss(h_eta, y_true, *args, **kwargs)
        self.history.record('y_true_loss', y_true_loss.item())
        y_from_x_loss = super().get_loss(h_eta_f_gamma, y_true, *args, **kwargs)
        self.history.record('y_from_x_loss', y_from_x_loss.item())
               
        loss = reconst_loss + y_true_loss + y_from_x_loss

        return loss
```

#### Skorch Implementation
```
f_input_dim = input_dim
c_input_dim = output_dim
f_layers_dim     = [1024, 1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 32, 32]
c_layers_dim     = [256, 256, 256, 256]
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
```



### Experiments


#### Constant parameters
n_pca_components=190
n_splits=10
n_repeats = 5
latent_space_dim = 6


#### First test (1 repeat)


f_layers_dim     = [180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
c_layers_dim     = [256, 256]



##### Plots during training (Losses)
![](https://notes.inria.fr/uploads/upload_bedfe319bfe93172912fdaab9be22470.png)
![](https://notes.inria.fr/uploads/upload_13b383addf2a935e9aa9eea39e7eb95e.png)
![](https://notes.inria.fr/uploads/upload_d150c5db3c3ded2e892011151aa8ba0f.png)
![](https://notes.inria.fr/uploads/upload_790fa584cf89a11d0bafd088a6a16c60.png)
![](https://notes.inria.fr/uploads/upload_d2c16592cd07ce3b91eaf65998aafc7d.png)

##### Plots during testing (Metrics)

![](https://notes.inria.fr/uploads/upload_7c0d3189eae498675de7e5c168ed7090.png)
![](https://notes.inria.fr/uploads/upload_8067706b2533ed75274688034520d92a.png)




##### First clues
- The reconstruction loss remains very low, indicating that the points between $X$ and $Y$ in the latent space are well-aligned.

- The autoencoder also trains very effectively.

- However, the loss for $Y$ predicted from $X$ remains high and does not decrease significantly.  **Could this be due to the dimensions of the decoder being unable to capture these features? Or perhaps it is related to the encoder's handling of the features themselves?**

- I think MAPE is not the best metric to measure because is very sensitivity to small true values. This is the mape formula: $\text{MAPE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \frac{{}\left| y_i - \hat{y}_i \right|}{\max(\epsilon, \left| y_i \right|)}$.
For example, if we have true value ($y_i$) = 0.1, and predicted value ($\hat{y}_i$)= 1.1, MAPE is going to be = 10 or 1000%. 


#### Second test

f_layers_dim=[1024, 1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 32, 32]
c_layers_dim=[512, 512]

##### Plots during training (Losses)
![](https://notes.inria.fr/uploads/upload_96b5e4f95ec1356352d2b7831538a54e.png)
![](https://notes.inria.fr/uploads/upload_ef01e2d3af3cc4c708306e740b9d7406.png)
![](https://notes.inria.fr/uploads/upload_8467bcfa6325bd322640eae76594d2c9.png)
![](https://notes.inria.fr/uploads/upload_ba3029aed36d765130c0660de20e8fa6.png)
![](https://notes.inria.fr/uploads/upload_0b6e02ffc9c03c0953b91c2c809fd1f2.png)
##### Plots during testing (Metrics)
![](https://notes.inria.fr/uploads/upload_52e43388513ef255608dd02a4724f28c.png)
![](https://notes.inria.fr/uploads/upload_40e06dc7dd4081ff1119385f731fb146.png)



#### Third test
f_layers_dim=[1024, 1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 32, 32]
c_layers_dim=[1024, 512, 256]
##### Plots during training (Losses)
![](https://notes.inria.fr/uploads/upload_14d0707e1cd3ae20c37f3d810c64a3fd.png)
![](https://notes.inria.fr/uploads/upload_d1171e4b02ec86d4ae92d290d19ff94d.png)
![](https://notes.inria.fr/uploads/upload_937a4284fae3bb8c7f82ba3955ee9237.png)
![](https://notes.inria.fr/uploads/upload_4c009f58db8c4343e5149324197f5584.png)
![](https://notes.inria.fr/uploads/upload_3504eaf5bc98b8f25a5572114bf541c9.png)
##### Plots during testing (Metrics)
![](https://notes.inria.fr/uploads/upload_97ef30ab2076f4b6c67b2e904ede65c4.png)
![](https://notes.inria.fr/uploads/upload_72c8f9dbb3f9ecc948e00efcbc195201.png)

#### Fourth test
f_layers_dim=[180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20]
c_layers_dim=[512, 512]

##### Plots during training (Losses)
![](https://notes.inria.fr/uploads/upload_8386d6acee16ffae731d0e014e08c8e1.png)
![](https://notes.inria.fr/uploads/upload_e50cc56030df34a6f7d5077a9ba0b59d.png)
![](https://notes.inria.fr/uploads/upload_91773c220f31dc1a04f87024e0730e33.png)
![](https://notes.inria.fr/uploads/upload_bd1cb889495d648bf37603fbb793980a.png)
![](https://notes.inria.fr/uploads/upload_41efc563e38cd44a031529cd33e15332.png)

##### Plots during testing (Metrics)
![](https://notes.inria.fr/uploads/upload_910f2cde1fa621bf49348e78efb0b8bd.png)
![](https://notes.inria.fr/uploads/upload_d8c64892cf5a4f39f92b572bd41269ff.png)

#### Fifth test
DATOS NORMALIZADOS PREVIO A PCA
f_layers_dim=[1024, 1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 32, 32]
c_layers_dim=[512, 512]
##### Plots during training (Losses)
![](https://notes.inria.fr/uploads/upload_8a367767f69b88f82dde88cf7cbf18fd.png)
![](https://notes.inria.fr/uploads/upload_133ac8fd8d6854be944261e4aa8b6392.png)
![](https://notes.inria.fr/uploads/upload_1ab0a6087b07998581569fe4e9285c77.png)
![](https://notes.inria.fr/uploads/upload_7ba2d2b43050193533a5dca05e9bbd80.png)
##### Plots during testing (Metrics)
![](https://notes.inria.fr/uploads/upload_64e31f3cb8b589f77ba2732d0fc6737a.png)
![](https://notes.inria.fr/uploads/upload_2dab8e7d06b4a40e6d7d36dc46f33715.png)

#### Sixth test
NORMALIZADO ANTES Y DESP DE PCA
f_layers_dim=[1024, 1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 32, 32]
c_layers_dim=[512, 512]
##### Plots during training (Losses)
![](https://notes.inria.fr/uploads/upload_0f10fc2260a0d5f2d4bc8ab493dec693.png)
![](https://notes.inria.fr/uploads/upload_7d8920e12185a9321b13977ef9276701.png)
![](https://notes.inria.fr/uploads/upload_c337bebd519793b5e4adf98f6d8e0879.png)
![](https://notes.inria.fr/uploads/upload_2c7a366eb8c906bf9ce04b4aea447895.png)
![](https://notes.inria.fr/uploads/upload_ec8668b241fdec7f069d8f6a4efadec8.png)


##### Plots during testing (Metrics)
![](https://notes.inria.fr/uploads/upload_297f2d8a23909756c6c06f42f3f14a58.png)
![](https://notes.inria.fr/uploads/upload_f8a1acdb39f2925a6c54bbb927b45dfe.png)


#### Testing the autoencoder loss


##### Model 1: Autoencoder with 3 layers of 11, 9, 7 neurons, using ReLU activation for each layer

Test MAPE: 9.4474483
Train MAPE: 0.58291

##### Model 2: Autoencoder with 3 layers of 512 neurons each, using ReLU activation for each layer

Test MAPE: 4.4474483
Train MAPE: 0.005852234

##### Model 3: Autoencoder with 2 layers of 2048 and 512 neurons, using ReLU activation for each layer

Test MAPE: 1.5439097
Train MAPE: 0.1717046

##### Model 4: Autoencoder with 2 layers of 2048 neurons each, using ReLU activation for each layer

Test MAPE: 5.940394
Train MAPE: 0.9548608
 
##### Conclusion for autoencoder part

Mape in test set is very high. It seems to be an overfitting there. Nevertheless, I figured it out the need of having more neurones per layers since the previous experiments had only 10 and 8 neurons each layer. It improved a lot the results. 

### Last tests
#### First test
f_layers_dim=[1024, 1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 32, 32]
c_layers_dim=[256, 256, 256, 256]
latent_space_dim = 6
PCA = 190 components
##### Plots during training (Losses)
![](https://notes.inria.fr/uploads/upload_818222002d5896eedeb71aaa8ab37e1e.png)
![](https://notes.inria.fr/uploads/upload_dc3118c1a99c70a1aee6f01971786563.png)
![](https://notes.inria.fr/uploads/upload_2907bb355631539e25856da2ca3dc023.png)
![](https://notes.inria.fr/uploads/upload_3d3dab70b60c135c43d47d3ce1510b76.png)
![](https://notes.inria.fr/uploads/upload_d59c742f200a777923777b8e82955409.png)
##### Plots during testing (Metrics)
![](https://notes.inria.fr/uploads/upload_9e8844398a7041cdd2d802799824e966.png)
![](https://notes.inria.fr/uploads/upload_def509b85d1b3586bbd0f798fa909251.png)
#### Second test
f_layers_dim=[1024, 1024, 1024, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 32, 32]
c_layers_dim=[1024, 512, 256, 128]
latent_space_dim = 6
PCA = 190 components

##### Plots during training (Losses)
![](https://notes.inria.fr/uploads/upload_b5c463ac9e99d39d3a1776f074ed1970.png)
![](https://notes.inria.fr/uploads/upload_8f2404a4a82e47887b0a96e680de4291.png)
![](https://notes.inria.fr/uploads/upload_96e97aac9d79270e087c105d33057b25.png)
![](https://notes.inria.fr/uploads/upload_f6672a3a91d4192d7cbca3b062898657.png)

##### Plots during testing (Metrics)
![](https://notes.inria.fr/uploads/upload_1a9559610aa5d72a97948542e61211c3.png)
![](https://notes.inria.fr/uploads/upload_410d92289340c3e72b8e772e13c8287b.png)



### Conclusion

If we compare the results obtained in the previous section (0.0028 +/- 0.0223) with those obtained with this new loss function in this section, we can observe the remarkable improvement of the model performance However, I believe that there is room for improvement since the results vary considerably depending on the depth and number of neurons per layer used in the decoder.


