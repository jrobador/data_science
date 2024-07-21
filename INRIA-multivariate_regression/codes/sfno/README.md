# SFNO Convolution
The code is separated under different blocks:

1. Data and vertices representation
2. Interpolation to a grid
3. VMF Convolution
4. Autoencoder architecture

## Data and vertices representation

We take data for each network that is a logit with a softmax function. For vertices representation, we use HCP spherical mesh normalized. Then we apply a mask to avoid medial-area.

## Interpolation to a grid

We initialize parameters and grids for spherical coordinate system. Then convert this data to spherical function with DIPY kernels. Then show plots: One for data in spherical representation and convolved with $e^{-x}$, for each network.

## VMF Convolution

Here we define a convolution operation on the 2-sphere using the Von Mises-Fisher (VMF) kernel. The VMFConvolution class implements a convolution operation on spherical data using the VMF kernel, with support for adjustable input and output resolutions, and optional learning of kernel weights and bias.

The data is processed and visualized for different kappa values, demonstrating the effect of varying the concentration parameter on the convolution kernel and the resulting data.

## Autoencoder architecture

This code is essentially setting up and training a learnable VMF convolutional neural network (CNN) for processing spherical data.

Each VMF convolutional layer is defined with the following parameters:

- kappa: Set to None which means the kernel is learnable and not initialized with a predefined concentration.
- nlat and nlon: Number of latitude and longitude divisions.
- input_ratio and output_ratio: These control the input and output resolutions of each layer.
- weights: Set to False, meaning no learnable weights in the VMF convolution.
- bias: Set to True, meaning learnable bias is enabled.
