import torch
import torch.nn as nn
from torch_harmonics.sht import RealSHT, InverseRealSHT

from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.core.sphere import Sphere


class VMFConvolutionIsometric(nn.Module):
    r'''
    Convolution on the 2-sphere with the Von Mises-Fisher (VMF) kernel centered
    at $\phi = 0$, $\theta = 0$:
        $f(theta, phi) = C(\kappa)exp(\kappa\cos(\phi))$
        where $C(\kappa)$ is a constant depending on the concentration parameter such that $\int d\phi d\theta f=1$
    The kernel can be learnt as a linear perturbation of the VMF one. The resolution of the input and output will be changed by the ratios `input_ratio` and `output_ratio`
    '''
    def __init__(self, kappa: float, nlat: int, nlon: int, lmax=None, mmax=None, weights: bool=False, bias: bool=False, grid="equiangular", input_ratio: float=1., output_ratio: float=1.):
        super().__init__()
        self.input_ratio = input_ratio
        self.output_ratio = output_ratio

        in_nlat = int(round(nlat * self.input_ratio))
        in_nlon = int(round(nlon * self.input_ratio))

        out_nlat = int(round(nlat * self.output_ratio))
        out_nlon = int(round(nlon * self.output_ratio))

        self.lmax = lmax
        if self.lmax is None:
            self.lmax = in_nlat
        self.mmax = mmax
        if self.mmax is None:
            self.mmax = lmax

        self.output_ratio = output_ratio

        if kappa is not None:
            theta = torch.linspace(-torch.pi, torch.pi, in_nlon + 1)[1:]
            phi = torch.linspace(0, torch.pi, in_nlat)
            phi, theta = torch.meshgrid(phi, theta)
            self.kernel = nn.Parameter(nn.functional.normalize(torch.exp(kappa * torch.cos(phi)), dim=[0, 1]), requires_grad=False)
        else:
            self.kernel = nn.Parameter(torch.zeros((nlat, nlon)), requires_grad=False)
                                         
        self.sht = RealSHT(in_nlat, in_nlon, lmax=self.lmax, mmax=self.mmax, grid=grid)
        self.isht = InverseRealSHT(out_nlat, out_nlon, lmax=self.sht.lmax, mmax=self.sht.mmax, grid=grid)

        if kappa is None:
            kernel_coeffs = torch.zeros((self.sht.lmax, 1), dtype=torch.cdouble)
        else:
            kernel_coeffs = self.sht(self.kernel)[:, :1]
            if (kernel_coeffs == 0).all():
                raise ValueError("kappa too big")

        self.kernel_coeffs = nn.Parameter(
            kernel_coeffs,
            requires_grad=False
        )

        self.learn_weights = weights
        self.learn_bias = bias
        if self.learn_weights:
            self.weights = nn.Parameter(
                torch.ones_like(self.kernel_coeffs)
            )
        if self.learn_bias:
            self.bias = nn.Parameter(
                torch.zeros_like(self.kernel_coeffs)
            )
            
    def forward(self, x):
        xf = self.sht(x)
        kernel_coeffs = self.kernel_coeffs
        if self.learn_weights:
            kernel_coeffs = kernel_coeffs * self.weights
        if self.learn_bias:
            kernel_coeffs = kernel_coeffs + self.bias
        convolution = kernel_coeffs * xf
        x_est = self.isht(convolution)
        return x_est


class VMFConvolution(nn.Module):
    r'''
    Convolution on the 2-sphere with the Von Mises-Fisher (VMF) kernel centered
    at $\phi = 0$, $\theta = 0$:
        $f(theta, phi) = C(\kappa)exp(\kappa\cos(\phi))$
        where $C(\kappa)$ is a constant depending on the concentration parameter such that $\int d\phi d\theta f=1$
    The kernel can be learnt as a linear perturbation of the VMF one. The resolution of the input and output will be changed by the ratios `input_ratio` and `output_ratio`
    '''
    def __init__(self, kappa: float, nlat: int, nlon: int, lmax=None, mmax=None, weights: bool=False, bias: bool=False, grid="equiangular", input_ratio: float=1., output_ratio: float=1.):
        super().__init__()
        self.input_ratio = input_ratio
        self.output_ratio = output_ratio

        self.in_nlat = int(round(nlat * self.input_ratio))
        self.in_nlon = int(round(nlon * self.input_ratio))

        self.out_nlat = int(round(nlat * self.output_ratio))
        self.out_nlon = int(round(nlon * self.output_ratio))

        self.lmax = lmax
        if self.lmax is None:
            self.lmax = self.in_nlat
        self.mmax = mmax
        if self.mmax is None:
            self.mmax = lmax

        self.output_ratio = output_ratio

        if kappa is not None:
            theta = torch.linspace(-torch.pi, torch.pi, self.in_nlon + 1)[1:]
            phi = torch.linspace(0, torch.pi, self.in_nlat)
            phi, theta = torch.meshgrid(phi, theta)
            self._kernel = nn.Parameter(nn.functional.normalize(torch.exp(kappa * torch.cos(phi)), dim=[0, 1]), requires_grad=False)
        else:
            self._kernel = nn.Parameter(torch.zeros((nlat, nlon)), requires_grad=False)
                                         
        self.sht = RealSHT(self.in_nlat, self.in_nlon, lmax=self.lmax, mmax=self.mmax, grid=grid)
        self.isht = InverseRealSHT(self.out_nlat, self.out_nlon, lmax=self.sht.lmax, mmax=self.sht.mmax, grid=grid)
        self.coeff_indices = torch.tril_indices(self.sht.lmax, self.sht.mmax)

        if kappa is None:    
            kernel_coeffs = torch.zeros(self.coeff_indices.shape[1], dtype=torch.cdouble)
        else:
            kernel_coeffs = self.sht(self._kernel)[self.coeff_indices[0], self.coeff_indices[1]]
            if (kernel_coeffs == 0).all():
                raise ValueError("kappa too big")

        self.kernel_coeffs = nn.Parameter(
            kernel_coeffs,
            requires_grad=False
        )

        self.learn_weights = weights
        self.learn_bias = bias
        if self.learn_weights:
            self.weights = nn.Parameter(
                torch.ones_like(self.kernel_coeffs)
            )
        if self.learn_bias:
            self.bias = nn.Parameter(
                torch.zeros_like(self.kernel_coeffs)
            )
    def _coeffs(self):
        kernel_coeffs = self.kernel_coeffs
        if self.learn_weights:
            kernel_coeffs = kernel_coeffs * self.weights
        if self.learn_bias:
            kernel_coeffs = kernel_coeffs + self.bias
        coeffs = torch.zeros((self.sht.lmax, self.sht.mmax), dtype=kernel_coeffs.dtype, device=kernel_coeffs.device)
        coeffs[self.coeff_indices[0], self.coeff_indices[1]] = kernel_coeffs
        return coeffs
    
    def kernel(self):
        coeffs = self._coeffs()
        k = self.isht(coeffs)
        return k

    def forward(self, x):
        xf = self.sht(x)
        coeffs = self._coeffs()
        convolution = coeffs * xf
        x_est = self.isht(convolution)
        return x_est


def compute_new_angles_grid(nlat: int, nlon: int):
    new_phi = torch.linspace(0, 2 * torch.pi, nlon + 1)[1:]
    new_theta = torch.linspace(0, torch.pi, nlat)
    new_phi, new_theta = torch.meshgrid(new_phi, new_theta)

    return new_theta, new_phi


def resample_spherical_functions(src_vertices, data, dst_vertices, sh_order=4):
    if hasattr(src_vertices, 'shape'):
        src_vertices = tuple(src_vertices.T)

    if hasattr(dst_vertices, 'shape'):
        dst_vertices = tuple(dst_vertices.T)

    if len(src_vertices) == 2:
        src_sphere = Sphere(theta=src_vertices[0], phi=src_vertices[1])
    elif len(src_vertices) == 3:
        src_sphere = Sphere(x=src_vertices[0], y=src_vertices[1], z=src_vertices[1])
    else:
        raise ValueError("Problem with src_vertices")

    if len(dst_vertices) == 2:
        dst_sphere = Sphere(theta=dst_vertices[0], phi=dst_vertices[1])
    elif len(dst_vertices) == 3:
        dst_sphere = Sphere(x=src_vertices[0], y=src_vertices[1], z=src_vertices[1])
    else:
        raise ValueError("Problem with dst_vertices")
    
    sh = sf_to_sh(data, src_sphere, sh_order=sh_order)

    return sh_to_sf(sh, dst_sphere, sh_order=sh_order)