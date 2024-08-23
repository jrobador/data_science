from dipy.core.sphere import Sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf
import numpy as np

def interpolation_to_grid(vertices_left, vertices_right, sh_order=24):
    def compute_new_angles_grid(nlat, nlon):
        new_phi = np.linspace(0, 2 * np.pi, nlon + 1)[1:]
        new_theta = np.linspace(0, np.pi, nlat)
        new_phi, new_theta = np.meshgrid(new_phi, new_theta)

        return new_theta, new_phi

    print (f"{sh_order=}")

    nlat = int(np.sqrt(len(vertices_left)/2))
    nlon = 2 * nlat
    sphere_src_left = Sphere(xyz=vertices_left)
    sphere_src_right = Sphere(xyz=vertices_right)
    mesh_theta, mesh_phi = compute_new_angles_grid(nlat, nlon)
    sphere_dst = Sphere(theta=mesh_theta.ravel(), phi=mesh_phi.ravel())
    
    return mesh_theta, sphere_src_left, sphere_src_right, sphere_dst


def hemisphere_to_spherical(network, sphere_src, sphere_dst, sh_order=24):
    if (len(network.shape) == 2):
        sh             = sf_to_sh(network, sphere_src, sh_order_max=sh_order)
        spherical_data = sh_to_sf(sh, sphere_dst, sh_order_max=sh_order)
    elif(len(network.shape) == 3):
        spherical_data = np.zeros((network.shape[0], sphere_dst.vertices.shape[0], network.shape[2]))
        for i in range(network.shape[2]):
            network_slice = network[:, :, i]
            sh          = sf_to_sh(network_slice, sphere_src, sh_order_max=sh_order)
            spherical_data[:, :, i] = sh_to_sf(sh, sphere_dst, sh_order_max=sh_order)
    else: 
        raise ValueError("Network should have a length of 2 or 3.")

    return spherical_data


def spherical_to_hemisphere(spherical_data, sphere_src, sphere_dst, sh_order=24):
    sh_2                = sf_to_sh(spherical_data, sphere_dst, sh_order_max=sh_order)
    data_original_space = sh_to_sf(sh_2, sphere_src, sh_order_max=sh_order)

    return data_original_space