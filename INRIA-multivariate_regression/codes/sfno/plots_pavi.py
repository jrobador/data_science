import matplotlib.pyplot as plt
import plots

def plot_sphere(sph_data, net_number):
    fig = plt.figure(layout='constrained')
    fig.suptitle("Resampled data for network " + str(net_number+1) + " left")
    subfigs = fig.subfigures(1, 5).ravel()

    for i, subfig in enumerate(subfigs):
        plots.plot_sphere(sph_data[i], fig=subfig, cmap='plasma')
