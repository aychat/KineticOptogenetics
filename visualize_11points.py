import matplotlib.pyplot as plt
import numpy as np
import pickle
import itertools

from matplotlib.colors import Normalize

# Load the data
with open("Data/result_11points.pickle", 'rb') as f:
    data = pickle.load(f)
    scan = data['result']

# find the position of maximum pop
indx = np.unravel_index(scan.argmax(), scan.shape)


axes_names = ['pump_energy', 'dump_energy', 'pump_width', 'dump_width', 't0_pump', 't0_dump']


#np.clip(scan, 0, 1, out=scan)
scan[(scan > 1)|(scan < 0.)] = np.nan

for axis in itertools.combinations(range(6), 4):
    plot_axis = tuple(sorted(
        set(range(6)).difference(axis)
    ))

    xlabel = axes_names[plot_axis[0]]
    ylabel = axes_names[plot_axis[1]]

    x_min = data['params'][xlabel].min()
    x_max = data['params'][xlabel].max()
    y_min = data['params'][ylabel].min()
    y_max = data['params'][ylabel].max()

    plt.clf()
    plt.imshow(
        np.mean(scan, axis=axis),
        origin='lower',
        interpolation='nearest',
        extent=[x_min, x_max, y_min, y_max],
        aspect=(x_max - x_min)/(y_max - y_min),
        vmin=np.nanmin(scan),
        vmax=np.nanmax(scan)
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.colorbar()
    plt.savefig('Plots_11_points_transfer_matrix/scan_%d_%d.png' % plot_axis)
