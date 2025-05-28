import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

"""
This module provides functions to generate different layouts for wind farms.

Generate_circular_farm functions create circular arrangements of turbines,
- Originally coded by #JensMVPeter 
"""

def generate_square_grid(turbine, nx, ny, xDist, yDist):
    """
    Create a square grid of turbines.

    Parameters
    ----------
    turbine : WindTurbine
        The wind turbine object.
    nx : int
        Number of turbines in the x-direction.
    ny : int
        Number of turbines in the y-direction.
    xDist : float
        Diameter distance between turbines in the x-direction.
    yDist : float
        Diameter distance between turbines in the y-direction.

    Returns
    -------
    np.ndarray
        Array of turbine positions.
    """
    D = turbine.diameter()
    x = np.linspace(0, D * xDist * nx, nx)
    y = np.linspace(0, D * yDist * ny, ny)
    xv, yv = np.meshgrid(x, y, indexing="xy")

    x_pos = xv.flatten()
    y_pos = yv.flatten()

    return x_pos, y_pos


def generate_circle(n, r, angle_offset=0):
    """
    Generate a circular grid of n points with radius r.
    """
    if n > 1:
        t = np.linspace(0, 2 * np.pi, n + 1)[:n]
    else:
        t = [0]
    t += np.deg2rad(angle_offset)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y


def generate_cirular_farm(
    n_list: ArrayLike, 
    turbine, 
    r_dist: float = 5, 
    angle_offset_list: ArrayLike = None
):
    """
    Generate a circular farm of n circular grids with radius r and m points.
    """
    D = turbine.diameter()
    r_list = np.arange(len(n_list), dtype=float)
    if n_list[0] != 1:
        r_list = r_list + 0.5
    r_list *= r_dist * D  # Diameter factors


    x = []
    y = []
    if angle_offset_list is None:
        angle_offset_list = np.zeros(len(n_list))
    for n, r, angle_offset in zip(n_list, r_list, angle_offset_list):
        x_, y_ = generate_circle(n, r, angle_offset)
        x.append(x_)
        y.append(y_)
    x = np.concatenate(x)
    y = np.concatenate(y)
    return x, y


def plot_farm(x, y, turbine=None, D=None):
    """
    Plots the turbines in the farm layout, and their minimum distance to the closest turbine
    """
    if D is None and turbine is None:
        D = 1.0 # Default diameter if not provided
    elif turbine is not None:
        D = turbine.diameter()

    min_distances = []
    for i in range(len(x)):
        distances = np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2)
        distances[i] = np.inf  # Ignore self-distance
        min_distance = np.min(distances)
        min_distances.append(min_distance)
    min_distances = np.array(min_distances) / D # Convert to diameter units
    plt.scatter(x, y, c=min_distances, cmap='viridis', label='Min Distance (m)')
    plt.colorbar(label='Min Distance (Diameter Units)')
    # plt.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor='C0', label='Turbines'),
    #                     Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', label='Min Distance')])
    plt.show()