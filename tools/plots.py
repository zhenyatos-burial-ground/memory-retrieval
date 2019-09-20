"""
Copyright (C) <2019>  <de la Torre-Ortiz C, Nioche A>
See the 'LICENSE' file for details.

The docstring for a module should generally list the classes, exceptions
and functions (and any other objects) that are exported by the module,
with a one-line summary of each. (These summaries generally give less
detail than the summary line in the object's docstring.) The docstring
for a package (i.e., the docstring of the package's __init__.py module)
should also list the modules and subpackages exported by the package.

:class Network: artificial neural network model of memory retrieval
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

FIG_FOLDER = "fig"
os.makedirs(FIG_FOLDER, exist_ok=True)


def phi(phi_values, dt=1.):
    """
    The docstring for a module should generally list the classes, exceptions
    and functions (and any other objects) that are exported by the module,
    with a one-line summary of each. (These summaries generally give less
    detail than the summary line in the object's docstring.) The docstring
    for a package (i.e., the docstring of the package's __init__.py module)
    should also list the modules and subpackages exported by the package.

    :class Network: artificial neural network model of memory retrieval
    """

    fig, ax = plt.subplots()

    n_iteration = len(phi_values)

    x = np.arange(n_iteration, dtype=float) * dt
    y = phi_values

    ax.plot(x, y)
    ax.set_title("Inhibitory oscillations")
    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("$\phi$")

    plt.savefig(os.path.join(FIG_FOLDER, "sin.pdf"))


def noise(noise_values, dt=1.):

    fig, ax = plt.subplots()

    n_iteration = noise_values.shape[1]

    x = np.arange(n_iteration, dtype=float) * dt
    ys = noise_values

    for y in ys:
        ax.plot(x, y, linewidth=0.5, alpha=0.2)

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Noise")

    plt.savefig(os.path.join(FIG_FOLDER, "noise.pdf"))


def activity_curve(firing_rates, dt=1.):

    fig, ax = plt.subplots()

    n_iteration = firing_rates.shape[1]

    x = np.arange(n_iteration, dtype=float) * dt
    ys = firing_rates

    for i, y in enumerate(ys):
        ax.plot(x, y, linewidth=0.5, alpha=1)

    ax.set_xlabel('Time (cycles)')
    ax.set_ylabel('Average firing rate')

    plt.savefig(os.path.join(FIG_FOLDER, "activity_curve.pdf"))


def activity_image(firing_rates, dt=1.):

    fig, ax = plt.subplots()

    n_memory, n_iteration = firing_rates.shape

    im = ax.imshow(firing_rates, cmap="jet",
                   extent=[
                        0, n_iteration * dt,
                        n_memory - 0.5, -0.5
                   ])

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Attractor number")

    fig.colorbar(im, ax=ax)

    ax.set_aspect(aspect='auto')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    plt.savefig(os.path.join(FIG_FOLDER, "activity_image.pdf"))


def inhibition(inhibition_values, dt=1.):

    fig, ax = plt.subplots()

    n_iteration = len(inhibition_values)

    x = np.arange(n_iteration, dtype=float) * dt
    y = inhibition_values

    ax.plot(x, y)

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Inhibition")

    plt.savefig(os.path.join(FIG_FOLDER, "inhibition_values.pdf"))


def weights(weights_array, name='weights_array'):

    fig, ax = plt.subplots()

    ax.set_xlabel("Neuron v")
    ax.set_ylabel("Neuron w")

    ax.set_title(name)

    im = ax.imshow(weights_array)

    fig.colorbar(im, ax=ax)

    plt.savefig(os.path.join(FIG_FOLDER, f"{name}.pdf"))


def current_curve(currents, dt=1., name="current"):

    fig, ax = plt.subplots()

    n_iteration = currents.shape[1]

    x = np.arange(n_iteration, dtype=float) * dt
    ys = currents

    for i, y in enumerate(ys):
        ax.plot(x, y, linewidth=0.5, alpha=1)

    ax.set_xlabel("Time (cycles)")
    ax.set_ylabel("Average current")
    ax.set_title(f"{name}")

    plt.savefig(os.path.join(FIG_FOLDER, f"{name}.pdf"))


def romani(activity):

    fig, ax = plt.subplots()
    im = ax.imshow(activity, aspect="auto",
                   cmap="jet")
    fig.colorbar(im, ax=ax)

    ax.set_xlabel("Time")
    ax.set_ylabel("Memories")

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()

    plt.show()
