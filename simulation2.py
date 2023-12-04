"""
<[Re] Recanatesi (2015). Neural Network Model of Memory Retrieval>
Copyright (C) <2020>  <de la Torre-Ortiz C, Nioche A>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils.plots as plots
import utils.simulation as sim
from settings import paths

from embeddings import load_n_glove_embeddings, find_closest_embeddings, continuous_to_discrete

# -------------------------------------------------------------------------- #
# -------------------- Change parameters if needed here -------------------- #
# -------------------------------------------------------------------------- #

PATTERNS_DIR = paths.PATTERNS_SEEDS_DIR  # paths. ...
RECALLS_DIR = paths.RECALLS_SEEDS_DIR  # paths. ...

assert (
    PATTERNS_DIR is not None and RECALLS_DIR is not None
), "Please choose a saving directory"

try:
    JOB_ID = int(
        os.getenv("SLURM_ARRAY_TASK_ID")
    )  # Changes seed per cluster simulation
except:
    JOB_ID = 33 # Default seed for non-cluster use
    
if len(sys.argv) == 2:
    JOB_ID = int(sys.argv[1])

np.random.seed(JOB_ID)

PARAMETERS_DF = pd.read_csv(
    os.path.join(paths.PARAMETERS_DIR, "simulation2.csv"), index_col=0
)

NUM_NEURONS = int(PARAMETERS_DF.loc["num_neurons"].array[0])
NUM_MEMORIES = int(PARAMETERS_DF.loc["num_memories"].array[0])
# Activation
T_DECAY = PARAMETERS_DF.loc["t_decay"].array[0]
RECALL_THRESHOLD = PARAMETERS_DF.loc["recall_threshold"].array[0]
# Time
T_STEP = PARAMETERS_DF.loc["t_step"].array[0]
T_TOT = PARAMETERS_DF.loc["t_tot"].array[0]
T_SIMULATED = int(T_TOT // T_STEP)
# Hebbian rule
EXCITATION = PARAMETERS_DF.loc["excitation"].array[0]
SPARSITY = PARAMETERS_DF.loc["sparsity"].array[0]
# Gain
GAIN_THRESHOLD = PARAMETERS_DF.loc["gain_threshold"].array[0]
GAIN_EXP = PARAMETERS_DF.loc["gain_exp"].array[0]
# Inhibition
SIN_MIN = PARAMETERS_DF.loc["sin_min"].array[0] * EXCITATION
SIN_MAX = PARAMETERS_DF.loc["sin_max"].array[0] * EXCITATION
# Noise
NOISE_VAR = PARAMETERS_DF.loc["noise_var"].array[0]
# Forward and backward contiguity
CONT_FORTH = PARAMETERS_DF.loc["cont_forth"].array[0] / NUM_NEURONS
CONT_BACK = PARAMETERS_DF.loc["cont_back"].array[0] / NUM_NEURONS
# For parameter sweeps (uncomment to select)
# CONT_FORTH = sim.get_simulation_range_param("cont_forth", JOB_ID, 100) / NUM_NEURONS
# CONT_FORTH = sim.get_simulation_range_param("cont_forth_low", JOB_ID, 100) / NUM_NEURONS
# NOISE_VAR = sim.get_simulation_range_param("noise_var", JOB_ID, 100)

# -------------------------------------------------------------------------- #
# ------------------------- END parameter changes -------------------------- #
# -------------------------------------------------------------------------- #

from scipy import spatial
# Functions dependent on seed
def make_patterns(num_memories: int, n_base_words=4) -> np.ndarray:
    """Build memory neural patterns according to word embeddings."""
    embeddings_dict = load_n_glove_embeddings(10_000)
    base_words = np.random.choice(list(embeddings_dict.keys()), n_base_words)
    num_neurons = 100000
    output = np.zeros((num_neurons, num_memories))
    n_words_per_base = num_memories // n_base_words
    remainder = num_memories % n_base_words
    if remainder == 0:
        remainder = n_base_words
    cycle_count = 1
    word_count = 0
    used_words = []
    for word in base_words:
        if cycle_count >= n_base_words:
            similar_words = find_closest_embeddings(embeddings_dict, embeddings_dict[word])[0:n_words_per_base]
        else:
            similar_words = find_closest_embeddings(embeddings_dict, embeddings_dict[word])[0:remainder]

        print(similar_words)
        for s_word in similar_words:
            output[:, word_count] = continuous_to_discrete(embeddings_dict[s_word])
            word_count += 1
            used_words.append(s_word)
            print(spatial.distance.euclidean(embeddings_dict[word], embeddings_dict[s_word]))
        cycle_count += 1
    
    return output, used_words
        

def get_noise(
    noise_var: int,
    population_sizes: np.ndarray,
    times: np.ndarray,
    num_populations: int,
) -> np.ndarray:
    """Computes noise for all time iterations."""

    std = noise_var / (population_sizes ** 0.5)
    return std * np.random.normal(size=(len(times), num_populations))


# Connectivity
patterns, words = make_patterns(NUM_MEMORIES)
populations, population_sizes = sim.get_populations_and_sizes(patterns)
num_populations = population_sizes.shape[0]

connectivity_reg, connectivity_back, connectivity_forth = sim.get_connectivities(
    populations, NUM_MEMORIES
)

print(connectivity_reg.shape)

populations_sized = populations * population_sizes[:, None]
memories_similarities = populations_sized.T @ populations

# Dynamics
time = sim.prepare_times(T_TOT, T_STEP)
sparsity_vect = np.mean(patterns, axis=0) # np.full(NUM_MEMORIES, np.mean(patterns))
initial_memory = np.random.choice(range(0, NUM_MEMORIES))
oscillation = np.vectorize(
    sim.oscillation_closure(sim.oscillation, SIN_MIN, SIN_MAX, NUM_NEURONS)
)(time)

currents = np.zeros((num_populations, len(time)), dtype=np.float16)
firing_rates = np.zeros((num_populations, len(time)), dtype=np.float16)
n_currents = connectivity_reg[:, initial_memory].astype(np.float16)

noise = get_noise(NOISE_VAR, population_sizes, time, num_populations)  # long


# Simulation
for n_iter, t_cycle in tqdm(enumerate(time)):

    activations = sim.gain(n_currents.copy(), GAIN_EXP)
    sized_activations = population_sizes * activations
    total_activation = np.sum(sized_activations)

    # (NUM_MEMORIES,) = (num_populations,) @ (num_populations, NUM_MEMORIES)
    memory_activations = sized_activations @ connectivity_reg
    # (num_populations) = (num_populations, NUM_MEMORIES) @ (NUM_MEMORIES,)
    # - (NUM_MEMORIES,) @ (NUM_MEMORIES,) - (num_populations, NUM_MEMORIES)
    # @ (NUM_MEMORIES,) * () + (NUM_MEMORIES,) @ (NUM_MEMORIES,) * ()
    connectivity_term = (
        connectivity_reg @ memory_activations
        - sparsity_vect @ memory_activations
        - connectivity_reg @ sparsity_vect * total_activation
        + sparsity_vect @ sparsity_vect * total_activation
    )

    # (num_populations,) = () * (num_populations, NUM_MEMORIES) @ (NUM_MEMORIES,)
    # + () * (num_populations, NUM_MEMORIES) @ (NUM_MEMORIES,)
    contiguity_term = (
        CONT_FORTH * connectivity_forth @ memory_activations
        + CONT_BACK * connectivity_back @ memory_activations
    )

    updated_currents = (
        T_STEP
        / T_DECAY
        * (
            -n_currents
            + EXCITATION / NUM_NEURONS * (connectivity_term + contiguity_term)
            - oscillation[n_iter] * total_activation
            + noise[n_iter] / np.sqrt(T_STEP)
        )
    )

    n_currents += updated_currents
    firing_rates[:, n_iter] = activations.copy()
    currents[:, n_iter] = n_currents.copy()


# Save simulation results
file_name = f"s{JOB_ID}-jf{int(CONT_FORTH * NUM_NEURONS)}-n{int(NOISE_VAR)}"
populations_sized = (connectivity_reg * population_sizes[:, None]).T

memories_similarities = populations_sized @ connectivity_reg
firing_rates_memories = sim.get_dynamics_memories(
    firing_rates, population_sizes, connectivity_reg, memories_similarities
)

np.save(os.path.join(PATTERNS_DIR, file_name), memories_similarities)
np.save(
    os.path.join(RECALLS_DIR, file_name),
    sim.get_recall_sequence(firing_rates_memories, RECALL_THRESHOLD),
)


# Transform data and plot detailed dynamics only on selected seed
if JOB_ID == 11:

    # Dynamics
    print("Preparing to plot dynamics")
    currents_memories = sim.get_dynamics_memories(
        currents, population_sizes, connectivity_reg, memories_similarities
    )
    currents_populations = (population_sizes * currents.T).T
    print("Done!")

    plots.plot_firing_rates_attractors(firing_rates_memories, T_STEP, 20, 0, words)
    plots.plot_lines(
        firing_rates_memories,
        T_STEP,
        15,
        1,
        "Firing Rates",
        "Average firing rate",
        "firing_rates_lines.pdf",
    )

    plots.plot_lines(
        currents_populations,
        T_STEP,
        20,
        0,
        "Population Current",
        "Average current",
        "currents_populations.pdf",
    )
    plots.plot_lines(
        currents_memories,
        T_STEP,
        20,
        1,
        "Memory Current",
        "Average current",
        "currents_memories.pdf",
    )

    # Oscillation
    plots.plot_lines(
        oscillation[None, :], T_STEP, 15, 0, "Oscillation", "$\phi$", "oscillation.pdf"
    )
    plots.plot_lines(
        oscillation[None, :] * EXCITATION / NUM_NEURONS,
        T_STEP,
        15,
        1,
        "Inhibition",
        "Inhibition",
        "inhibition.pdf",
    )

    # Weights
    weights_reg = sim.get_connectivity_term_new(
        connectivity_reg, EXCITATION, NUM_NEURONS, sparsity_vect
    )
    weights_back = sim.get_connectivity_term_new(
        connectivity_back, EXCITATION, NUM_NEURONS, sparsity_vect
    )
    weights_forth = sim.get_connectivity_term_new(
        connectivity_forth, EXCITATION, NUM_NEURONS, sparsity_vect
    )
    weigths_without_inhibition = weights_reg + weights_back + weights_forth

    plots.plot_weights(
        weigths_without_inhibition,
        0,
        "Weights Without Inhibition",
        "weights_without_inhibition.pdf",
    )
    plots.plot_weights(weights_reg, 1, "Regular Weights", "weights_reg.pdf")
    plots.plot_weights(weights_back, 2, "Backward Weights", "weights_back.pdf")
    plots.plot_weights(weights_forth, 3, "Forward Weights ", "weights_forth.pdf")

    # Noise
    #plots.plot_lines(
    #    noise.T / T_STEP ** 0.5, T_STEP, 15, 0, "Noise", "Noise", "noise.png"
    #)
