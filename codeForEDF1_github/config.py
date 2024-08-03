import os
import numpy as np
from matplotlib import pyplot as plt


# Save folder
base_save_folder = 'C:/Users/yangzd/Inagaki Lab Dropbox/Zidan Yang/ED1code'
os.makedirs(base_save_folder, exist_ok=True)

network_type = 'data' # this and the conditions below correspond to EDF. 1a-f

# other network_type:
# 'externally_driven_1input'
# 'two_region_integrator_1input'
# 'two_integrator_1input'
# 'one_integrator_follower_opposite_1input'
# 'one_integrator_follower_opposite_leaky_1input'
# 'data'


# Parameters
timePoints = 2820 
h_init = np.array([5, 5, 5, 5])  # Baseline current

if network_type == 'data':
    h_init = np.array([0, 5, 5, 5])  # Baseline current redefined for data, make the input neuron has 0 baseline

r = np.maximum(0, h_init)  # Baseline spike rate

I_dc = 0 * np.ones(4)

I_adj = np.zeros((4, timePoints))
Inh = np.zeros((4, timePoints))


def ensure_directory_exists(network_type):
    """Ensure the folder exists for the specific network type."""
    network_folder = os.path.join(base_save_folder, network_type)
    if not os.path.exists(network_folder):
        os.makedirs(network_folder)
    return network_folder


def save_figure(fig, directory_path, filename):
    full_path = os.path.join(directory_path, filename)
    fig.savefig(full_path)
    plt.close(fig)
    print(f"Figure saved to {full_path}")
