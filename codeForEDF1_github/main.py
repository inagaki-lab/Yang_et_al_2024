import numpy as np
from config import h_init, I_adj, Inh, timePoints, network_type, I_dc, r
from network import configure_network
from simulation import run_silencing
from analysis import compute_correlation_matrix, plot_correlation_matrix
from fp_plots import perform_fixed_point_analysis

W, inputVector, cueVector, s1, s2 = configure_network(network_type)

Ithresh = -(-r + np.dot(W, r))
Inh_start = 1100 
Inh_dur = 300
# Run silencing
results = run_silencing(
    W, h_init, I_adj, I_dc, Inh, timePoints, inputVector, cueVector, Ithresh, Inh_start, Inh_dur, network_type)


# Save correlation matrices and vector space figs
for condition, data in results.items():
    correlation_matrix = compute_correlation_matrix(data['trajectory'][-1])
    plot_correlation_matrix(correlation_matrix, condition, network_type)
    perform_fixed_point_analysis(W, data['I'], data['r'], data['trajectory'], Inh_start, network_type, condition)
