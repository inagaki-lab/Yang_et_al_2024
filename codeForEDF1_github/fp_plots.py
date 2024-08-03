import numpy as np
from analysis import nullclines_and_fixed_points, plot_results


def perform_fixed_point_analysis(W, I_ctrl, r_ctrl, traj_ctrl, Inh_start, network_type, condition):
    
    fixed_indices_list = [     
        [1, 0],
    ]
    
    titles = [
        'x4,x3',
    ]
    
    t_fp2 = 499 #199 # cue
    if condition == 'Control':
        t_fp  = -1  # pick the alst time point for the control trajectories
        
    else:
        t_fp = Inh_start + 100  # pick 100 ms after the start of the inhibition for the other conditions

    for fixed_idx, title_suffix in zip(fixed_indices_list, titles):
        fixed_vals = np.take(r_ctrl[:, t_fp], fixed_idx)
        x1, x2, null_x1, null_x2, U, V, fixed_point = nullclines_and_fixed_points(
            W, I_ctrl[:, t_fp], fixed_indices=fixed_idx, fixed_values=fixed_vals
        )
        plot_results(x1, x2, null_x1, null_x2, U, V, fixed_point, traj_ctrl, network_type,
                     title_suffix=f"{condition}_{title_suffix}",
                     magnitude=np.sqrt(U ** 2 + V ** 2), fixed_indices=fixed_idx)

        fixed_vals = np.take(r_ctrl[:, t_fp2], fixed_idx)
        x1, x2, null_x1, null_x2, U, V, fixed_point = nullclines_and_fixed_points(
            W, I_ctrl[:, t_fp2], fixed_indices=fixed_idx, fixed_values=fixed_vals
        )
        plot_results(x1, x2, null_x1, null_x2, U, V, fixed_point, traj_ctrl, network_type,
                     title_suffix=f"cue_{condition}_{title_suffix}",
                     magnitude=np.sqrt(U ** 2 + V ** 2), fixed_indices=fixed_idx)