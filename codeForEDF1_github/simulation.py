import numpy as np
import matplotlib.pyplot as plt
from config import ensure_directory_exists, save_figure


def iteration(W, h_initial, I, Inh, timePoints, Ithresh, I_dc):
    tau = 0.010
    dt = 0.001
    rmax = 100
    h = np.zeros((4, timePoints))
    r = np.zeros((4, timePoints))
    I_out = np.zeros((4, timePoints))
    h[:, 0] = h_initial.squeeze()
    r[:, 0] = np.maximum(0, h[:, 0])

    for n in range(1, timePoints):
        I_tmp = I[:, n] + I_dc + Ithresh + Inh[:, n]
        h[:, n] = h[:, n - 1] + dt / tau * (-h[:, n - 1] + W @ r[:, n - 1] + I_tmp)
        I_out[:, n] = I_tmp

        # threshold-linear transfer function
        r[:, n] = np.maximum(0, h[:, n])
        r[:, n] = np.minimum(rmax, r[:, n])

    return r, I_out, h


def run_simulation(W, h_init, I, Inh, timePoints, inputVector, cueVector, Ithresh, I_dc, network_type, save_name='',
                   ALMRM=None, STRRM=None, ALMprojs=None, STRprojs=None):
    if ALMRM is None or np.all(ALMRM == 0):
        ALMRM = np.zeros((2, 1))
    if STRRM is None or np.all(STRRM == 0):
        STRRM = np.zeros((2, 1))
        
    cueStart = 500 # 200
    

    network_folder = ensure_directory_exists(network_type)
    
    f1, axarr1 = plt.subplots(4, 1, figsize=(4, 8))
    f2, axarr2 = plt.subplots(2, 1, figsize=(4, 8))
    f3, axarr3 = plt.subplots(1, 1, figsize=(4, 3)) # input
    
    tAxis = (np.arange(1, timePoints + 1) - cueStart) / 1000.0
    numConditions = 5
    Icue = np.zeros((4, timePoints))
    f1.suptitle(save_name, fontsize=16)
    f2.suptitle(save_name, fontsize=16)
    f3.suptitle('input', fontsize=16)
    trajectories = []
    
    flag = 0
    if ALMprojs is None:
         ALMprojs = np.zeros((numConditions, timePoints))
         STRprojs = np.zeros((numConditions, timePoints))
         flag = 1

    
    for i in range(1, numConditions + 1):
        
        cueStart = 500 # 200
        cueIdx = 499 #199'
        
        if network_type in ['externally_driven', 'externally_driven_1input']:
            scaled_input = (inputVector * (10 - 1.5 * i) * np.linspace(1, timePoints + 1 - cueStart, timePoints + 1 - cueStart) / (timePoints + 1 - cueStart)).reshape(4, -1)
            I[:, cueIdx:timePoints] = scaled_input
        else:
            I[:, cueIdx:timePoints] = np.tile((inputVector * (10 - 1.5 * i) / cueStart).reshape(4, 1), (1, timePoints + 1 - cueStart))
            Icue[:, 500:600] = np.tile(cueVector.reshape(4, 1), (1, 100))            


        r, I_out, h = iteration(W, h_init, I + Icue, Inh, timePoints, Ithresh, I_dc)
        
        
        # Plot the input I
        axarr3.plot(tAxis, I.T, label='Input')
        axarr3.set_xlabel('Time')
        axarr3.set_ylabel('Input Amplitude')
        
        for idx in range(4):  # Plot activity
            axarr1[idx].plot(tAxis, r[idx, :], label=f'Condition {i}')
            axarr1[idx].set_title(f'Neuron {idx + 1} Activity')
            axarr1[idx].set_xlabel('Time from cue')
            axarr1[idx].set_ylabel('Activity (Hz)')
        trajectories.append(h.copy())

        # Calculate modes
        if i == 1:
            endpoint_activity = r[:, -1]
            pre_baseline_activity = r[:, cueIdx]
            ramp_mode = endpoint_activity - pre_baseline_activity
            if np.all(ALMRM == 0):
                ALMRM = ramp_mode[:2].reshape(2, 1) / np.linalg.norm(ramp_mode[:2])
            if np.all(STRRM == 0):
                STRRM = ramp_mode[2:].reshape(2, 1) / np.linalg.norm(ramp_mode[2:])

        # Project onto ramp mode
        ALM_projection = np.dot(ALMRM.T, r[:2, :]).flatten()
        STR_projection = np.dot(STRRM.T, r[2:, :]).flatten()
        
        
        axarr2[0].plot(tAxis, ALM_projection, label=f'ALM Ramp Mode Condition {i}')
        axarr2[1].plot(tAxis, STR_projection, label=f'STR Ramp Mode Condition {i}')
        
        
        axarr2[0].axvline(x = 0, color = 'k', label = 'axvline - full height',linestyle = ':')
        axarr2[0].axvline(x = 0.6, color = 'k', label = 'axvline - full height',linestyle = ':')
        axarr2[0].axvline(x = 1.2, color = 'k', label = 'axvline - full height',linestyle = ':')
       
        axarr2[1].axvline(x = 0, color = 'k', label = 'axvline - full height',linestyle = ':')
        axarr2[1].axvline(x = 0.6, color = 'k', label = 'axvline - full height',linestyle = ':')
        axarr2[1].axvline(x = 1.2, color = 'k', label = 'axvline - full height',linestyle = ':')


        if flag == 1:   # control condition
            ALMprojs[i-1,:] = ALM_projection
            STRprojs[i-1,:] = STR_projection
            
        else:  # overlay control
            axarr2[0].plot(tAxis, ALMprojs[i-1,:], label=f'ALM Ramp Mode Control Condition {i}',linestyle = ':')
            axarr2[1].plot(tAxis, STRprojs[i-1,:], label=f'STR Ramp Mode Control Condition {i}',linestyle = ':')            
     
    
    # Set x-limits for subplots
    axarr2[0].set_xlim(min(tAxis), max(tAxis))
    axarr2[1].set_xlim(min(tAxis), max(tAxis))
    axarr3.set_xlim(min(tAxis), max(tAxis))

    axarr2[0].set_title('ALM Ramp Mode Projection')
    axarr2[0].set_xlabel('Time from cue')
    axarr2[0].set_ylabel('Projection Value')

    axarr2[1].set_title('STR Ramp Mode Projection')
    axarr2[1].set_xlabel('Time from cue')
    axarr2[1].set_ylabel('Projection Value')
    
    axarr2[0].spines['top'].set_visible(False)
    axarr2[0].spines['right'].set_visible(False)
    axarr2[1].spines['top'].set_visible(False)
    axarr2[1].spines['right'].set_visible(False)
    axarr3.spines['top'].set_visible(False)
    axarr3.spines['right'].set_visible(False)


    f1.tight_layout()
    f2.tight_layout()
    
    network_folder = ensure_directory_exists(network_type)
    
    save_figure(f1, network_folder, f"{save_name}_Activity.png")
    save_figure(f2, network_folder, f"{save_name}_Projections.png")
    save_figure(f3, network_folder, "input.png")
    
    save_figure(f1, network_folder, f"{save_name}_Activity.eps")
    save_figure(f2, network_folder, f"{save_name}_Projections.eps")
    save_figure(f3, network_folder, "input.eps")
    
    return f1, f2, ALMRM, STRRM, r, I_out, trajectories,ALMprojs,STRprojs,tAxis


def run_silencing(W, h, I, I_dc, Inh, timePoints, inputVector, cueVector, Ithresh, Inh_start, Inh_dur, network_type):
    # Define the directory path
    save_dir = 'C:/Users/yangzd/Inagaki Lab Dropbox/Zidan Yang/ED1code/data'
    
    inputVector = inputVector.reshape(4, 1)
    results = {}
    f1, f2, ALMRM, STRRM, r_ctrl, I_ctrl, traj_ctrl, ALMprojs, STRprojs,tAxis = run_simulation(W, h, I, Inh, timePoints, inputVector, cueVector, Ithresh,
                                                                     I_dc, network_type, 'Control_' + network_type, None, None, None, None)
    results['Control'] = {'r': r_ctrl, 'I': I_ctrl, 'trajectory': traj_ctrl}
    
    # save projection
    np.save(f'{save_dir}/Activity_ctrl.npy', r_ctrl)
    np.save(f'{save_dir}/tAxis.npy', tAxis)
    
    
    Inh_stop = Inh_start + Inh_dur
    
    ramp = 300

    # ALM complete silencing
    s3 = -10 
    Inh[:, Inh_start:Inh_stop] = np.tile(np.array([s3, s3, 0, 0]), (ramp, 1)).T
    for i in range(ramp):
        Inh[:, Inh_stop + i] = np.array([s3 + i * (-s3) / ramp, s3 + i * (-s3) / ramp, 0, 0])
    f1, f2, _, _, r_ALM_f, I_ALM_f, traj_af, ALMprojs_ALMs, STRprojs_ALMs,tAxis = run_simulation(W, h, I, Inh, timePoints, inputVector, cueVector,
                                                       Ithresh, I_dc, network_type, 'ALMfullsilencing_' + network_type,
                                                       ALMRM, STRRM, ALMprojs, STRprojs)
    results['ALMfullsilencing'] = {'r': r_ALM_f, 'I': I_ALM_f, 'trajectory': traj_af}

    # Save projection
    np.save(f'{save_dir}/Activity_ALMs.npy', r_ALM_f)
    np.save(f'{save_dir}/ALMprojs_ALMs.npy', ALMprojs_ALMs)
    np.save(f'{save_dir}/STRprojs_ALMs.npy', STRprojs_ALMs)


    # STR complete silencing one neuron
    s3 = -0.03
    if network_type in ['one_integrator_follower_opposite_1input']:
        s3 = -0.2
    elif network_type in ['externally_driven_1input']:
        s3 = -0.2
    elif network_type in ['one_integrator_follower_opposite_leaky_1input']:
        s3 = -0.02
    elif network_type in ['two_region_integrator_1input']:
        s3 = -0.1 
    elif network_type in ['two_integrator_1input']:
        s3 = -0.1    

    Inh[:, Inh_start:Inh_stop] = np.tile(np.array([0, 0, s3, 0]), (ramp, 1)).T
    for i in range(ramp):
        Inh[:, Inh_stop + i] = np.array([0, 0, s3 + i * (-s3) / ramp, 0])
    f1, f2, _, _, r_STR_f, I_STR_f, traj_sf, ALMprojs_STRs, STRprojs_STRs,tAxis = run_simulation(W, h, I, Inh, timePoints, inputVector, cueVector,
                                                       Ithresh, I_dc, network_type,
                                                       'STRonesilencing' + network_type, ALMRM, STRRM, ALMprojs, STRprojs)
    results['STRonesilencing'] = {'r': r_STR_f, 'I': I_STR_f, 'trajectory': traj_sf}
    
    # save projection
    np.save(f'{save_dir}/Activity_STRs.npy', r_STR_f)
    np.save(f'{save_dir}/ALMprojs_STRs.npy', ALMprojs_STRs)
    np.save(f'{save_dir}/STRprojs_STRs.npy', STRprojs_STRs)
       
    # ALM complete silencing entire epoch
    s3 = -10 
    Inh[:, 0:timePoints] = np.tile(np.array([s3, s3, 0, 0]), (timePoints, 1)).T
  
    f1, f2, _, _, r_ALM_f, I_ALM_f, traj_af, _, _ ,tAxis = run_simulation(W, h, I, Inh, timePoints, inputVector, cueVector,
                                                        Ithresh, I_dc, network_type, 'ALMfullsilencing_entireepoch' + network_type,
                                                        ALMRM, STRRM, ALMprojs, STRprojs)
    results['ALMfullsilencing_entireepoch'] = {'r': r_ALM_f, 'I': I_ALM_f, 'trajectory': traj_af}
    

    return results
