import numpy as np
import matplotlib.pyplot as plt
from config import ensure_directory_exists, save_figure
import os


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
                   ALMRM=None, STRRM=None, ALMprojs=None, STRprojs=None, ALM_norm_factor_ctrl = None, STR_norm_factor_ctrl= None):
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
         flag = 1 # control

    # Variables to store normalization factors from first condition that reaches threshold
    ALM_norm_factor = np.zeros((numConditions, 4))
    STR_norm_factor = np.zeros((numConditions, 4))
    baseline_ALM = None
    baseline_STR = None
    LT = np.full((numConditions+1), np.nan) # LT as in the time when activity crosses the threshold

    
    for i in range(1, numConditions + 1):
        
        cueStart = 500 # 200
        cueIdx = 499 #199'
        
        if network_type in ['externally_driven', 'externally_driven_1input']:

            # make input across conditions not even so the ramp look more evenly spaced
            if i == 1:
                scaled_input = (inputVector * (10 - 1.5 * i) * np.linspace(1, timePoints + 1 - cueStart, timePoints + 1 - cueStart) / (timePoints + 1 - cueStart)).reshape(4, -1)
            elif i == 2:
                scaled_input = (inputVector * (10 - 1.5 * i) * np.linspace(1, timePoints + 1 - cueStart, timePoints + 1 - cueStart) / (timePoints + 1 - cueStart)).reshape(4, -1)
            elif i == 3:
                scaled_input = (inputVector * (10 - 1.4* i) * np.linspace(1, timePoints + 1 - cueStart, timePoints + 1 - cueStart) / (timePoints + 1 - cueStart)).reshape(4, -1)
            elif i == 4:
                scaled_input = (inputVector * (10 - 1.3 * i) * np.linspace(1, timePoints + 1 - cueStart, timePoints + 1 - cueStart) / (timePoints + 1 - cueStart)).reshape(4, -1)
            elif i == 5:
                scaled_input = (inputVector * (10 - 1.2 * i) * np.linspace(1, timePoints + 1 - cueStart, timePoints + 1 - cueStart) / (timePoints + 1 - cueStart)).reshape(4, -1)
            I[:, cueIdx:timePoints] = scaled_input
        else:

            # make input across conditions not even so the ramp look more evenly spaced
            if i == 1:
                I[:, cueIdx:timePoints] = np.tile((inputVector * (20 - 6 * i) / cueStart).reshape(4, 1), (1, timePoints + 1 - cueStart)) 
            elif i == 2:
                I[:, cueIdx:timePoints] = np.tile((inputVector * (20 - 4.5 * i) / cueStart).reshape(4, 1), (1, timePoints + 1 - cueStart)) 
            elif i == 3:
                I[:, cueIdx:timePoints] = np.tile((inputVector * (20 - 3.8 * i) / cueStart).reshape(4, 1), (1, timePoints + 1 - cueStart)) 
            elif i == 4:
                I[:, cueIdx:timePoints] = np.tile((inputVector * (20 - 3.2 * i) / cueStart).reshape(4, 1), (1, timePoints + 1 - cueStart)) 
            elif i == 5:
                I[:, cueIdx:timePoints] = np.tile((inputVector * (20 - 2.8 * i) / cueStart).reshape(4, 1), (1, timePoints + 1 - cueStart)) 

            Icue[:, 500:600] = np.tile(cueVector.reshape(4, 1), (1, 100))            


        r, I_out, h = iteration(W, h_init, I + Icue, Inh, timePoints, Ithresh, I_dc)
        
        
        # Plot the input I
        axarr3.plot(tAxis, I.T, label='Input')
        axarr3.set_xlabel('Time')
        axarr3.set_ylabel('Input Amplitude')
        
        trajectories.append(h.copy())

        # use neuron 3 reaching 10 Hz as a threshold
        # define target neuron based on network type
        if network_type == 'one_integrator_follower_opposite_leaky_1input':
            target_neuron_idx = 3  # 4th neuron 
        else:
            target_neuron_idx = 2  # 3rd neuron 
        
        target_activity = r[target_neuron_idx, :]
        target_threshold = 10.0  # Hz

        # find when target neuron crosses threshold
        threshold_crossings = np.where(target_activity >= target_threshold)[0]
        if len(threshold_crossings) > 0:
            threshold_time = tAxis[threshold_crossings[0]]
            # mark threshold crossing on plots
            axarr1[target_neuron_idx].axhline(y=target_threshold, color='r', linestyle='--')
            axarr1[target_neuron_idx].axvline(x=threshold_time, color='r', linestyle='--')
        
         # plot truncated activity until threshold crossing
            for idx in range(4):  # Plot activity
                axarr1[idx].plot(tAxis, r[idx, :], label=f'Condition {i}')
                axarr1[idx].set_title(f'Neuron {idx + 1} Activity')
                axarr1[idx].set_xlabel('Time from cue')
                axarr1[idx].set_ylabel('Activity (Hz)')
                trajectories.append(h.copy())

            if network_type in ['data'] and 'Control' in save_name:
                axarr1[target_neuron_idx].set_ylim(5, 11) 
                axarr1[1].set_ylim(5, 8.5)   
                axarr1[3].set_ylim(3.6, 5)   
                xlims = axarr1[1].get_xlim()

                for ax in axarr1:
                    ax.set_xlim(xlims)
        
        else:
            print(f"Warning: Target neuron {target_neuron_idx+1} did not reach threshold of {target_threshold} Hz")

        # calculate modes
        if i == 1:
            endpoint_activity = r[:, -1]
            pre_baseline_activity = r[:, cueIdx]
            ramp_mode = endpoint_activity - pre_baseline_activity
            if np.all(ALMRM == 0):
                ALMRM = ramp_mode[:2].reshape(2, 1) / np.linalg.norm(ramp_mode[:2])
            if np.all(STRRM == 0):
                STRRM = ramp_mode[2:].reshape(2, 1) / np.linalg.norm(ramp_mode[2:])

        # project onto ramp mode
        ALM_projection = np.dot(ALMRM.T, r[:2, :]).flatten()
        STR_projection = np.dot(STRRM.T, r[2:, :]).flatten()
        
        # get baseline activity (average of first 100 time points before cue)
        baseline_end = cueIdx
        baseline_start = min(cueIdx - 100, timePoints)
        current_ALM_baseline = np.mean(ALM_projection[baseline_start:baseline_end])
        current_STR_baseline = np.mean(STR_projection[baseline_start:baseline_end])
        
        # for first condition, store the baselines
        if i == 1:
            baseline_ALM = current_ALM_baseline
            baseline_STR = current_STR_baseline
        
        # subtract the first condition's baseline from all conditions   
        ALM_projection = ALM_projection - baseline_ALM
        STR_projection = STR_projection - baseline_STR

        if 'entire' in save_name:
            ALM_projection = ALM_projection - 5 #subtract baseline
            STR_projection = STR_projection - 5

        # normalize to reach 1 at threshold crossing if it occurs
        threshold_crossings = np.where(target_activity >= target_threshold)[0]
        if len(threshold_crossings) > 0:
            threshold_time = tAxis[threshold_crossings[0]]          
        if len(threshold_crossings) > 0 and (ALM_norm_factor_ctrl is None or STR_norm_factor_ctrl is None):
            threshold_idx = threshold_crossings[0]
              
            ALM_at_threshold = ALM_projection[threshold_idx]
            STR_at_threshold = STR_projection[threshold_idx]
            
            if ALM_at_threshold > 0:
                ALM_norm_factor = ALM_at_threshold
                ALM_norm_factor_ctrl = ALM_norm_factor
            if STR_at_threshold > 0:
                STR_norm_factor = STR_at_threshold
                STR_norm_factor_ctrl = STR_norm_factor

        else:
            ALM_norm_factor = ALM_norm_factor_ctrl
            STR_norm_factor = STR_norm_factor_ctrl
        
        # apply normalization if we have factors
        if ALM_norm_factor is not None and ALM_norm_factor > 0:
            ALM_projection = ALM_projection / ALM_norm_factor
        if STR_norm_factor is not None and STR_norm_factor > 0:
            STR_projection = STR_projection / STR_norm_factor

        # define lick time as when str ramp reaches 1
        threshold_crossings_ramp = np.where(STR_projection >= 1)[0]
        if len(threshold_crossings_ramp) > 0:
            LT[i] = tAxis[threshold_crossings_ramp[0]]  

        # clip projections between 0 and 1
        ALM_projection = np.minimum(ALM_projection, 1)#np.clip(ALM_projection, 0, 1)
        STR_projection = np.minimum(STR_projection, 1)#np.clip(STR_projection, 0, 1)
        
        if 'entire' not in save_name:
            # find the first index where projection hits 1 (normalized target)
            ALM_end_idx = np.argmax(ALM_projection >= 1) + 1 if np.any(ALM_projection >= 1) else len(ALM_projection)
            STR_end_idx = np.argmax(STR_projection >= 1) + 1 if np.any(STR_projection >= 1) else len(STR_projection)

            # plot truncated projections
            axarr2[0].plot(tAxis[:ALM_end_idx], ALM_projection[:ALM_end_idx], label=f'ALM Ramp Mode Condition {i}')
            axarr2[1].plot(tAxis[:STR_end_idx], STR_projection[:STR_end_idx], label=f'STR Ramp Mode Condition {i}')

        else: 
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
            # Clip control projection to end at value 1
            def clip_projection(projection, tAxis):
                threshold_idx = np.where(projection >= 1)[0]
                if len(threshold_idx) > 0:
                    idx = threshold_idx[0] + 1  # include the point at or just above 1
                    return tAxis[:idx], projection[:idx]
                else:
                    return tAxis, projection

            t_clip, ALMproj_clip = clip_projection(ALMprojs[i-1, :], tAxis)
            t_clip_str, STRproj_clip = clip_projection(STRprojs[i-1, :], tAxis)

            axarr2[0].plot(t_clip, ALMproj_clip, label=f'ALM Ramp Mode Control Condition {i}', linestyle=':')
            axarr2[1].plot(t_clip_str, STRproj_clip, label=f'STR Ramp Mode Control Condition {i}', linestyle=':')
           

    # # Set y-limits for subplots based on min and max values from projections
    if 'ALMfullsilencing' in save_name and 'entire' not in save_name:
        axarr2[0].set_ylim(-0.3, 1)  # for data
    #     axarr2[0].set_ylim(2, 6.5)  # for two region & two integrators
    #     axarr2[0].set_ylim(6.5, 8.8)  # for STR integrator ALM follower
    #     axarr2[0].set_ylim(1, 5)  # for ALM integrator STR follower
    #     axarr2[0].set_ylim(5, 12.8)  # for EXTERNAL
    #     axarr2[0].set_ylim(-0.1, 1)  # for EXTERNAL, 1 input
    #     axarr2[0].set_ylim(2, 7)  # for leaky data
    #     axarr2[1].set_ylim(np.min(Ylims_STR[:, 0]), np.max(Ylims_STR[:, 1]))       

    if 'entire' in save_name:
        minvalueALM = np.min(ALM_projection)
        minvalueSTR = np.min(STR_projection)
        axarr2[0].set_ylim(minvalueALM, 1) 
        axarr2[1].set_ylim(minvalueSTR, 1)  # for data
    
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
    
    return f1, f2, ALMRM, STRRM, r, I_out, trajectories, ALMprojs, STRprojs, tAxis, ALM_norm_factor_ctrl, STR_norm_factor_ctrl,LT


def run_silencing(W, h, I, I_dc, Inh, timePoints, inputVector, cueVector, Ithresh, Inh_start, Inh_dur, network_type):
    # Define the directory path
    #save_dir = 'C:/Users/yangzd/Inagaki Lab Dropbox/Zidan Yang/ED1code/data'
    save_dir = os.path.join('C:/Users/yangzd/Inagaki Lab Dropbox/Zidan Yang/ED1code',network_type)
    
    inputVector = inputVector.reshape(4, 1)
    results = {}
    f1, f2, ALMRM, STRRM, r_ctrl, I_ctrl, traj_ctrl, ALMprojs, STRprojs,tAxis, ALM_norm_factor_ctrl, STR_norm_factor_ctrl,LT_ctrl = run_simulation(W, h, I, Inh, timePoints, inputVector, cueVector, Ithresh,
                                                                     I_dc, network_type, 'Control_' + network_type, None, None, None, None,None, None)
    results['Control'] = {'r': r_ctrl, 'I': I_ctrl, 'trajectory': traj_ctrl}

    
    # save projection
    np.save(f'{save_dir}/Activity_ctrl.npy', r_ctrl)
    np.save(f'{save_dir}/tAxis.npy', tAxis)
    
    
    Inh_stop = Inh_start + Inh_dur
    
    ramp = 300 # ramp down for the silencing

    # ALM complete silencing
    s3 = -10 
    if network_type in ['two_region_integrator_1input']:
        s3 = -0.25
    Inh[:, Inh_start:Inh_stop] = np.tile(np.array([s3, s3, 0, 0]), (ramp, 1)).T
    for i in range(ramp):
        Inh[:, Inh_stop + i] = np.array([s3 + i * (-s3) / ramp, s3 + i * (-s3) / ramp, 0, 0])
    f1, f2, _, _, r_ALM_f, I_ALM_f, traj_af, ALMprojs_ALMs, STRprojs_ALMs,tAxis,_,_,LT_ALM = run_simulation(W, h, I, Inh, timePoints, inputVector, cueVector,
                                                       Ithresh, I_dc, network_type, 'ALMfullsilencing_' + network_type,
                                                       ALMRM, STRRM, ALMprojs, STRprojs,ALM_norm_factor_ctrl, STR_norm_factor_ctrl)
    results['ALMfullsilencing'] = {'r': r_ALM_f, 'I': I_ALM_f, 'trajectory': traj_af}

    # Save projection
    np.save(f'{save_dir}/Activity_ALMs.npy', r_ALM_f)
    np.save(f'{save_dir}/ALMprojs_ALMs.npy', ALMprojs_ALMs)
    np.save(f'{save_dir}/STRprojs_ALMs.npy', STRprojs_ALMs)


    # STR complete silencing one neuron
    s3 = -0.07 #-0.03
    if network_type in ['one_integrator_follower_opposite_1input']:
        s3 = -0.2
    elif network_type in ['externally_driven_1input']:
        s3 = -0.4 #-0.2
    elif network_type in ['one_integrator_follower_opposite_leaky_1input']:
        s3 = -0.2 #-0.02
    elif network_type in ['two_region_integrator_1input']:
        s3 = -0.3
    elif network_type in ['two_integrator_1input']:
        s3 = -0.1    

    Inh[:, Inh_start:Inh_stop] = np.tile(np.array([0, 0, s3, 0]), (ramp, 1)).T
    for i in range(ramp):
        Inh[:, Inh_stop + i] = np.array([0, 0, s3 + i * (-s3) / ramp, 0])
    f1, f2, _, _, r_STR_f, I_STR_f, traj_sf, ALMprojs_STRs, STRprojs_STRs,tAxis,_,_,LT_STR = run_simulation(W, h, I, Inh, timePoints, inputVector, cueVector,
                                                       Ithresh, I_dc, network_type,
                                                       'STRonesilencing' + network_type, ALMRM, STRRM, ALMprojs, STRprojs,ALM_norm_factor_ctrl, STR_norm_factor_ctrl)
    results['STRonesilencing'] = {'r': r_STR_f, 'I': I_STR_f, 'trajectory': traj_sf}
    
    # save projection
    np.save(f'{save_dir}/Activity_STRs.npy', r_STR_f)
    np.save(f'{save_dir}/ALMprojs_STRs.npy', ALMprojs_STRs)
    np.save(f'{save_dir}/STRprojs_STRs.npy', STRprojs_STRs)
       

    # save lick times across conditions
    LTs = {'LT_ctrl': LT_ctrl, 'LT_ALM': LT_ALM, 'LT_STR': LT_STR}
    np.save(f'{save_dir}/LickTimes.npy', LTs)
          

    # ALM complete silencing entire epoch
    s3 = -10 
    if network_type in ['two_region_integrator_1input']: # add 2025/6/16
        s3 = -10
    if network_type in ['data']:
        s3 = -10
    Inh[:, 0:timePoints] = np.tile(np.array([s3, s3, 0, 0]), (timePoints, 1)).T
  
    f1, f2, _, _, r_ALM_f, I_ALM_f, traj_af, _, _ ,tAxis,_,_,_ = run_simulation(W, h, I, Inh, timePoints, inputVector, cueVector,
                                                        Ithresh, I_dc, network_type, 'ALMfullsilencing_entireepoch' + network_type,
                                                        ALMRM, STRRM, ALMprojs, STRprojs,ALM_norm_factor_ctrl, STR_norm_factor_ctrl)
    results['ALMfullsilencing_entireepoch'] = {'r': r_ALM_f, 'I': I_ALM_f, 'trajectory': traj_af}
    


    return results
