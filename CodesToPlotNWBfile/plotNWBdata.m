function [] = plotNWBdata()
% code to plot data in https://dandiarchive.org/dandiset/001610?pos=1
% modified the original code to plot NWB data @ https://neurodatawithoutborders.github.io/matnwb/tutorials/html/basicUsage.html
% modified by Hidehiko  11/22/2025 



% add matnwb to your path 
addpath('matnwb-2.4.0.0');

exampl_session = 'StrInhibition';

if strcmp(exampl_session,'ALMsilencing')
    nwb = nwbRead('sub-ZY78_ses-ZY78-20211014NP-ALM.nwb');
    plot_cell_id = 12;
elseif strcmp(exampl_session,'StrInhibition')
    nwb = nwbRead('sub-ZY116_ses-ZY116-20220815-ALM.nwb');
    plot_cell_id = 6;
end

%% read nwb file
unit_names = keys(nwb.analysis);

unit_ids = nwb.units.id.data.load(); % array of unit ids represented within this
% Initialize trials & times Map containers indexed by unit_ids
unit_trials = containers.Map('KeyType',class(unit_ids),'ValueType','any');
unit_times  = containers.Map('KeyType',class(unit_ids),'ValueType','any');
last_idx = 0;
for i = 1:length(unit_ids)
    unit_id = unit_ids(i);
    
    row = nwb.units.getRow(unit_id, 'useId', true, 'columns', {'spike_times', 'trialsID'});
    unit_trials(unit_id) = row.trialsID{1};
    unit_times(unit_id)  = row.spike_times{1};
end

sorted_ids = sort(unit_ids);
Photostim = struct(...
    'ind', true,... % mask into xs and ys for this photostim
    'name', 'none',...
    'stim_duration', 0,... % in seconds after the onset of normal go cue
    'stim_onset', 0); % in seconds after the onset of normal go cue



% Initialize Map container of plotting data for each unit, stored as structure
Unit = containers.Map('KeyType',class(unit_ids),'ValueType','any');
unit_struct = struct(...
    'id', [],...
    'xs', [],...
    'ys', [],...
    'xlim', [-Inf Inf],...
    'trialID',[],...
    'trialTypes', 0,...
    'spikeWidth',[],...
    'cellTypes',[],...
    'ontology',[],...
    'photostim', Photostim); % can have multiple photostim



% read data from indv units
for unit_id = unit_ids'
       
    unit_trial_id = unit_trials(unit_id);
    
    % extract good trials to find trial range
    trial = nwb.intervals_trials.getRow(unit_trial_id, 'useId', true,...
        'columns', {'CueOnset','GoodTrials'});

    unit_good_trials = logical(trial.GoodTrials) & ~isnan(trial.CueOnset);
    unit_trial_id    = unit_trial_id(unit_good_trials);
    
    unit_spike_time = unit_times(unit_id);
    unit_spike_time = unit_spike_time(unit_good_trials) - trial.CueOnset(unit_good_trials); % algin to cue
    

    % count number of trials per condition
    % we need to do this as there could be trial w.o. spikes
    first_trial = min(unit_trial_id); 
    last_trial  = max(unit_trial_id);
    trialIDs    = first_trial:last_trial;
    
    trial_in_range = nwb.intervals_trials.getRow(first_trial:last_trial, 'useId', true,...
        'columns', {'CueOnset', 'DelayDuration', 'FirstLick', 'GoodTrials',...
        'Unrewarded','Rewarded','NoLick','NoCue','StimTrials', 'PhotostimulationType'});
    
  
    % spike width
    SpikeWidth = nwb.general_extracellular_ephys_electrodes.getRow(1, 'useId', true,...
        'columns', {'spike_width'});
    % note SpikeWidth>0.5 is regular spiking cells, <0.35 is fast spiking
    % (FS) cells. ALM silencing increases FS activity while suppress RS
    % activity
    
    % putative cell type
    CellTypes = nwb.general_extracellular_ephys_electrodes.getRow(1, 'useId', true,...
        'columns', {'cell_type'});
    
    % ontology
    Ontology = nwb.general_extracellular_ephys_electrodes.getRow(1, 'useId', true,...
        'columns', {'ontology'});
    % for Str neuropixels recording we analyzed units in 'Caudoputamen','Striatum','Fundus of striatum'
    
    
    % summarize spike info for plotting
    
    xs = unit_spike_time; 
    ys = unit_trial_id;
    
    curr_unit = unit_struct;
    
    curr_unit.xs = xs;
    curr_unit.ys = ys;
    curr_unit.trialID    = trialIDs;
    curr_unit.trialTypes = trial_in_range;
    curr_unit.spikeWidth = SpikeWidth;
    curr_unit.cellTypes  = CellTypes;
    curr_unit.ontology   = Ontology;
    
    % ALM delay silencing   
    Stim = Photostim;
    Stim.name = 'Delay silencing';
    Stim.stim_duration = 0.6;
    Stim.stim_onset    = 0.6; % from cue (s)
    curr_unit.photostim = Stim;
    

    Unit(unit_id) = curr_unit;
      
end

 
%plot PSTH
plot_PSTH(Unit(plot_cell_id))

end



%% PSTH helper function


function plot_PSTH(Unit)


time_bin   = 0.001;
T_axis     =  -6:time_bin:6;
smooth_bin = 100;
v1 = ones(smooth_bin,1)/smooth_bin;
    
spk        = Unit.xs;
trials     = Unit.ys;
trialID    = Unit.trialID;
trialTypes = Unit.trialTypes;
numTrial   = numel(trialID);

StimInfo   = Unit.photostim;

PSTH        = nan(numTrial,numel(T_axis));
lick_time   = trialTypes{:,3};
stim_type   = trialTypes{:,10};


for i = 1:numTrial
    spk_mask     = trials==trialID(i);
    spk_in_trial = spk(spk_mask);
    
    spk_in_trial_aligned = spk_in_trial;
    counts               = hist(spk_in_trial_aligned,T_axis);
    mean_spike_rate      = counts/time_bin;    
    PSTH_tmp     = conv(mean_spike_rate,v1,'full');
    smoothedPSTH = PSTH_tmp(1:numel(mean_spike_rate)); % causal filtering
    PSTH(i,:)     = smoothedPSTH;               
end


% sort trial based on lick time
[lickSorted,lickSortedID] = sort(lick_time);
trial_sorted = nan(size(trials));
for i=1:numel(trialID)
    trial_tmp = trials== trialID(lickSortedID(i));
    trial_sorted(trial_tmp) = i;
end

% pool PSTH per lick time
LT_ranges    = [0.8 1.1;1.1 1.4;1.4 1.7;1.7 2.0;2.0 2.3;2.3 2.6];
num_LT_range = size(LT_ranges,1);

meanPSTH{1} = nan(num_LT_range,numel(T_axis));
meanPSTH{2} = nan(num_LT_range,numel(T_axis));

for lt = 1:num_LT_range
   for st = [0 2] 
        trMask = lick_time>=LT_ranges(lt,1) & lick_time<LT_ranges(lt,2) & stim_type==st;
        
        if sum(trMask)>=5 && st==0
            meanPSTH{1}(lt,:) = mean(PSTH(trMask,:)); 
        elseif sum(trMask)>=5 && st==2
            meanPSTH{2}(lt,:) = mean(PSTH(trMask,:)); 
        end
   end
end

lineColor = turbo(num_LT_range);


%% plot spike raster & PSTH
figure;set(gcf,'Color','w','Position',[110 182 450 700])
subplot(3,1,1);hold on
plot(spk,trial_sorted,'k.')
plot(lickSorted,1:numel(trialID),'m*')
xlim([-1 3])
ylim([0.5 numTrial+0.5])
xline(0,'k:')
set(gca,'tickdir','out','box','off')
xlabel('Time from cue (s)')
ylabel('Trials')
title('All trials')

subplot(3,1,2);hold on
for lt = 1:num_LT_range
    plot(T_axis,meanPSTH{1}(lt,:),'Color',lineColor(lt,:))
    xline(mean(LT_ranges(lt,:)),':','Color',lineColor(lt,:))
end
xlim([-1 3])
ylims = ylim();
yMax  = ylims(2)*1.1;
ylim([0 yMax]);
xline(0,'k:')
set(gca,'tickdir','out','box','off')
xlabel('Time from cue (s)')
ylabel('Spikes per s')
title('Non stim trials')

subplot(3,1,3);hold on
for lt = 1:num_LT_range
    plot(T_axis,meanPSTH{2}(lt,:),'Color',lineColor(lt,:))
    xline(mean(LT_ranges(lt,:)),':','Color',lineColor(lt,:))
end
xFill = [StimInfo.stim_onset StimInfo.stim_duration+StimInfo.stim_onset StimInfo.stim_duration+StimInfo.stim_onset StimInfo.stim_onset];
yFill = [ylims(2) ylims(2) yMax yMax];
fill(xFill,yFill,'c','edgecolor','none')
xlim([-1 3])
ylim([0 yMax]);

xline(0,'k:')
set(gca,'tickdir','out','box','off')
xlabel('Time from cue (s)')
ylabel('Spikes per s')
title('Delay stim trials')

end

















