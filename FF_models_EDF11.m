%% plot FF model results as in EDF11

clear;

network_type = {'FF_mix','twoFFnetwork','FF_ALMrecurrent','FF_STRrecurrent','FF_ALMlate','FF_STRlate','OneRegion'}

for n = 1:numel(network_type)
    plot_FF_model(network_type{n})
end


function plot_FF_model(network_type)

    close all

    % create folder to save
    saveFolder = 'FF_MultiRegionalModel';
    mkdir(saveFolder);


    timePoints = 5000; % num of time points

    % neuron 1, 2 ALM
    % neuron 3,4 STR
    % neuron 1 recieves ITI input and provide to STR
    % neuron 2 receives ramp and send it back to STR (but orthogonal to eigenvector with eigen value 1)

    if nargin()==0
        % select network type to plot
        network_type = 'OneRegion' ;
    end

    switch network_type

          case 'OneRegion'

            numN = 8;
            c = 0.92; f = 0.2;
            W1 = diag(c*ones(numN,1));
            W2 = diag(f*ones(numN-1,1),-1);

            W = W1+W2; 
            inputVector = zeros(numN,1);
            inputVector(1) = 100;

            ALM = [1:8];
            STR = [];
            isALMtrigger = 1;
            inhbtmp = -1.5;
            s2 = -5;

          case 'FF_mix'

            numN = 8;
            c = 0.86; f = 0.25;
            W1 = diag(c*ones(numN,1));
            W2 = diag(f*ones(numN-1,1),-1);

            W = W1+W2; 
            inputVector = zeros(numN,1);
            inputVector(1) = 100;

            ALM = [2,4,6,8];
            STR = [1,3,5,7];
            isALMtrigger = 1;
            inhbtmp = -0.7;
            s2 = -5;

          case 'twoFFnetwork'

            numN = 4;
            c = 0.9; f = 0.25;c2 = 0.01;
            W1 = diag(c*ones(numN,1));
            W2 = diag(f*ones(numN-1,1),-1);
            WSA = diag(c2*(1:numN)/numN); % str to ALM
            WSTR = W1+W2; 
            W = [WSTR WSA;WSA WSTR]; 

            inputVector = zeros(numN*2,1);
            inputVector(1) = 100;
            inputVector(5) = 100;
            ALM = 5:8;
            STR = 1:4; 
            isALMtrigger = 1;
            numN = numN*2;
            inhbtmp = -0.7;
            s2 = -5;

          case 'FF_ALMrecurrent'

            numN = 4;
            c = 0.81; f = 0.25;c2 = 0.35;c3=0.3;c4=0.35;
            W1 = diag(c*ones(numN,1));
            W2 = diag(f*ones(numN-1,1),-1);
            WSTR = W1+W2; 
            WSA = diag(c2*(1:numN)/numN); % str to ALM
            WALM = diag(c4*(1:numN)/numN);
            WAS = diag(c3*(1:numN)/numN);
            
            W = [WSTR WAS;WSA WALM]; 

            inputVector = zeros(numN*2,1);
            inputVector(1) = 100;
            ALM = 5:8;
            STR = 1:4; 
            isALMtrigger = 1;
            numN = numN*2;
            inhbtmp = -0.7;
            s2 =-2.5; 

         case 'FF_STRrecurrent'

            numN = 4;
            c = 0.81; f = 0.25;c2 = 0.35;c3=0.3;c4=0.35;
            W1 = diag(c*ones(numN,1));
            W2 = diag(f*ones(numN-1,1),-1);
            WSTR = W1+W2; 
            WSA = diag(c2*(1:numN)/numN); % str to ALM
            WALM = diag(c4*(1:numN)/numN);
            WAS = diag(c3*(1:numN)/numN);
            
            W = [WSTR WAS;WSA WALM]; 

            inputVector = zeros(numN*2,1);
            inputVector(1) = 100;
            ALM = 1:4;
            STR = 5:8; 
            isALMtrigger = 0;
            numN = numN*2;
            inhbtmp = -0.7;
            s2 = -100; 

    end




    s1 = -30; 

    saveFolder = fullfile('FF_MultiRegionalModel',network_type);
    mkdir(saveFolder);

    r1 = 0;
    r(:,1) =  r1*ones(numN,1); % baseline spike rate, initial activity
    Inh = zeros(numN,timePoints); % silencing


    if strcmp(network_type,'FF_ALMrecurrent')
        Ithresh = zeros(numN,1);
        Ithresh(STR) = inhbtmp;
        network_prop.maxSR = 50;
        
    elseif strcmp(network_type,'FF_STRrecurrent')
        Ithresh = zeros(numN,1);
        Ithresh(ALM) = inhbtmp;
        network_prop.maxSR = 50;
    elseif strcmp(network_type,'twoFFnetwork')
        Ithresh = ones(numN,1)*inhbtmp;
        network_prop.maxSR = 200;
    elseif strcmp(network_type,'OneRegion')
        Ithresh = ones(numN,1)*inhbtmp;
        network_prop.maxSR = 200;
    else
        Ithresh = ones(numN,1)*inhbtmp;
        network_prop.maxSR = 200;
    end

    [A,B,C] = eig(W)

    network_prop.W            =W;
    network_prop.r            =r;
    network_prop.Inh          =Inh;
    network_prop.timePoints   =timePoints;
    network_prop.numN         =numN;
    network_prop.inputVector  =inputVector;
    network_prop.Ithresh      =Ithresh;
    network_prop.network_type = network_type;
    network_prop.ALM          = ALM;
    network_prop.STR          = STR;
    network_prop.isALMtrigger = isALMtrigger;

    %% connectivity matrix
    f = figure;set(gcf,'Color','w','Position',[542 505 350 350]);
    imagesc(W)
    set(gca,'tickdir','out','fontsize',14)
    fileName = ['Matrix_',network_type];
    savefig(f,fullfile(saveFolder,fileName))
    print(f,'-dtiff' ,fullfile(saveFolder,fileName))
    print(f,'-deps' ,fullfile(saveFolder,fileName)) 

    W2 = W([ALM,STR],[ALM,STR]);
    f = figure;set(gcf,'Color','w','Position',[542 505 350 350]);
    imagesc(W2)
    set(gca,'tickdir','out','fontsize',14,'ytick',1:8,'xtick',1:8,...
        'xticklabel',{'A1','A2','A3','A4','S1','S2','S3','S4'},...
        'yticklabel',{'A1','A2','A3','A4','S1','S2','S3','S4'})
    fileName = ['Matrix_rearrnage_',network_type];
    savefig(f,fullfile(saveFolder,fileName))
    print(f,'-dtiff' ,fullfile(saveFolder,fileName))
    print(f,'-deps' ,fullfile(saveFolder,fileName)) 

    %% run simulation 
    [f1 f2 f3 corr_f param] = run_simulation(network_prop,[]);
    fileName = ['Control_',network_type];
    savefig(f1,fullfile(saveFolder,fileName))
    print(f1,'-dtiff' ,fullfile(saveFolder,fileName))
    print(f1,'-deps' ,fullfile(saveFolder,fileName))  
    fileName1 = [fileName,'_ALMmodes'];
    savefig(f2,fullfile(saveFolder,fileName1))
    print(f2,'-dtiff' ,fullfile(saveFolder,fileName1))
    print(f2,'-deps' ,fullfile(saveFolder,fileName1))  
    fileName2 = [fileName,'_STRmodes'];
    savefig(f3,fullfile(saveFolder,fileName2))
    print(f3,'-dtiff' ,fullfile(saveFolder,fileName2))
    print(f3,'-deps' ,fullfile(saveFolder,fileName2))  
    
    for i = 1:numel(corr_f)
        fileName4 = [fileName,'_corr_',num2str(i)];
        savefig(corr_f{i},fullfile(saveFolder,fileName4))
        print(corr_f{i},'-dtiff' ,fullfile(saveFolder,fileName4))
        print(corr_f{i},'-deps' ,fullfile(saveFolder,fileName4))  
    end
        


     % ALM complete silencings3 = -10;Inh = zeros(4,timePoints); % silencinginhtmp = zeros(numN,1);
    inhtmp = zeros(numN,1);
    inhtmp(ALM) = s1;
    Inh(:,1100:1400) = repmat(inhtmp,1,301);
    for i=1:300
        Inh(:,1400+i) = [inhtmp+i*(-inhtmp)/300]';
    end
    network_prop.Inh          =Inh;
    [f1 f2 f3 corr_f] = run_simulation(network_prop,param);
    fileName = ['ALMcompletesilencing_',network_type];
    savefig(f1,fullfile(saveFolder,fileName))
    print(f1,'-dtiff' ,fullfile(saveFolder,fileName))
    print(f1,'-deps' ,fullfile(saveFolder,fileName))  
    fileName1 = [fileName,'_ALMmodes'];
    savefig(f2,fullfile(saveFolder,fileName1))
    print(f2,'-dtiff' ,fullfile(saveFolder,fileName1))
    print(f2,'-deps' ,fullfile(saveFolder,fileName1))  
    fileName2 = [fileName,'_STRmodes'];
    savefig(f3,fullfile(saveFolder,fileName2))
    print(f3,'-dtiff' ,fullfile(saveFolder,fileName2))
    print(f3,'-deps' ,fullfile(saveFolder,fileName2))  
    
    for i = 1:numel(corr_f)
        fileName4 = [fileName,'_corr_',num2str(i)];
        savefig(corr_f{i},fullfile(saveFolder,fileName4))
        print(corr_f{i},'-dtiff' ,fullfile(saveFolder,fileName4))
        print(corr_f{i},'-deps' ,fullfile(saveFolder,fileName4))  
    end

    % STR  silencing of middl emodes

    inhtmp = zeros(numN,1);
    if numN>4
        idx = 2:3;
    else
        idx = 2;
    end
    inhtmp(STR(idx)) = s2;
    Inh(:,1100:1400) = repmat(inhtmp,1,301);
    for i=1:300
        Inh(:,1400+i) = [inhtmp+i*(-inhtmp)/300]';
    end

    network_prop.Inh          =Inh;
    [f1 f2 f3 corr_f] = run_simulation(network_prop,param);
    fileName = ['STRhalfsilencing_',network_type];
    savefig(f1,fullfile(saveFolder,fileName))
    print(f1,'-dtiff' ,fullfile(saveFolder,fileName))
    print(f1,'-deps' ,fullfile(saveFolder,fileName))  
    fileName1 = [fileName,'_ALMmodes'];
    savefig(f2,fullfile(saveFolder,fileName1))
    print(f2,'-dtiff' ,fullfile(saveFolder,fileName1))
    print(f2,'-deps' ,fullfile(saveFolder,fileName1))  
    fileName2 = [fileName,'_STRmodes'];
    savefig(f3,fullfile(saveFolder,fileName2))
    print(f3,'-dtiff' ,fullfile(saveFolder,fileName2))
    print(f3,'-deps' ,fullfile(saveFolder,fileName2))  
    
    for i = 1:numel(corr_f)
        fileName4 = [fileName,'_corr_',num2str(i)];
        savefig(corr_f{i},fullfile(saveFolder,fileName4))
        print(corr_f{i},'-dtiff' ,fullfile(saveFolder,fileName4))
        print(corr_f{i},'-deps' ,fullfile(saveFolder,fileName4))  
    end

end




function [f1 f2 f3 corr_f param] = run_simulation(network_prop,param)
%% run simulation and plot results

    if ~isempty(param)
        CM = param.CM;
        MM = param.MM;
        RM = param.RM;
        RMthr = param.RMthr;
        proj  = param.projOut;    
    end
   
    f1 = figure;set(gcf,'Color','w','Position',[151 5 350 990]);
    f2 = figure;set(gcf,'Color','w','Position',[151 5 350 990]);
    f3 = figure;set(gcf,'Color','w','Position',[151 5 350 990]);

    numN = size(network_prop.W,1);
    numConditions = 5;
    color_line = jet(numConditions);
    
    tAxis = ([1:network_prop.timePoints]-500)/1000;
    I     = zeros(size(network_prop.r)); % input

    for i = 1:numConditions
        
        inputTmp = (10-0.8*i)*1;
        tdiff = numel(tAxis)- 500+1;
        if strcmp(network_prop.network_type,'externally_driven')
            I(:,500:numel(tAxis))  = ((inputVector*inputTmp)*[1:tdiff]/tdiff);
        else
            I(:,500:numel(tAxis))  = repmat(network_prop.inputVector*inputTmp/200,1,tdiff);
        end
        
        [r,In] = iteration(network_prop,I);

        
        
        maxSR = network_prop.maxSR;
        if i==1 && isempty(param)
            
           if network_prop.isALMtrigger == 1 % is threhsold 
               last_neuron = network_prop.ALM(end);
           else
               last_neuron = network_prop.STR(end);
           end
           
           tIDl = find(r(last_neuron,:)>=maxSR,1);
           if isempty(tIDl);tIDl = numel(tAxis);end
               
           LT = tIDl;    
           tIDm = find(tAxis>=tAxis(tIDl-200)/2,1);tID2 = find(tAxis<0,1,'last');
           tCue = find(tAxis>0.15,1);
           
           %% modes in each region
           for region =1:2
           
               if region == 1
                   cellID = network_prop.ALM;
               else
                   cellID = network_prop.STR;
               end
               
               Cuemode    =  r(cellID,tCue) - r(cellID,tID2);if sum(Cuemode)>0;CMtmp = Cuemode/norm(Cuemode);else;CMtmp = Cuemode; end
               Middlemode =  r(cellID,tIDm) - r(cellID,tID2);if sum(Middlemode)>0;MMtmp = Middlemode/norm(Middlemode);else;MMtmp = Middlemode; end
               Rampmode   =  r(cellID,tIDl-200) - r(cellID,tID2);if sum(Rampmode)>0;RMtmp = Rampmode/norm(Rampmode);else;RMtmp = Rampmode; end

               MMtmp = MMtmp - (MMtmp'*RMtmp)*RMtmp;
               MMtmp = MMtmp/norm(MMtmp);
               if sum(MMtmp)==0
                   MMtmp = zeros(size(MMtmp));
               else
                   MMtmp = MMtmp/norm(MMtmp);
               end

               CMtmp = CMtmp - (CMtmp'*RMtmp)*RMtmp;
               CMtmp = CMtmp - (CMtmp'*MMtmp)*MMtmp;
               if sum(CMtmp)==0
                   CMtmp = zeros(size(CMtmp));
               else
                   CMtmp = CMtmp/norm(CMtmp);
               end


               RMthrtmp = RMtmp'*r(cellID,1:LT);
               RMthr(region) = RMthrtmp(end);

               
               if ~isempty(CMtmp)
                   CM(region,:) = CMtmp;
                   MM(region,:) = MMtmp;
                   RM(region,:) = RMtmp;
               end
           end

        else
            
           if network_prop.isALMtrigger == 1 % is threhsold 
               RMtmp = RM(1,:)*r(network_prop.ALM,:);
               LT = find(RMtmp>=RMthr(1),1);
           else
               RMtmp = RM(2,:)*r(network_prop.STR,:);
               LT = find(RMtmp>=RMthr(2),1);
           end
           
           if isempty(LT)
              LT =  numel(tAxis);
           end
            
        end
        
        %LT = find(r(numN,:)>=maxSR,1);
        tAxis_LT = tAxis(1:LT);
        %% fig1
        figure(f1);
        
        % input to neuron 1
        subplot(6,1,1);hold on
        plot(tAxis,max(I,[],1),'color',color_line(i,:),'linewidth',2)
        xline(0,'k:');xlabel('Time from cue');ylabel('Input (a.u.)');set(gca,'tickdir','out','fontsize',14)
        xlim([-0.4 2])
        
        % perturbation
        subplot(6,1,2);hold on
        plot(tAxis,1-mean(network_prop.Inh(network_prop.ALM,:),1),'b','linewidth',2)
        plot(tAxis,1-mean(network_prop.Inh(network_prop.STR,:),1),'m','linewidth',2)
        xline(0,'k:');xlabel('Time from cue');ylabel('Inhibition (a.u.)');set(gca,'tickdir','out','fontsize',14)
        legend({'ALM','STR'},'location','NorthWest')
        xlim([-0.4 2])

        % neuron 1
        subplot(6,1,3);hold on 
        plot(tAxis_LT,r(1,1:LT),'color',color_line(i,:),'linewidth',2)
        xline(0,'k:');xlabel('Time from cue');ylabel('Neuron 1 (Hz)');set(gca,'tickdir','out','fontsize',14)
        xlim([-0.4 2]);xline(0.6,'k:');xline(1.2,'k:');

        subplot(6,1,4);hold on 
        plot(tAxis_LT,r(2,1:LT),'color',color_line(i,:),'linewidth',2)
        xline(0,'k:');xlabel('Time from cue');ylabel('Neuron 2 (Hz)');set(gca,'tickdir','out','fontsize',14)
        xlim([-0.4 2]);xline(0.6,'k:');xline(1.2,'k:');

        subplot(6,1,5);hold on
        plot(tAxis_LT,r(3,1:LT),'color',color_line(i,:),'linewidth',2)
        xline(0,'k:');xlabel('Time from cue');ylabel('Neuron 3 (Hz)');set(gca,'tickdir','out','fontsize',14)
        xlim([-0.4 2]);xline(0.6,'k:');xline(1.2,'k:');

        subplot(6,1,6);hold on
        plot(tAxis_LT,r(8,1:LT),'color',color_line(i,:),'linewidth',2)
        xline(0,'k:');xlabel('Time from cue');ylabel('Neuron 4 (Hz)');set(gca,'tickdir','out','fontsize',14)
        xlim([-0.4 2]);xline(0.6,'k:');xline(1.2,'k:');
        
        
        
        %% mode
        
        for region =1:size(CM,1) % ALM, STR
        
            if region==1
                figure(f2);subplot(5,1,1);hold on;sgtitle('ALM')
                cellID = network_prop.ALM;
            else
                figure(f3);subplot(5,1,1);hold on;sgtitle('STR')
                cellID = network_prop.STR;
            end
        
            % input to neuron 1
            plot(tAxis,max(I,[],1),'color',color_line(i,:),'linewidth',2)
            xline(0,'k:');xlabel('Time from cue');ylabel('Input (a.u.)');set(gca,'tickdir','out','fontsize',14)
            xlim([-0.4 2.4])

            % perturbation
            subplot(5,1,2);hold on
            plot(tAxis,1-mean(network_prop.Inh(network_prop.ALM,:),1),'b','linewidth',2)
            plot(tAxis,1-mean(network_prop.Inh(network_prop.STR,:),1),'m','linewidth',2)
            xline(0,'k:');xlabel('Time from cue');ylabel('Inhibition (a.u.)');set(gca,'tickdir','out','fontsize',14)
            legend({'ALM','STR'},'location','NorthWest')
            xlim([-0.4 2.4])

            % projections
            if isempty(param)
                proj{region,i}(1,:) = CM(region,:)*r(cellID,1:LT);
                proj{region,i}(2,:) = MM(region,:)*r(cellID,1:LT);
                proj{region,i}(3,:) = RM(region,:)*r(cellID,1:LT);
            else
                % unperturbed trials
                subplot(5,1,3);hold on 
                plot(tAxis(1:numel(proj{region,i}(1,:))),proj{region,i}(1,:),':','color',color_line(i,:),'linewidth',2)
            
                subplot(5,1,4);hold on 
                plot(tAxis(1:numel(proj{region,i}(2,:))),proj{region,i}(2,:),':','color',color_line(i,:),'linewidth',2)
                
                subplot(5,1,5);hold on 
                plot(tAxis(1:numel(proj{region,i}(3,:))),proj{region,i}(3,:),':','color',color_line(i,:),'linewidth',2)
                
            end
            
            
            % CM
            subplot(5,1,3);hold on 
            plot(tAxis_LT,CM(region,:)*r(cellID,1:LT),'color',color_line(i,:),'linewidth',2)
            xline(0,'k:');xlabel('Time from cue');ylabel('CM');set(gca,'tickdir','out','fontsize',14)
            xlim([-0.4 2.4]);xline(0.6,'k:');xline(1.2,'k:');

            % MM
            subplot(5,1,4);hold on 
            plot(tAxis_LT,MM(region,:)*r(cellID,1:LT),'color',color_line(i,:),'linewidth',2)
            xline(0,'k:');xlabel('Time from cue');ylabel('MM');set(gca,'tickdir','out','fontsize',14)
            xlim([-0.4 2.4]);xline(0.6,'k:');xline(1.2,'k:');

            % RM
            subplot(5,1,5);hold on 
            plot(tAxis_LT,RM(region,:)*r(cellID,1:LT),'color',color_line(i,:),'linewidth',2)
            xline(0,'k:');xlabel('Time from cue');ylabel('RM');set(gca,'tickdir','out','fontsize',14)
            xlim([-0.4 2.4]);xline(0.6,'k:');xline(1.2,'k:');
        
        end
     

        if i==1 && isempty(param)
           r_1 = r(network_prop.ALM,:); LT_1 = LT;
        elseif ~isempty(param)
           r_1 = param.r_1; LT_1 = param.LT_1; 
        end
        corr_f{i} = figure;set(gcf,'Color','w','Position',[542 505 350 350]);
        imagesc(tAxis,tAxis,corr(r_1,r(network_prop.ALM,:)));colormap('jet')
        xline(0,'w:');xline(tAxis(LT),'w:')
        yline(0,'w:');yline(tAxis(LT_1),'w:')
        axis([0 2.5 0 tAxis(LT_1)+0.25]);caxis([0 1])
        set(gca,'tickdir','out','fontsize',14,'ydir','normal')
    end
    
    param.CM = CM;
    param.MM = MM;
    param.RM = RM;
    param.RMthr = RMthr;
    param.projOut = proj;
    param.r_1  = r_1;
    param.LT_1 = LT_1;
end




function [r,i] = iteration(network_prop,I)

W       = network_prop.W;
r       = network_prop.r;
Ithresh = network_prop.Ithresh;
Inh     = network_prop.Inh;
timePoints = network_prop.timePoints;


tau =0.025;dt = 0.001;
rmax = 10000; 
i = zeros(size(r));
h = zeros(size(r));


% repeat
for n=2:timePoints
   
    Itmp   = I(:,n) + Ithresh + Inh(:,n);
    h(:,n) = h(:,n-1)+ dt/tau*(-h(:,n-1)+W*r(:,n-1)+Itmp);
    i(:,n) = Itmp;
    
    rNew  = max(0,h(:,n));  
    r(:,n) = min(rmax,rNew);  

end

end




