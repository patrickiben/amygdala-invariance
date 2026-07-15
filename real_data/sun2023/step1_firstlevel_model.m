% First level modelling
firstlevelmasking=1;
 movementparameters=1;
    % high pass filter
        hpf = 128;
        % YOU WILL NEED TO CHANGE THIS SECTION
     
        % cnames={100 70 60 50 40 30 0};
        cnames={'Face' 'Fix'};% onset file with two regressors
        con_params=[2 0];% when there is no modulator, set 0, con_params=[] 
        name_params={'fear' 'ambiguity'};% name of two modulators

        ncond=length(cnames);
        
datapath='F:\01_FaceMorph\Scientific data\upload\fMRI_processed\';
subjects={'sub1' 'sub2' 'sub3' 'sub4' 'sub5' 'sub6' 'sub7' 'sub8' 'sub9' 'sub10' 'sub11' 'sub12' 'sub13' 'sub14' 'sub15' 'sub16' 'sub17' 'sub18' 'sub19'};
nsub=length(subjects);
stats_suffix='face_emotion_para3';     %create a folder to save the output files

 
for i=1:nsub  
    i

        %get subject directory
       
        %number of blocks
        nsess = 1;
        modeldur = 0;
        

        %-----------------------------------------------------------------------
        %Design setup
        %-----------------------------------------------------------------------

        % basis functions and timing parameters
        %---------------------------------------------------------------------------
        % OPTIONS:'hrf'
        %         'hrf (with time derivative)'
        %         'hrf (with time and dispersion derivatives)'
        %         'Fourier set'
        %         'Fourier set (Hanning)'
        %         'Gamma functions'
        %         'Finite Impulse Response'
        %---------------------------------------------------------------------------

        clear SPM

        % Set up basis functions
        
        SPM.xBF.T          = 16; % number of time bins per scan
        SPM.xBF.UNITS      = 'secs';           % OPTIONS: 'scans'|'secs' for onsets
        SPM.xBF.Volterra   = 1;                 % OPTIONS: 1|2 = order of convolution
        SPM.xBF.name       = 'hrf';
        SPM.xBF.length     = 32;              % length in seconds
        SPM.xBF.order      = 1;                 % order of basis set
        SPM.xY.RT = 2;
        SPM.xGX.iGXcalc = 'None';
        SPM.xVi.form = 'AR(1)';

        % Adjust time bin T0 according to reference slice & slice order
        %  implements email to CBU from Rik Henson 27/06/07
        %  assumes timings are relative to beginning of scans
%         refwhen=(find(aap.spmanalysis.sliceorder==aap.spmanalysis.refslice)-1)/(length(aap.spmanalysis.sliceorder)-1);
%         SPM.xBF.T0 = round(SPM.xBF.T*refwhen);        
% 
% 
%         subdata = aas_getsubjpath(aap,i);
        
        % New option to allow suffix to output file in extraparameters
       sessdata=fullfile(datapath, subjects{i});             
        anadir = fullfile(sessdata, stats_suffix);
        if exist(anadir)~=7; mkdir(anadir);end
        cd(anadir);


        tc = 0;
        allfiles='';
        files=[];

        cols_nuisance=[];
        cols_interest=[];
        currcol=1;
        
        movefilt = '^rp_.*\.txt$';
        mnames = {'x_trans' 'y_trans' 'z_trans' 'x_rot' 'y_rot' 'z_rot'};
        
        for sess = 1:nsess
          %  sessdata=aas_getsesspath(aap,i,sess);
            tc = tc+1;
            files = dir(fullfile(sessdata,'sw*.img'));
            for m=1:length(files)
                allfiles = strvcat(allfiles,fullfile(sessdata, files(m).name));

            end 
            
            if movementparameters
                mfname = spm_select ('List', sessdata, movefilt);
                moves = load(fullfile(sessdata,mfname));
            end
            SPM.nscan(tc) = size(files,1);

           
            for c = 1:ncond
                % YOU MAY NEED TO CHANGE THIS
                % It currently expects onset files
                %  onsets_[conditionname].txt (e.g., onsets_cue.txt) for event onsets (in scans)
                %  and durations_[conditionname].txt for event durations (in scans)
                % These should be in the directory corresponding to the
                % session they refer to.
                
                onset=[];            
                efile = fullfile(datapath, 'onsets', [subjects{i} cnames{c} '.txt']); % read files from onsets folder
                onset = load(efile);
                ons=[]; durs=[];
                ons=onset(:,1);
%               ons=(ons-2000*4)/1000; % the first 4 images have been
%               removed during preprocessing

                durs = 0;  % from start to end
                ons2=onset(:,2:end); % parameters values (fear, ambiguity levels)

                SPM.Sess(tc).U(c) = struct(...
                    'ons',ons,...
                    'dur',durs,...
                    'name',{cnames(c)},...
                    'P',struct('name','none'));
                
                cols_interest=[cols_interest currcol];
                currcol=currcol+1;
                
                 %-------------------------parametric modulators---------------

                for p=1:con_params(c)

                    eval(sprintf('SPM.Sess(%d).U(%d).P(%d).name = name_params{%d};',tc,c,p,p));	% with parametric modulation
                    eval(sprintf('SPM.Sess(%d).U(%d).P(%d).P = ons2(:,%d);',tc,c,p,p)); % parameters values (fear, ambiguity levels)
                    eval(sprintf('SPM.Sess(%d).U(%d).P(%d).h = 1;',tc,c,p));

                end
                
                if con_params(c)==0
                    eval(sprintf('SPM.Sess(%d).U(%d).P(1).name = ''none'';',tc,c));     % no parametric modulation
                end

                %---------------------end of parametricpart----------------------
                
            end

            SPM.xX.K(tc).HParam = hpf;


            if movementparameters==1
                SPM.Sess(tc).C.C    = moves;     % [n x c double] covariates
                SPM.Sess(tc).C.name = mnames; % [1 x c cell]   names
                cols_nuisance=[cols_nuisance [currcol:(currcol+5)]];
                currcol=currcol+6;
            else
                SPM.Sess(tc).C.C = [];
                SPM.Sess(tc).C.name = {};
            end
        end


        cons=[];
     cd(anadir);

        SPM.xY.P = allfiles;
        
            
        SPMdes = spm_fmri_spm_ui(SPM);

        % now check real covariates and nuisance variables are
        % specified correctly
        SPMdes.xX.iG=cols_nuisance;
        SPMdes.xX.iC=cols_interest;
        
        % Turn off masking if requested
        if (~firstlevelmasking)
            SPMdes.xM.I=0;
            SPMdes.xM.TH=-inf(size(SPMdes.xM.TH));
        end;

        
        spm_unlink(fullfile('.', 'mask.img')); % avoid overwrite dialog
       
           
             SPMest = spm_spm(SPMdes);

    
  
end;



