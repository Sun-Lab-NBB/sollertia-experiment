% setupAcquisition.m
%
% Top-level MATLAB script that prepares the ScanImage-controlled 2P-RAM Mesoscope for a Sollertia
% data-acquisition runtime. It configures the online motion-estimation reference, acquires a
% high-definition reference z-stack, and arms the acquisition loop that the sollertia-experiment
% VRPC drives via the kinase.bin and phosphatase.bin marker files.
%
% This script is designed to work with the MariusMotionEstimator and MariusMotionCorrector2
% ScanImage motion-correction classes, which are expected to be available in the ScanImage
% installation on the ScanImagePC.
%
% This file is deployed to the ScanImagePC and registered with MATLAB; see the accompanying
% README.md for setup and usage details.

function setupAcquisition(hSI, hSICtl, arguments)
    % SETUPACQUISITION Prepares the Mesoscope system for acquiring experiment data on the Sollertia platform.
    %
    %   This is a heavily refactored 'setupZstackALL' function used in the original manuscript. The function was 
    %   refactored to work with the Sollertia platform's data acquisition infrastructure. The function should be
    %   used to prepare the system for each data acquisition runtime.
    %
    %   Example function call (using default parameters): setupAcquisition(hSI, hSICtl)
    %
    % Arguments:
    % - hSI: The ScanImage handle object.
    % - hSICtl: The ScanImage Controller.
    % - zum: The z-spacing step, in micrometers, to use for z-stack generation.
    % - zrange: The range of z-planes to image. When imaging a single plane, set to plane number, e.g. 1050. When
    % imaging a slice, set to a numeric array that stores the slice boundaries in the order [min, max], e.g. [20, 400].
    % - zmirror: Only for two-plane imaging. When imaging two planes separated by a non-imaged volume, e.g. when imaging
    % only at the very top and bottom of the fastZ range, provide a numeric array of exclusion zone boundaries in the 
    % order [min, max]. Note, both boundaries must be within the overall slice dimensions defined by the zrange, e.g.
    % [40, 300]. The acquisition in this example case would be set to image data in the regions 30-40 and 300-400.
    % - order: The order to use for z-stack acquisition. Supported options are '' (default) and 'smooth'. Default
    % acquisition order iterates over the slices at each volume acquisition, e.g.: Z1, Z2, Z1, Z2. The 'smooth' order 
    % instead acquires all frames at the given z-plane before moving to the next one, e.g. Z1, Z1, Z2, Z2.
    % - channel: The channel to use for motion registration. Since high-definition zstack is primarily intended to
    % support advanced post-hoc motion analysis, it also uses the same channel.
    % - curvcorrection: Determines whether to enable or disable Field Curvature Correction support. Whether to enable
    % this feature depends on the acquisition system (microscope).
    % - naverage: The number of frames to acquire and average for each frame. Larger number of frames results in better
    % motion characterization at the expense of longer processing times and higher draw on acquisition machine
    % resources.
    % - root: The path at which to output the generated MotionEstimator.me and zstack.tiff files. On the Sollertia
    % platform, this is always set to the 'shared' mesoscope data folder.
    % - scalefactor: The factor by which to scale the X and Y resolution of all ROIs during the acquisition of the
    % high-definition zstack.tiff file. The scaling maintains the initial ROI aspect ratios.
    % - recovery: Determines whether the function is called to recover a failed runtime. In rare circumstances, the
    % ScanimagePC or the ScanImage software may fail during an experiment runtime. In this case, the VRPC tries to
    % execute a recovery sequence, which requires the Mesoscope to be re-armed to receive kinase and phosphatase
    % triggers. If the function is called with this argument set to true, it will skip all runtime preparation steps, 
    % load the existing MotionEstimator.me file from the mesoscope_data folder and arm the Mesoscope for receiving the
    % acquisition trigger.

    % Limited argument validation and default value assignment support. May not
    % work on older MatLab versions, but good for R2022b+.
    arguments
        hSI  % Cannot be validated due to how MBF implemented the class.
        hSICtl  % Cannot be validated due to how MBF implemented the class.
        arguments.zum (1,1) double {mustBePositive, mustBeFinite, mustBeInteger} = 20
        arguments.zrange double {mustBePositive, mustBeInteger} = 1050
        arguments.zmirror double {mustBePositive, mustBeInteger} = []
        arguments.order {mustBeMember(arguments.order, {'', 'smooth'})} = ''
        arguments.channel (1,1) double {mustBePositive, mustBeInteger} = 1
        arguments.curvcorrection (1,1) logical = false
        arguments.naverage (1,1) double {mustBePositive, mustBeInteger} = 20
        arguments.root (1,:) char {mustBeNonempty} = 'F:\mesodata\mesoscope_data'
        arguments.scalefactor (1,:) double {mustBePositive} = 2.0
        arguments.recovery (1,1) logical = false
    end
    
    zum = arguments.zum;
    zrange = arguments.zrange;
    zmirror = arguments.zmirror;
    order = arguments.order;
    channel = arguments.channel;
    curvcorrection = arguments.curvcorrection;
    naverage = arguments.naverage;
    root = arguments.root;
    scalefactor = arguments.scalefactor;
    recovery = arguments.recovery;

    % Clears the CLI
    clc;

    % Statically resolves the paths to marker files used to externally trigger and stop acquisition.
    kinase = fullfile(root, "kinase.bin");
    phosphatase = fullfile(root, "phosphatase.bin");
    
    % Converts single zrange values to [min, max] format expected by the rest of the function.
    if isscalar(zrange)
        fprintf('Configuring single plane imaging at z = %d µm.\n', zrange);
        zrange = [zrange, zrange];
    else
        fprintf('Configuring Z-stack imaging from %d to %d µm.\n', zrange(1), zrange(2));
    end
    
    % Validates zmirror has correct format if provided
    if ~isempty(zmirror)
        if numel(zmirror) ~= 2
            error('zmirror must be empty or contain exactly 2 values.');
        end
        zmirror = zmirror(:)';  % Ensures row vector
    end
    
    % Instructs the user to verify important imaging parameters before generating the reference stack.
    fprintf('Ensure that the following configuration parameters are applied:\n')
    fprintf('a) Laser is enabled and power is set.\n')
    fprintf('b) ROI frame rate is ~10 Hz.\n')
    fprintf('c) Scan phase is ~0.8888.\n')
    fprintf('d) PMTs AutoOn is enabled.\n')
    input('Enter anything to continue: ');
    
    %% Parameter Definition
    
    % Calculates reference half-width, maxing out at 12 reference planes on either end of the 
    % imaged plane range. This determines how many sub-planes are acquired zum microns above and below
    % each of the target imaging plane(s) to support Z-drift correction.
    nzhalf = min(floor((zum-1)/2),12);
    
    % Generates z-plane imaging positions.
    if isempty(zmirror)
        % If zmirror is not provided, creates a uniform set of 'zum'-spaced planes from minimum to maximum. 
        centerZs = zrange(1):zum:zrange(2);
    else
        % If zmirror is provided, creates two plane sequences expanding outward from each of the imaging 
        % focal points, given by zmirror. Assumes that both zmirror coordinates are within the range of planes
        % specified by zrange. This excludes the middle region (slices within zmirror) from processing. This mode of
        % stack definition is used exclusively with two-plane imaging modes.
        centerZs = [zmirror(1)-nzhalf:-zum:zrange(1) zmirror(2)+nzhalf:zum:zrange(2)];
    end
    
    % Sorts the center-points of each target plane resolved above and generates a set of 
    % planes above and below each imaging plane (reference Z-planes).
    centerZs = sort(centerZs);
    refZs = centerZs(:) + (-nzhalf:nzhalf);
    
    % 'Smooth' acquisition order acquires all frames for the target plane before moving to the
    % next one (Z1-Z1-Z1-Z2-Z2-Z2). Default acquisition order loops over planes (Z1-Z2-Z1-Z2-Z1-Z2)
    if strcmp(order, 'smooth')
        refZs = refZs';
    end
    
    % Depending on the configuration, ensures that FieldCurvatureCorrection is either enabled or disabled.
    % For Mesoscope, it should be enabled in most cases.
    if hSI.hFastZ.enableFieldCurveCorr ~= curvcorrection
        if curvcorrection
            fprintf('Field curvature correction: Enabled.\n');
        else
            fprintf('Field curvature correction: Disabled.\n');
        end
        hSI.hFastZ.enableFieldCurveCorr = curvcorrection;
    end

    % Only runs motion detection and high-definition zstack preparation steps if the runtime is not in recovery mode
    if ~recovery
        %% Reference ROI setup
        % If an acquisition is active, aborts it before changing system configuration.
        hSI.abort();  
        
        % Moves to the lowest plane to be imaged. Assumes that the fast-z is inverted, so smaller planes are
        % actually closest to the surface of the brain.
        hSI.hFastZ.hFastZs{1}.move(min(centerZs))
        
        % Grabs the reference volumes
        fprintf('Grabbing reference volume...\n');
        
        % Configures the acquisition to operate on the set of reference Z-planes and acquire the requested number of 
        % frames at each plane (20).
        % Configures the stack manager to target requested planes
        hSI.hStackManager.stackMode = 'fast';  % Enables fast-z (voice-coil)
        hSI.hStackManager.stackFastWaveformType = 'step';  % Step mode is required for volumetric averaging
        hSI.hStackManager.stackDefinition = "arbitrary";  % Stack has to be in arbitrary mode.
        hSI.hStackManager.arbitraryZs = sort(refZs(:));
        hSI.hStackManager.numVolumes = naverage;
        hSI.hStackManager.enable = true;
        
        % Buffers frames in frame averaging buffer.
        hSI.hDisplay.displayRollingAverageFactor = naverage;
        
        % Disables motion correction during z-stack acquisition and ensures MROI mode is active.
        hSI.hMotionManager.enable = false;
        hSI.hRoiManager.mroiEnable = true;
        hSI.hStackManager.stackReturnHome = true;
        
        % Ensures that the grabbed frames are discarded after runtime
        hSI.hChannels.loggingEnable = false;  % Disables data logging (saving)
        hSI.acqsPerLoop = 1;  % Ensures that the number of acquisitions is set to 1. This is a safety check.
        hSI.hChannels.channelDisplay = 1;  % Ensures channel 1 is displayed
        hSI.extTrigEnable = false;  % Ensures that external trigger mode is disabled.
        
        % Activates frame acquisition (starts grabbing)
        hSI.startGrab();
        while hSI.active
            pause(1); % waits for reference volume to be completed
        end
        
        fprintf('Reference volumes: Grabbed.\n');
        
        %% Motion Estimators generation
        fprintf('Setting up Motion Estimators...\n');
        
        hSI.hMotionManager.clearAndDeleteEstimators();  % Removes existing estimators.
        
        % Configures MotionEstimation plugin to use Marius scripts.
        hSI.hMotionManager.estimatorClassName = 'scanimage.components.motionEstimators.MariusMotionEstimator';
        hSI.hMotionManager.correctorClassName = 'scanimage.components.motionCorrectors.MariusMotionCorrector2';
        
        % Loads the ROI stack data from the ROI manager
        roiDatas = hSI.hDisplay.getRoiDataArray();
        
        % Filters the ROI data to only contain the motion registration channel data.
        arrayfun(@(rd)rd.onlyKeepChannels(channel),roiDatas);
        
        % First dimension is roi index, second dimension is volume index
        nRois = size(roiDatas,1);
        nVolumes = size(roiDatas,2);
        
        % Preallocates Z as cell array for efficiency, since the number of ROIs and planes is known.
        Z = cell(nRois, numel(centerZs));
        
        % Preallocates each Z cell by sampling ROI dimensions.
        for roiIdx = 1:nRois
            if ~isempty(roiDatas(roiIdx,1).imageData{1})
                sampleImg = roiDatas(roiIdx,1).imageData{1}{1};
                for iz = 1:numel(centerZs)
                    % Preallocates for expected number of reference planes.
                    Z{roiIdx, iz} = zeros(size(sampleImg,1), size(sampleImg,2), ...
                                          2*nzhalf+1, 'single');
                end
            end
        end
        
        fprintf('Aligning %d stacks...\n',nVolumes);
        
        % Loops over all ROIs
        for roiIdx = 1:nRois
        
        % Copies ROI data from the ROI manager
            roi0 = copy(roiDatas(roiIdx,:));
            
            % Precreates a template storage structure with correct dimensions, but no image data.
            for j = 1:naverage
                roi0(j).imageData = [];  % Clears image data, but keeps the metadata
            end
            
            % Finds reference planes for each target imaging position
            for iz = 1:numel(centerZs)
                id = find(ismember(refZs', centerZs(iz) + (-nzhalf:nzhalf)));
                
                % Extracts the data for relevant reference z-planes.
                roi1 = copy(roi0);
                for j = 1:naverage
                    roi1(j).imageData{1}  = roiDatas(roiIdx, j).imageData{1}(id);
                    roi1(j).zs  = roi1(j).zs(id);
                end
                
                % Aligns the requested number of frames (default is 20) for each plane to 
                % create a stable reference point and adds generated reference frame data to
                % the storage tensor.
                alignedRoiData = hSI.hMotionManager.alignZStack(roi1);
                img = alignedRoiData.imageData{1};
                for j = 1:numel(img)
                    Z{roiIdx, iz}(:,:,j) = img{j};
                end
                
                % Generates and adds the motion estimator for the target ROI to the Motion Detection
                % manager
                hSI.hMotionManager.addEstimator(alignedRoiData);
            end
            
        end
        
        % Saves the generated and applied motion estimators as MotionEstimator.me file
        fprintf('Generating MotionEstimator.me file...\n');
        hSI.hMotionManager.saveManagedEstimators(fullfile(root, 'MotionEstimator.me'));
        %%
        
        %% High-definition zstack acquisition
        fprintf('Acquiring a high-definition zstack...\n');
    
        % Disables motion correction during z-stack acquisition and ensures MROI mode is active.
        hSI.hMotionManager.enable = false;
        hSI.hRoiManager.mroiEnable = true;
        hSI.hStackManager.stackReturnHome = true;
        
        % Scales X and Y resolution of each ROI by the requested scale factor, generating a higher-definition zstack.
        for i = 1:numel(hSI.hRoiManager.currentRoiGroup.rois)
            roi = hSI.hRoiManager.currentRoiGroup.rois(i);
            sf = roi.scanfields(1);
        
            % Scales current pixel dimensions
            sf.pixelResolutionXY(1) = round(scalefactor * sf.pixelResolutionXY(1));
            sf.pixelResolutionXY(2) = round(scalefactor * sf.pixelResolutionXY(2));
        
            % Reassigns updated scan field
            roi.scanfields(1) = sf;
            hSI.hRoiManager.currentRoiGroup.rois(i) = roi;  % Updates the ROI in the manager
        end
        
        % Configures the acquisition to operate on the set of reference Z-planes and acquire the requested number of 
        % frames at each plane.
        hSI.hStackManager.arbitraryZs = sort(refZs(:));
        hSI.hStackManager.numVolumes = naverage;
        hSI.hStackManager.enable = true;
        
        % Buffers frames in frame averaging buffer.
        hSI.hDisplay.displayRollingAverageFactor = naverage;
        
        % Ensures that the grabbed frames are saved as 'zstack_0000.tiff' file.
        hSI.hScan2D.logOverwriteWarn = false; % Disables overwrite warning
        hSI.hChannels.loggingEnable = true;  % Enables data logging (saving)
        hSI.hScan2D.logAverageFactor = 1;  % Saves every frame (no averaging in saved data)
        hSI.hScan2D.logFilePath = root;  % Configures the root output directory
        hSI.hScan2D.logFileStem = 'zstack';  % Saves the stack data as 'zstack'
        hSI.hScan2D.logFileCounter = 0;  % Resets the acquisition file counter
        hSI.hScan2D.logFramesPerFile = 500;  % Configures tiff stacks to store at most 500 frames.
        hSI.acqsPerLoop = 1;  % Ensures that the number of acquisitions is set to 1. This is a safety check.
        hSI.hChannels.channelDisplay = 1;  % Ensures channel 1 is displayed
        hSI.extTrigEnable = false;  % Ensures that external trigger mode is disabled.
        
        % Activates frame acquisition (starts grabbing)
        hSI.startGrab();
        while hSI.active
            pause(1); % Waits for the stack to be acquired
        end
        
        fprintf('High-definition zstack: Acquired.\n');

        % Renames the acquired zstack file from ScanImage's default naming to 'zstack.tiff'
        sourceFile = fullfile(root, 'zstack_00000_00001.tif');
        destFile = fullfile(root, 'zstack.tiff');
        if isfile(sourceFile)
            movefile(sourceFile, destFile);
            fprintf('Renamed zstack file to: zstack.tiff\n');
        else
            warning('Expected zstack file not found: %s', sourceFile);
        end
        %%
        
        %% Prepares the system for acquisition
        fprintf('Preparing system for acquisition...\n');
    
        % Loops through each ROI and rescales it back to the original dimensions (from high-definition ones)
        for i = 1:numel(hSI.hRoiManager.currentRoiGroup.rois)
            roi = hSI.hRoiManager.currentRoiGroup.rois(i);
            sf = roi.scanfields(1);
        
            % Scales current pixel dimensions
            sf.pixelResolutionXY(1) = round(1/scalefactor * sf.pixelResolutionXY(1));
            sf.pixelResolutionXY(2) = round(1/scalefactor * sf.pixelResolutionXY(2));
    
            % Reassigns updated scan field
            roi.scanfields(1) = sf;
            hSI.hRoiManager.currentRoiGroup.rois(i) = roi;  % Updates ROI in the manager
        end
        
        % Saves the imaging field ROI to an .roi file before proceeding.
        fprintf('Generating a snapshot of the imaged ROIs...\n');
        hSI.hRoiManager.saveRoiGroupMroi(fullfile(root, 'fov.roi'))
    end
    
    % Ensures no acquisition is running before preparing for runtime
    hSI.abort();

    % Re-enables the Motion Detection plugin and shows it to user.
    hSI.hMotionManager.enable = true;
    hSICtl.showGUI('MotionDisplay');
    
    % For recovery runtimes, reloads motion estimation data from the root output folder.
    if recovery
        hSI.hMotionManager.clearAndDeleteEstimators();  % Removes existing estimators.
        
        % Configures MotionEstimation plugin to use Marius scripts.
        hSI.hMotionManager.estimatorClassName = 'scanimage.components.motionEstimators.MariusMotionEstimator';
        hSI.hMotionManager.correctorClassName = 'scanimage.components.motionCorrectors.MariusMotionCorrector2';
        
        % Loads motion estimation files currently saved in the root output folder.
        hSI.hMotionManager.loadEstimators(fullfile(root, 'MotionEstimator.me'));
    end
    
    % Configures the stack manager to target requested planes
    hSI.hStackManager.stackDefinition = 'arbitrary';  % Enables arbitrary stack traversal.
    hSI.hStackManager.stackMode = 'fast';  % Enables fast-z (voice-coil)
    hSI.hStackManager.stackFastWaveformType = 'step';  % Step mode is required for volumetric averaging
    hSI.hStackManager.arbitraryZs = centerZs(:);  % Configures z-stack manager to target the requested imaging planes.
    
    % Configures the acquisition settings
    hSI.hStackManager.enable = true;
    hSI.hStackManager.numVolumes = 100000;  % Prevents the runtime from stopping without triggers.
    hSI.hChannels.channelDisplay = 1;  % Ensures channel 1 is displayed
    hSI.acqsPerLoop = 1;  % A total of one acquisition per loop
    hSI.loopAcqInterval = 1;  % Although grab mode does not use intervals, it is minimized for added safety.
    
    % Presets frame averaging to 5 to give better picture at runtime.
    hSI.hDisplay.displayRollingAverageFactor = 5;
    
    % Ensures external trigger mode is disabled
    hSI.extTrigEnable = false;  % Ensures that external trigger mode is disabled.
    
    % Configures data output stream
    hSI.hScan2D.logFileStem = 'session';  % All data files will use the root name 'session'
    if ~recovery
        % Only resets the acquisition counter if the runtime is not in recovery mode
        hSI.hScan2D.logFileCounter = 0;  % Resets the acquisition file counter
    end
    hSI.hScan2D.logFramesPerFile = 500;  % Configures tiff stacks to store at most 500 frames.
    
    % Configures the motion estimation parameters
    tau = 100; % Increased from the default value of 50
    fs = hSI.hRoiManager.scanFrameRate;
    hSI.hMotionManager.hMotionCorrector.correctionThreshold = [.1 .1 .5];  % x, y, z
    hSI.hMotionManager.correctionDeviceZ  = 'fastz';  % Uses fastZ to correct z-drift
    hSI.hMotionManager.correctionEnableZ = true;  % Enables Z-drift correction
    hSI.hMotionManager.correctionDeviceXY = 'galvos';  % Uses galvos to correct X/Y drift
    hSI.hMotionManager.correctionEnableXY = true;  % Enables X/Y correction
    hSI.hMotionManager.hMotionCorrector.pC  = exp(-1/(tau*fs));
    hSI.hMotionManager.correctionBoundsZ    = [-100 100];  % Z-correction is performed within +- 100 um of target plane
    % Correction is only applied if the deviation has been above the correction threshold for this many seconds.
    hSI.hMotionManager.hMotionCorrector.thresholdExceedTime_s = 5;  
    % The interval at which to evaluate the need for correction (check the deviation of all managed axes).
    hSI.hMotionManager.hMotionCorrector.correctionInterval_s = 5;
    hSI.hMotionManager.resetCorrectionAfterAcq = 1; % Ensures that the correction offsets are reset between acquisitions
    
    % Arms the acquisition loop by starting to monitor the presence of the necessary marker files
    fprintf('Acquisition loop: Armed.\n');
    fprintf('Waiting for the kinase.bin marker to be created...\n');

    %% Runtime loop control
    % Waits for the kinase marker to appear.
    while ~isfile(kinase)

        % If phosphatase marker is created, aborts the runtime early.
        if isfile(phosphatase)
            fprintf('Phosphatase marker detected. Aborting.\n');
            
            % Resets the motion estimation and ROI windows to support the next acquisition
            hSI.hMotionManager.clearAndDeleteEstimators();
            hSI.hRoiManager.currentRoiGroup.clear
        
            % To optimize batch experiments, disables automatic correction. When new animals are placed under the 
            % mesoscope, the user usually carries out correction manually before enabling automated adjustments.
            hSI.hMotionManager.correctionEnableXY = false;
            hSI.hMotionManager.correctionEnableZ = false;

            return;
        end
        pause(1); % Avoid busy waiting, checks once per second.
    end

    % Activates frame acquisition (starts grabbing frames)
    fprintf('Kinase marker detected. Initializing frame acquisition...\n');
    hSI.startGrab();
    
    
    % This loop remains active as long as the kinase marker exists.
    while isfile(kinase)
        % This is not strictly necessary, but ensures that phosphatase marker can eliminate any ongoing 
        % Mesoscope acquisition at any point.
        if isfile(phosphatase)
            fprintf('Phosphatase marker detected. Aborting.\n');
            
            hSI.abort(); % Ends acquisition
    
            % Resets the motion estimation and ROI windows to support the next acquisition
            hSI.hMotionManager.clearAndDeleteEstimators();
            hSI.hRoiManager.currentRoiGroup.clear
        
            % To optimize batch experiments, disables automatic correction. When new animals are placed under the 
            % mesoscope, the user usually carries out correction manually before enabling automated adjustments.
            hSI.hMotionManager.correctionEnableXY = false;
            hSI.hMotionManager.correctionEnableZ = false;

            return;
        end
        pause(1); % Pause to reduce CPU load.
    end
    
    fprintf('Kinase marker removed. Terminating frame acquisition...\n');
    hSI.abort(); % Ends acquisition
    
    % Resets the motion estimation and ROI windows to support the next acquisition
    hSI.hMotionManager.clearAndDeleteEstimators();
    hSI.hRoiManager.currentRoiGroup.clear

    % To optimize batch experiments, disables automatic correction. When new animals are placed under the mesoscope, 
    % the user usually carries out correction manually before enabling automated adjustments.
    hSI.hMotionManager.correctionEnableXY = false;
    hSI.hMotionManager.correctionEnableZ = false;

    fprintf('Runtime: Complete.\n');


