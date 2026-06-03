% runAcquisition.m
%
% Top-level MATLAB script that prepares the ScanImage-controlled 2P-RAM Mesoscope for the Sollertia
% platform Mesoscope-VR data-acquisition runtime. It configures the online motion-estimation
% reference, acquires a high-definition reference z-stack, and services the acquisition commands
% that the sollertia-experiment VRPC issues over MQTT.
%
% This script is designed to work with the MariusMotionEstimator and MariusMotionCorrector2
% ScanImage motion-correction classes, which are expected to be available in the ScanImage
% installation on the ScanImagePC.
%
% The script connects to the shared MQTT broker also used by the Unity Virtual Reality task and
% exchanges messages over a dedicated 'Mesoscope' topic namespace. It requires the MATLAB
% Industrial Communication Toolbox, which provides the mqttclient interface.
%
% This file is deployed to the ScanImagePC and registered with MATLAB; see the accompanying
% README.md for setup and usage details.

function runAcquisition(hSI, hSICtl, arguments)
    % RUNACQUISITION Prepares and runs a Mesoscope data-acquisition runtime within the Mesoscope-VR system.
    %
    %   This is a heavily refactored 'setupZstackALL' function used in the original manuscript. The function was
    %   refactored to work with the Mesoscope-VR system implementation of the Sollertia platform's data acquisition
    %   infrastructure. The function should be used to prepare the system for each data acquisition runtime.
    %
    %   Example function call (using default parameters): runAcquisition(hSI, hSICtl)
    %
    % After the configuration arguments are applied, the function connects to the MQTT broker and services the
    % acquisition commands published by the VRPC: it preloads the persisted reference estimator as an alignment aid,
    % generates the fresh session estimator and high-definition z-stack on request, and begins, aborts, or recovers
    % frame acquisition. The function reports command reception and progress on the 'MesoscopeStatus' topic. Unlike
    % the previous marker-file interface, the acquisition parameters are passed as function arguments and are not
    % exchanged over MQTT.
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
    % - broker: The address of the MQTT broker shared with the Unity Virtual Reality task, in the 'tcp://host:port'
    % format. The broker runs on the VRPC, not on this ScanImagePC. The default targets the current VRPC at
    % 'tcp://192.168.0.13:1883', which matches the broker listener configured on the VRPC's local network interface.
    % Override this only if the broker host address or listener port changes.

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
        arguments.broker (1,:) char {mustBeNonempty} = 'tcp://192.168.0.13:1883'
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
    broker = arguments.broker;

    % Clears the CLI
    clc;

    % Statically defines the MQTT topics used to communicate with the VRPC. The VRPC publishes the command topics
    % (including the state query) and subscribes to the status, error, and state reply topics; this namespace does not
    % overlap with the Unity task topics.
    topics = struct( ...
        'alive',    "MesoscopeAlive", ...
        'preload',  "MesoscopePreload", ...
        'generate', "MesoscopeGenerateReference", ...
        'begin',    "MesoscopeBeginAcquisition", ...
        'abort',    "MesoscopeAbort", ...
        'recover',  "MesoscopeRecover", ...
        'query',    "MesoscopeQueryState", ...
        'status',   "MesoscopeStatus", ...
        'error',    "MesoscopeError", ...
        'state',    "MesoscopeState");

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

    % Bundles the reference-plane configuration shared by the command handlers.
    config = struct( ...
        'zum', zum, 'naverage', naverage, 'channel', channel, 'scalefactor', scalefactor, ...
        'root', root, 'nzhalf', nzhalf, 'centerZs', centerZs, 'refZs', refZs);

    %% MQTT command loop

    % Connects to the shared broker and subscribes to the VRPC command topics.
    mqttClient = mqttclient(broker);
    subscribe(mqttClient, topics.preload);
    subscribe(mqttClient, topics.generate);
    subscribe(mqttClient, topics.begin);
    subscribe(mqttClient, topics.abort);
    subscribe(mqttClient, topics.recover);
    subscribe(mqttClient, topics.query);

    fprintf('Mesoscope control interface: Connected to %s.\n', broker);
    fprintf('Waiting for the VRPC to issue acquisition commands...\n');

    % Tracks whether the alive marker still needs to be republished and the timestamp of the last alive publication.
    announceAlive = true;
    aliveTimer = tic;

    % Services commands until the client loses its broker connection. Each iteration yields to the event queue so the
    % operator retains full control of the ScanImage GUI to align the mesoscope between commands.
    while mqttClient.Connected
        % Republishes the alive marker once per second until the first command arrives, closing the race where the
        % VRPC subscribes after the initial publication.
        if announceAlive && toc(aliveTimer) >= 1
            write(mqttClient, topics.alive, "");
            aliveTimer = tic;
        end

        % Drains and processes every message received since the previous iteration. read() returns the unread messages
        % as a timetable with Topic and Data variables, or an empty timetable when nothing arrived.
        messages = read(mqttClient);
        for index = 1 : height(messages)
            topic = string(messages.Topic(index));
            payload = string(messages.Data(index));
            announceAlive = false;

            try
                if topic == topics.preload
                    publishStatus(mqttClient, topics.status, topic, "received");
                    publishStatus(mqttClient, topics.status, topic, "preloading");
                    preloadEstimator(hSI, hSICtl, payload, config);
                    publishStatus(mqttClient, topics.status, topic, "preload_complete");

                elseif topic == topics.generate
                    publishStatus(mqttClient, topics.status, topic, "received");
                    generateReference(hSI, config, mqttClient, topics);
                    armMesoscope(hSI, hSICtl, config, true);
                    publishStatus(mqttClient, topics.status, topic, "armed");

                elseif topic == topics.recover
                    publishStatus(mqttClient, topics.status, topic, "received");
                    recoverAcquisition(hSI, hSICtl, config);
                    publishStatus(mqttClient, topics.status, topic, "armed");

                elseif topic == topics.begin
                    publishStatus(mqttClient, topics.status, topic, "received");
                    hSI.startGrab();
                    publishStatus(mqttClient, topics.status, topic, "grabbing");

                elseif topic == topics.abort
                    publishStatus(mqttClient, topics.status, topic, "received");
                    hSI.abort();
                    resetMotionState(hSI);
                    publishStatus(mqttClient, topics.status, topic, "stopped");

                elseif topic == topics.query
                    % Captures a one-shot system-state snapshot and publishes it to the VRPC as a named-field struct
                    % mirroring the queryAcquisitionState output, to populate a MesoscopePositions instance later.
                    vals = queryAcquisitionState(hSI);
                    write(mqttClient, topics.state, jsonencode(struct( ...
                        'x', vals(1), 'y', vals(2), 'r', vals(3), 'z', vals(4), ...
                        'fast_z', vals(5), 'tip', vals(6), 'tilt', vals(7), 'power_mW', vals(8))));
                end
            catch exception
                % Surfaces the failure to the VRPC and returns the mesoscope to a safe, idle state.
                write(mqttClient, topics.error, jsonencode(struct('message', exception.message)));
                hSI.abort();
                resetMotionState(hSI);
            end
        end

        pause(0.1);  % Yields to the event queue so the ScanImage GUI stays responsive during alignment.
    end

    fprintf('Mesoscope control interface: Disconnected. Runtime complete.\n');
end


function preloadEstimator(hSI, hSICtl, payload, ~)
    % PRELOADESTIMATOR Loads the persisted reference estimator as an alignment aid with correction disabled.
    %
    %   The estimator restores the imaging field to approximately the same location across days. Automatic correction
    %   is left disabled so the operator can run manual enable/disable cycles while aligning the mesoscope.

    % Aborts any active acquisition before reconfiguring the motion manager.
    hSI.abort();

    % Resets the motion manager and configures it to use the Marius motion-correction classes.
    hSI.hMotionManager.clearAndDeleteEstimators();
    hSI.hMotionManager.estimatorClassName = 'scanimage.components.motionEstimators.MariusMotionEstimator';
    hSI.hMotionManager.correctorClassName = 'scanimage.components.motionCorrectors.MariusMotionCorrector2';

    % Loads the persisted estimator when the VRPC dispatched a path. A null path indicates that no persisted estimator
    % exists for the animal (for example, on the first imaging day), so alignment proceeds without an aid.
    decoded = jsondecode(payload);
    if isfield(decoded, 'path') && ~isempty(decoded.path) && (ischar(decoded.path) || isstring(decoded.path))
        estimatorPath = char(decoded.path);
        if isfile(estimatorPath)
            hSI.hMotionManager.loadEstimators(estimatorPath);
            fprintf('Preloaded reference estimator: %s\n', estimatorPath);
        else
            warning('Preload estimator path does not exist: %s', estimatorPath);
        end
    else
        fprintf('No persisted estimator provided; proceeding without an alignment aid.\n');
    end

    % Shows the estimator so the operator can align the field, but leaves automatic correction disabled.
    hSI.hMotionManager.enable = true;
    hSICtl.showGUI('MotionDisplay');
    hSI.hMotionManager.correctionEnableXY = false;
    hSI.hMotionManager.correctionEnableZ = false;
end


function generateReference(hSI, config, mqttClient, topics)
    % GENERATEREFERENCE Grabs a fresh session estimator and a high-definition reference z-stack.
    %
    %   This is the lengthy preparation step. It acquires a reference volume, builds and saves the session
    %   MotionEstimator.me, then acquires and saves the high-definition zstack.tiff and the fov.roi snapshot.

    root = config.root;
    naverage = config.naverage;
    channel = config.channel;
    scalefactor = config.scalefactor;
    nzhalf = config.nzhalf;
    centerZs = config.centerZs;
    refZs = config.refZs;

    publishStatus(mqttClient, topics.status, topics.generate, "generating_estimator");

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

    %% High-definition zstack acquisition
    publishStatus(mqttClient, topics.status, topics.generate, "acquiring_zstack");
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

    %% Imaging field ROI snapshot
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


function recoverAcquisition(hSI, hSICtl, config)
    % RECOVERACQUISITION Reloads the session estimator and re-arms the mesoscope without regenerating the z-stack.
    %
    %   Used to resume an acquisition that started successfully and was then interrupted by a transient failure. The
    %   session estimator was already saved to the shared data directory during the reference generation step.

    % Ensures no acquisition is running before reloading the motion estimation data.
    hSI.abort();

    hSI.hMotionManager.clearAndDeleteEstimators();  % Removes existing estimators.

    % Configures MotionEstimation plugin to use Marius scripts.
    hSI.hMotionManager.estimatorClassName = 'scanimage.components.motionEstimators.MariusMotionEstimator';
    hSI.hMotionManager.correctorClassName = 'scanimage.components.motionCorrectors.MariusMotionCorrector2';

    % Loads motion estimation files currently saved in the root output folder.
    hSI.hMotionManager.loadEstimators(fullfile(config.root, 'MotionEstimator.me'));

    % Re-arms the mesoscope without resetting the acquisition file counter so the recovered runtime continues
    % numbering its output files where the interrupted runtime stopped.
    armMesoscope(hSI, hSICtl, config, false);
end


function armMesoscope(hSI, hSICtl, config, resetCounter)
    % ARMMESOSCOPE Configures the stack manager, acquisition, and motion-correction parameters and arms the mesoscope.
    %
    %   Shared by the reference-generation and recovery paths. The resetCounter flag controls whether the acquisition
    %   file counter is reset, which must happen for a fresh runtime but not when recovering an interrupted one.

    root = config.root;
    centerZs = config.centerZs;

    % Ensures no acquisition is running before preparing for runtime.
    hSI.abort();

    % Enables the Motion Detection plugin and shows it to the user.
    hSI.hMotionManager.enable = true;
    hSICtl.showGUI('MotionDisplay');

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
    if resetCounter
        % Only resets the acquisition counter for fresh runtimes, never when recovering an interrupted one.
        hSI.hScan2D.logFileCounter = 0;  % Resets the acquisition file counter
    end
    hSI.hScan2D.logFilePath = root;  % Configures the root output directory
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

    fprintf('Mesoscope: Armed.\n');
end


function resetMotionState(hSI)
    % RESETMOTIONSTATE Clears the motion estimators and ROI windows and disables automatic correction.
    %
    %   Returns the mesoscope to an idle state after an acquisition is stopped or aborted. Automatic correction is
    %   disabled because the operator re-enables it manually after positioning the next animal.

    hSI.hMotionManager.clearAndDeleteEstimators();
    hSI.hRoiManager.currentRoiGroup.clear

    % To optimize batch experiments, disables automatic correction. When new animals are placed under the mesoscope,
    % the user usually carries out correction manually before enabling automated adjustments.
    hSI.hMotionManager.correctionEnableXY = false;
    hSI.hMotionManager.correctionEnableZ = false;
end


function publishStatus(mqttClient, statusTopic, commandTopic, state)
    % PUBLISHSTATUS Publishes a command acknowledgement or progress update to the VRPC on the status topic.

    write(mqttClient, statusTopic, jsonencode(struct('command', char(commandTopic), 'state', char(state))));
end


function vals = queryAcquisitionState(hSI)
    % QUERYACQUISITIONSTATE Captures a one-shot snapshot of the Mesoscope stage, fast-Z, and laser state.
    %
    %   Records the system state at the boundaries of a runtime (before and optionally after acquisition) to populate
    %   a MesoscopePositions instance; it is not meant for repeated polling. Returns a 1x8 double row vector ordered as
    %   [x, y, r, z, fast_z, tip, tilt, power_mW], where power_mW is the CONFIGURED target power at the sample (valid
    %   while the laser is idle). tip and tilt are placeholders (always 0) reserved for hardware that may be added in a
    %   future update.

    % Caches the MCM6000 stage controller, the imaging beam, and the beam index between calls. Resolving these
    % resources from the ScanImage resource store is relatively expensive, so they are looked up once and reused.
    persistent mcm beam beamIdx

    % Resolves the MCM6000 stage controller if it has not been cached yet, or if the cached handle became invalid.
    if isempty(mcm) || ~isvalid(mcm)
        mcm = hSI.hResourceStore.filterByName('MCM6000');
        if iscell(mcm), mcm = mcm{1}; end
        assert(~isempty(mcm) && isvalid(mcm), ...
            'queryAcquisitionState:noMCM', ...
            'MCM6000 not found in hSI.hResourceStore.');
    end

    % Resolves the imaging beam and its index within the Beams component if it has not been cached yet.
    if isempty(beam) || ~isvalid(beam)
        beam = hSI.hResourceStore.filterByName('Thor Axon 920');
        if iscell(beam), beam = beam{1}; end
        assert(~isempty(beam) && isvalid(beam), ...
            'queryAcquisitionState:noBeam', ...
            'Beam ''Thor Axon 920'' not found in hSI.hResourceStore.');

        % Resolves the index of this beam within the Beams component, which parallels the hBeams.powers array.
        beamIdx = find(cellfun(@(b) b == beam, hSI.hBeams.hBeams), 1);
        assert(~isempty(beamIdx), 'queryAcquisitionState:noBeamIdx', ...
            'Beam not found in hSI.hBeams.hBeams.');
    end

    % Reads the last known MCM6000 stage position, indexing the raw position vector by each motor's hardware slot.
    raw = mcm.lastKnownPosition;
    x = raw(mcm.xMotorSlot);
    y = raw(mcm.yMotorSlot);
    r = raw(mcm.rMotorSlot);
    z = raw(mcm.zMotorSlot);

    % Reads the current fast-Z (voice-coil) position.
    fast_z = hSI.hFastZ.position;

    % Reports tip and tilt as fixed placeholders. The corresponding hardware is not installed, so these are always 0.
    % They are reserved in the output contract and may be populated with real values in a future update.
    tip  = 0;
    tilt = 0;

    % Computes the configured target laser power at the sample, in milliwatts. The power-fraction setpoint is converted
    % to watts through the beam's calibration lookup table, then scaled to milliwatts. This value persists while idle.
    frac     = hSI.hBeams.powers(beamIdx) / 100;
    lut      = beam.powerFraction2PowerWattLut;
    watts    = interp1(lut(:,1), lut(:,2), frac, 'linear', 'extrap');
    power_mW = watts * 1000;

    % Packs the queried state into the 1x8 output row vector.
    vals = [x, y, r, z, fast_z, tip, tilt, power_mW];
end
