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
    % This is a heavily refactored 'setupZstackALL' function used in the original manuscript. The function was
    % refactored to work with the Mesoscope-VR system implementation of the Sollertia platform's data acquisition
    % infrastructure. The function should be used to prepare the system for each data acquisition runtime.
    %
    % Example function call (using default parameters): runAcquisition(hSI, hSICtl)
    %
    % This is a lock-in function: it is launched ONCE and then runs a persistent command loop that holds the MATLAB
    % command line for the entire acquisition runtime. It does not return until it is interrupted with Ctrl-C or the
    % broker connection drops, so do not expect control to return to the command line between acquisition commands.
    %
    % After the broker connection is established, the function enters an MQTT command loop and acts as a state machine
    % that dispatches commands to the ScanImage software: it preloads the persisted reference estimator as an alignment
    % aid, generates the fresh session estimator and high-definition z-stack on request, and begins, aborts, or recovers
    % frame acquisition. The function reports command reception and progress on the 'MesoscopeStatus' topic. The
    % acquisition parameters are owned by the MesoscopeAcquisition section of the VRPC system configuration and are
    % delivered over MQTT with each command that consumes them; only the ScanImagePC-local output root and the broker
    % address remain function arguments.
    %
    % Arguments:
    % - hSI: The ScanImage handle object.
    % - hSICtl: The ScanImage Controller.
    % - root: The path to the ScanImagePC Mesoscope data root directory. The function resolves the shared
    % 'mesoscope_data' output folder for the generated MotionEstimator.me, zstack.tiff, and fov.roi files as the
    % 'mesoscope_data' subdirectory of this root. It also assumes the project/animal persistent data hierarchy lives
    % under the same root and uses that assumption to resolve the persisted per-animal reference estimator for the
    % preload command. This path is local to the ScanImagePC filesystem, so it remains a function argument rather than
    % part of the MQTT-delivered configuration.
    % - broker: The address of the MQTT broker shared with the Unity Virtual Reality task, in the 'tcp://host:port'
    % format. The broker runs on the VRPC, not on this ScanImagePC. The default targets the current VRPC at
    % 'tcp://192.168.0.13:1883', which matches the broker listener configured on the VRPC's local network interface.
    % Override this only if the broker host address or listener port changes.
    %
    % The remaining acquisition parameters - the z-step, z-range, exclusion zone, acquisition order, registration
    % channel, field curvature correction, reference frame averaging, and z-stack scale factor - are owned by the
    % MesoscopeAcquisition section of the VRPC system configuration. The VRPC delivers each command's required subset in
    % the command's MQTT payload: the reference-generation command carries the full set, while the recovery command
    % carries only the plane-geometry parameters needed to re-derive the imaging planes.

    % Limited argument validation and default value assignment support. May not
    % work on older MatLab versions, but good for R2022b+.
    arguments
        hSI  % Cannot be validated due to how MBF implemented the class.
        hSICtl  % Cannot be validated due to how MBF implemented the class.
        arguments.root (1,:) char {mustBeNonempty} = 'F:\mesodata'
        arguments.broker (1,:) char {mustBeNonempty} = 'tcp://192.168.0.13:1883'
    end

    root = arguments.root;
    broker = arguments.broker;

    % Resolves the shared Mesoscope data output directory under the data root. The generated reference files and the
    % session frames are written here, and the recovery command reloads the session estimator from here.
    dataRoot = fullfile(root, 'mesoscope_data');

    % Clears the CLI
    clc;

    % Statically defines the MQTT topics used to communicate with the VRPC. The VRPC publishes the command topics
    % (including the liveness probe and the state query) and subscribes to the status, error, and state reply topics;
    % this namespace does not overlap with the Unity task topics.
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

    %% MQTT command loop

    % Connects to the shared broker and subscribes to the VRPC command topics.
    mqttClient = mqttclient(broker);
    subscribe(mqttClient, topics.alive);
    subscribe(mqttClient, topics.preload);
    subscribe(mqttClient, topics.generate);
    subscribe(mqttClient, topics.begin);
    subscribe(mqttClient, topics.abort);
    subscribe(mqttClient, topics.recover);
    subscribe(mqttClient, topics.query);

    fprintf('Mesoscope control interface: Connected to %s.\n', broker);

    % Makes the lock-in behavior explicit to the operator. runAcquisition is a single, persistent command loop: it
    % is launched once and runs continuously, holding the MATLAB command line for the entire acquisition runtime, and
    % does not return until it is interrupted with Ctrl-C or the broker connection drops.
    fprintf('\n');
    fprintf('IMPORTANT: This is a persistent (lock-in) command loop.\n');
    fprintf('  - Launch it ONCE; it then services VRPC commands continuously for the whole acquisition runtime.\n');
    fprintf('  - It does NOT return and the MATLAB command line stays busy until the runtime ends.\n');
    fprintf('  - Press Ctrl-C in this window to stop the control loop and free the command line.\n');
    fprintf('\n');

    fprintf('Waiting for the VRPC to issue acquisition commands...\n');

    % Services commands until the client loses its broker connection. Each iteration yields to the event queue so the
    % operator retains full control of the ScanImage GUI to align the mesoscope between commands.
    while mqttClient.Connected
        % Drains and processes every message received since the previous iteration. read() returns the unread messages
        % as a timetable with Topic and Data variables, or an empty timetable when nothing arrived.
        messages = read(mqttClient);
        for index = 1 : height(messages)
            topic = string(messages.Topic(index));
            payload = string(messages.Data(index));

            try
                if topic == topics.alive
                    % Replies to the VRPC liveness probe. The reply itself confirms that the command loop is running;
                    % the VRPC treats the absence of a reply within its timeout as the control interface being offline.
                    publishStatus(mqttClient, topics.status, topic, "received");

                elseif topic == topics.preload
                    publishStatus(mqttClient, topics.status, topic, "received");
                    publishStatus(mqttClient, topics.status, topic, "preloading");
                    preloadEstimator(hSI, hSICtl, payload, root);
                    publishStatus(mqttClient, topics.status, topic, "preload_complete");

                elseif topic == topics.generate
                    publishStatus(mqttClient, topics.status, topic, "received");
                    config = buildAcquisitionConfig(payload, dataRoot, true);
                    applyFieldCurvature(hSI, config.curvcorrection);
                    generateReference(hSI, config, mqttClient, topics);
                    armMesoscope(hSI, hSICtl, config, true);
                    publishStatus(mqttClient, topics.status, topic, "armed");

                elseif topic == topics.recover
                    publishStatus(mqttClient, topics.status, topic, "received");
                    config = buildAcquisitionConfig(payload, dataRoot, false);
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


function config = buildAcquisitionConfig(payload, root, full)
    % BUILDACQUISITIONCONFIG Parses the acquisition parameters delivered with a command and resolves the
    % reference-plane geometry.
    %
    %   The geometry fields (root, nzhalf, centerZs, refZs) are always resolved, as both reference generation and
    %   recovery require them. The remaining acquisition fields (channel, naverage, scalefactor, curvcorrection) are
    %   resolved only for the full reference-generation payload. The 'root' argument is the shared Mesoscope data output
    %   directory (the 'mesoscope_data' subdirectory of the data root), where the session reference files are written
    %   and from which the recovery command reloads the session estimator.

    params = jsondecode(payload);
    [nzhalf, centerZs, refZs] = computeReferencePlanes( ...
        params.z_step_um, params.z_range_um, params.z_exclusion_um, params.acquisition_order);

    config = struct('root', root, 'nzhalf', nzhalf, 'centerZs', centerZs, 'refZs', refZs);

    if full
        config.channel = params.registration_channel;
        config.naverage = params.frames_per_reference_plane;
        config.scalefactor = params.zstack_scale_factor;
        config.curvcorrection = params.field_curvature_correction;
    end
end


function [nzhalf, centerZs, refZs] = computeReferencePlanes(zStep, zRange, zExclusion, order)
    % COMPUTEREFERENCEPLANES Resolves the target and reference imaging planes from the acquisition geometry.
    %
    %   Converts the imaging z-range and the optional two-plane exclusion zone into the sorted set of target imaging
    %   planes (centerZs) and the surrounding reference sub-planes (refZs) used to build the motion estimator. The
    %   z-range arrives as [minimum, maximum], where equal boundaries image a single plane. The exclusion zone arrives
    %   as [minimum, maximum], where equal boundaries disable two-plane imaging.

    zRange = double(zRange(:)');
    zExclusion = double(zExclusion(:)');

    if zRange(1) == zRange(2)
        fprintf('Configuring single plane imaging at z = %d µm.\n', zRange(1));
    else
        fprintf('Configuring Z-stack imaging from %d to %d µm.\n', zRange(1), zRange(2));
    end

    % Reference half-width, capped at 12 planes on either side of each target plane. Determines how many sub-planes are
    % acquired zStep micrometers above and below each target plane to support Z-drift correction.
    nzhalf = min(floor((zStep-1)/2), 12);

    % Resolves the target imaging planes. Equal exclusion boundaries disable two-plane imaging.
    if zExclusion(1) == zExclusion(2)
        % Uniform set of zStep-spaced planes spanning the imaged range (a single plane when the range collapses).
        centerZs = zRange(1):zStep:zRange(2);
    else
        % Two plane sequences expanding outward from each focal point, excluding the middle (two-plane imaging).
        centerZs = [zExclusion(1)-nzhalf:-zStep:zRange(1) zExclusion(2)+nzhalf:zStep:zRange(2)];
    end
    centerZs = sort(centerZs);

    % Reference sub-planes above and below each target plane.
    refZs = centerZs(:) + (-nzhalf:nzhalf);

    % 'smooth' order acquires all frames at a plane before advancing (Z1-Z1-Z2-Z2); the default interleaves planes.
    if strcmp(order, 'smooth')
        refZs = refZs';
    end
end


function applyFieldCurvature(hSI, enable)
    % APPLYFIELDCURVATURE Enables or disables ScanImage field curvature correction when the current setting differs.

    if hSI.hFastZ.enableFieldCurveCorr ~= enable
        if enable
            fprintf('Field curvature correction: Enabled.\n');
        else
            fprintf('Field curvature correction: Disabled.\n');
        end
        hSI.hFastZ.enableFieldCurveCorr = enable;
    end
end


function preloadEstimator(hSI, hSICtl, payload, root)
    % PRELOADESTIMATOR Loads the persisted reference estimator as an alignment aid with correction disabled.
    %
    %   The estimator restores the imaging field to approximately the same location across days. Automatic correction
    %   is left disabled so the operator can run manual enable/disable cycles while aligning the mesoscope. The VRPC
    %   sends only the project and animal identifiers, and this function resolves the persisted estimator path locally
    %   under the project/animal persistent data hierarchy assumed to live beneath the ScanImagePC data root.

    % Aborts any active acquisition before reconfiguring the motion manager.
    hSI.abort();

    % Resets the motion manager and configures it to use the Marius motion-correction classes. Both the existing
    % estimators and the existing ROI group are cleared before the persisted estimator is loaded, so the loaded
    % estimator's ROIs replace the current selection instead of being appended to it.
    hSI.hMotionManager.clearAndDeleteEstimators();
    hSI.hRoiManager.currentRoiGroup.clear
    hSI.hMotionManager.estimatorClassName = 'scanimage.components.motionEstimators.MariusMotionEstimator';
    hSI.hMotionManager.correctorClassName = 'scanimage.components.motionCorrectors.MariusMotionCorrector2';

    % Resolves the persisted estimator path from the project and animal identifiers. The 'persistent_data' folder name
    % mirrors the Sollertia platform per-animal persistent data directory. A missing file indicates that no persisted
    % estimator exists for the animal (for example, on the first imaging day), so alignment proceeds without an aid.
    decoded = jsondecode(payload);
    if isfield(decoded, 'project') && isfield(decoded, 'animal') ...
            && ~isempty(decoded.project) && ~isempty(decoded.animal)
        estimatorPath = fullfile( ...
            root, char(decoded.project), char(decoded.animal), 'persistent_data', 'MotionEstimator.me');
        if isfile(estimatorPath)
            hSI.hMotionManager.loadEstimators(estimatorPath);
            fprintf('Preloaded reference estimator: %s\n', estimatorPath);
        else
            fprintf('No persisted estimator found for this animal; proceeding without an alignment aid.\n');
        end
    else
        fprintf('No project/animal provided; proceeding without an alignment aid.\n');
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
