# Mesoscope-VR ScanImage PC Assets

Provides MATLAB assets used to acquire experiment data with the ScanImage-controlled 2-Photon Random Access
Mesoscope (2P-RAM).

![MATLAB](https://img.shields.io/badge/MATLAB-orange?logo=mathworks&logoColor=white&labelColor=grey)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../../LICENSE)

___

## Detailed Description

This directory stores the MATLAB assets that the Mesoscope-VR acquisition system deploys to the **ScanImagePC** (the
dedicated computer that runs the ScanImage software and controls the Mesoscope). The Python `sollertia_experiment`
package does not carry these assets, which are deployed to the ScanImagePC and registered with MATLAB, as described
below.

Currently, this directory provides a single asset, the `runAcquisition` function. This function automates the
preparation for data acquisition and allows the sollertia-experiment library to bidirectionally interface with the
ScanImage software during runtime. It carries out three runtime-critical tasks: setting up the online motion estimation
reference, acquiring a high-definition reference z-stack, and servicing the acquisition commands that the main
acquisition system PC (VRPC) issues over MQTT. The function connects to the shared MQTT broker, preloads the persisted
reference estimator as an alignment aid, generates the session reference data on request, and begins, aborts, or
recovers frame acquisition in response to the VRPC commands. It exchanges messages over a dedicated `Mesoscope` topic
namespace that does not overlap with the namespace used by the Unity Virtual Reality task sharing the same broker.

The `runAcquisition` function is designed to work with the `MariusMotionEstimator` and `MariusMotionCorrector2`
motion-correction classes. These classes are provided as part of the ScanImage installation on the ScanImagePC and are
assumed to be available, so this repository does not distribute them.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

___

## Dependencies

These dependencies apply to the **ScanImagePC** and are typically satisfied by the vendor that assembles the machine
(MBF Bioscience and ThorLabs):

- [MATLAB](https://www.mathworks.com/products/matlab.html) version R2022b or above with
  [ScanImage](https://www.mbfbioscience.com/products/scanimage/) version 2023.1.0 (Premium).
- [Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html), required
  by the `MariusMotionEstimator` class for fast online motion detection and correction.
- [Industrial Communication Toolbox](https://www.mathworks.com/products/industrial-communication.html),
  required by `runAcquisition` to connect to the MQTT broker shared with the sollertia-experiment runtime.
- An [NVIDIA CUDA GPU](https://www.nvidia.com/en-us/), used to accelerate online motion detection and correction.

___

## Installation

To make the `runAcquisition` function available on the ScanImagePC:

1. Copy this `assets/mesoscope_vr` directory to the ScanImagePC, or check out the sollertia-experiment repository on
   that machine.
2. Open MATLAB and navigate to the **Command Window**.
3. Run `addpath("PATH_TO_ASSET_DIRECTORY")`, replacing `PATH_TO_ASSET_DIRECTORY` with the **absolute** path to the
   directory that contains `runAcquisition.m` on the local machine. To persist the path across MATLAB sessions, follow
   the [MATLAB search-path tutorials](https://www.mathworks.com/help/matlab/matlab_env/add-remove-or-reorder-folders-on-the-search-path.html).

If registration works as expected, the `runAcquisition` function is now available for calling from the Command Window.

***Note,*** the `MariusMotionEstimator` and `MariusMotionCorrector2` motion-correction classes must be present in the
ScanImage installation for `runAcquisition` to work. These classes ship with ScanImage and require no separate
registration.

___

## Usage

The Mesoscope-VR runtime prompts the experimenter to call `runAcquisition(hSI, hSICtl)` as part of the broader Mesoscope
preparation sequence, where `hSI` is the ScanImage handle object and `hSICtl` is the ScanImage controller. Beyond these
two handles, the function accepts only two optional name-value arguments. The first is `root`, the ScanImagePC-local
Mesoscope data root (default `F:\mesodata`), beneath which the function resolves the shared `mesoscope_data` output
folder and the per-animal persistent reference hierarchy. The second is `broker`, the MQTT broker address described
below. The VRPC system configuration owns all remaining acquisition parameters, such as the z-step, z-range, and
exclusion zone, and delivers them over MQTT with each command. Use `help runAcquisition` in the MATLAB Command Window
for the full argument documentation.

***Critical!*** `runAcquisition` is a **lock-in** function. It is launched **once** and then runs a persistent command
loop that services VRPC commands continuously and **holds the MATLAB command line for the entire acquisition runtime**.
The command line stays busy from launch until the runtime ends. To stop the control loop and free the command line,
press **Ctrl-C** in the MATLAB Command Window (the broker connection dropping also ends the loop). The function prints
this reminder when it starts.

***Critical!*** The MQTT broker runs on the **VRPC**, not on the ScanImagePC, so the connection is cross-machine. The
`broker` argument defaults to `tcp://192.168.0.13:1883`, which targets the current VRPC and matches the broker listener
configured on the VRPC's local network interface. Override it only if the broker host address or listener port changes,
by passing the new address through the `broker` argument, for example
`runAcquisition(hSI, hSICtl, broker="tcp://VRPC-IP:1883")`. The VRPC's broker must also be configured to accept
connections from the ScanImagePC over the local network.

In most cases, the function executes three major steps:

1. **Motion estimation setup.** The function configures the acquisition according to the user-defined parameters and
   establishes the single plane or the z-stack of planes to image at runtime. It then acquires a set of reference
   sub-planes above and below each target plane and uses the resulting volume to build the `MotionEstimator.me` file,
   which stores the per-ROI motion estimators. During the runtime, the motion manager pairs these estimators with the
   `MariusMotionCorrector2` class to correct X and Y drift with the galvos and Z drift with the fast-Z actuator.
2. **High-definition z-stack acquisition.** The function increases the resolution of the target ROIs and repeats the
   z-stack acquisition, generating a high-definition `zstack.tiff` file that is kept alongside the TIFF files acquired
   during runtime. It then rescales the ROIs back to their runtime dimensions and saves a snapshot of the imaging field
   as a `fov.roi` file. Together with the `MotionEstimator.me` file from step 1, the `zstack.tiff` and `fov.roi` files
   make up the three reference files produced for each session.
3. **Data acquisition.** The function configures the acquisition and motion-detection parameters for the runtime and
   enters its MQTT command loop. While in the loop, it begins, aborts, or recovers frame acquisition in response to the
   commands published by the sollertia-experiment library, answers liveness probes and state queries, and reports
   command reception and progress back to the library.

***Note,*** the function can also resume an interrupted runtime in response to the recover command.
In this mode, it skips steps 1 and 2, reloads the existing `MotionEstimator.me` file from the shared
Mesoscope data directory, and resumes frame acquisition.

___

## API Documentation

Run `help runAcquisition` in the MATLAB Command Window for the full argument reference. See the
[sollertia-experiment API documentation](https://sollertia-experiment-api-docs.netlify.app/) for the Python side of the
MQTT contract these assets serve.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). See the
[tags on this repository](https://github.com/Sun-Lab-NBB/sollertia-experiment/tags) for the available project releases.

___

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))

___

## License

This project is licensed under the Apache 2.0 License: see the [LICENSE](../../LICENSE) file for details.

___

## Acknowledgments

- The members of the [Pachitariu and Stringer lab](https://mouseland.github.io/) and Georg Jaindl, who developed the
  `MariusMotionEstimator` and `MariusMotionCorrector2` motion-correction classes distributed with ScanImage, and the
  original z-stack acquisition routine that `runAcquisition` is derived from.
