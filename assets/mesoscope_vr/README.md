# Mesoscope-VR ScanImage PC Assets

Provides MATLAB assets used to acquire experiment data with the ScanImage-controlled 2-Photon Random Access
Mesoscope (2P-RAM).

___

## Detailed Description

This directory stores the MATLAB assets that the Mesoscope-VR acquisition system deploys to the
**ScanImagePC** (the dedicated computer that runs the ScanImage software and controls the
Mesoscope). These assets are not part of the Python `sollertia_experiment` package; they are
deployed to the ScanImagePC and registered with MATLAB, as described below.

Currently, this directory provides a single asset, the `setupAcquisition` function. This function
automates the preparation for data acquisition and allows the sollertia-experiment library to
bidirectionally interface with the ScanImage software during runtime. It carries out three
runtime-critical tasks: setting up the online motion estimation reference, acquiring a
high-definition reference z-stack, and arming the acquisition loop that the main acquisition system PC controls by
creating and removing the `kinase.bin` and `phosphatase.bin` marker files in the shared Mesoscope data directory.

The `setupAcquisition` function is designed to work with the `MariusMotionEstimator` and
`MariusMotionCorrector2` motion-correction classes. These classes are provided as part of
the ScanImage installation on the ScanImagePC and are assumed to be available; they are not
distributed with this repository.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Registering the Asset with MATLAB](#registering-the-asset-with-matlab)
- [Usage](#usage)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

___

## Dependencies

These dependencies apply to the **ScanImagePC** and are typically satisfied by the vendor that
assembles the machine (MBF Bioscience and ThorLabs):

- [MATLAB](https://www.mathworks.com/products/matlab.html) version R2022b or above with
  [ScanImage](https://www.mbfbioscience.com/products/scanimage/) version 2023.1.0 (Premium).
- [Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html), required
  by the `MariusMotionEstimator` class for fast online motion detection and correction.
- An [NVIDIA CUDA GPU](https://www.nvidia.com/en-us/), used to accelerate online motion detection
  and correction.

___

## Registering the Asset with MATLAB

To make the `setupAcquisition` function available on the ScanImagePC:

1. Copy this `assets/mesoscope_vr` directory to the ScanImagePC, or check out the
   sollertia-experiment repository on that machine.
2. Open MATLAB and navigate to the **Command Window**.
3. Run `addpath("PATH_TO_ASSET_DIRECTORY")`, replacing `PATH_TO_ASSET_DIRECTORY` with the
   **absolute** path to the directory that contains `setupAcquisition.m` on the local machine. To
   persist the path across MATLAB sessions, follow the
   [MATLAB search-path tutorials](https://www.mathworks.com/help/matlab/matlab_env/add-remove-or-reorder-folders-on-the-search-path.html).

If registration works as expected, the `setupAcquisition` function is now available for calling from
the Command Window.

***Note,*** the `MariusMotionEstimator` and `MariusMotionCorrector2` motion-correction classes must
be present in the ScanImage installation for `setupAcquisition` to work. These classes ship with
ScanImage and require no separate registration.

___

## Usage

The Mesoscope-VR runtime prompts the experimenter to call `setupAcquisition(hSI, hSICtl)` as part of
the broader Mesoscope preparation sequence, where `hSI` is the ScanImage handle object and `hSICtl`
is the ScanImage controller. Use `help setupAcquisition` in the MATLAB Command Window for the full
list of supported arguments.

In most cases, the function executes three major steps:

1. **Motion estimation setup.** The function configures the acquisition according to the
   user-defined parameters and establishes the single plane or the z-stack of planes to image at
   runtime. It then acquires a set of reference sub-planes above and below each target plane and
   uses the resulting volume to build the `MotionEstimator.me` file that detects and corrects motion
   in the X, Y, and Z axes.
2. **High-definition z-stack acquisition.** The function increases the resolution of the target ROIs
   and repeats the z-stack acquisition, generating a high-definition `zstack.tiff` file that is kept
   alongside the TIFF files acquired during runtime.
3. **Data acquisition.** The function configures the acquisition and motion-detection parameters for
   the runtime and enters the acquisition loop. While in the loop, it starts or stops the
   acquisition depending on the presence of the `kinase.bin` and `phosphatase.bin` marker files
   created by the sollertia-experiment library.

***Note,*** the function can also resume an interrupted runtime when called with the `recovery`
argument set to `true`. In this mode, it skips steps 1 and 2, loads the existing `MotionEstimator.me`
file, and proceeds directly to step 3.

___

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))

___

## Acknowledgments

- The members of the [Pachitariu and Stringer lab](https://mouseland.github.io/) and Georg Jaindl,
  who developed the `MariusMotionEstimator` and `MariusMotionCorrector2` motion-correction classes
  distributed with ScanImage, and the original z-stack acquisition routine that `setupAcquisition`
  is derived from.
