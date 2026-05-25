# sollertia-experiment

Provides data acquisition and preprocessing runtimes for Sollertia platform data acquisition systems.

![PyPI - Version](https://img.shields.io/pypi/v/sollertia-experiment)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sollertia-experiment)
[![uv](https://tinyurl.com/uvbadge)](https://github.com/astral-sh/uv)
[![Ruff](https://tinyurl.com/ruffbadge)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/sollertia-experiment)
![PyPI - Status](https://img.shields.io/pypi/status/sollertia-experiment)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/sollertia-experiment)

___

## Detailed Description

This library is part of the [Sollertia](https://github.com/Sun-Lab-NBB/sollertia) AI-assisted scientific data
acquisition and processing platform, built on the [Ataraxis](https://github.com/Sun-Lab-NBB/ataraxis) framework and
developed in the Sun (NeuroAI) lab at Cornell University. It is the data acquisition and preprocessing runtime of the
platform, pairing with the shared dataclasses and configuration tools provided by
[sollertia-shared-assets](https://github.com/Sun-Lab-NBB/sollertia-shared-assets) and feeding the downstream data
processing pipeline exposed by [sollertia-forgery](https://github.com/Sun-Lab-NBB/sollertia-forgery).

This library broadly consists of two parts: the shared assets reused by all acquisition systems (most of which are
inherited from [sollertia-shared-assets](https://github.com/Sun-Lab-NBB/sollertia-shared-assets)) and the
acquisition-system-specific bindings. The acquisition-system-specific code is tightly integrated with the platform
hardware and is generally not designed to be reused in any other context. See the
[data acquisition systems](#data-acquisition-systems) section for more details on currently supported acquisition
systems.

___

## Table of Contents

- [Installation](#installation)
- [Data Acquisition Systems](#data-acquisition-systems)
- [Mesoscope-VR System](#mesoscope-vr-data-acquisition-system)
- [Acquired Data Structure and Management](#acquired-data-structure-and-management)
- [Acquiring Data](#acquiring-data)
- [API Documentation](#api-documentation)
- [Recovering from Interruptions](#recovering-from-interruptions)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

___

## Installation

### Source

Note, installation from source is ***highly discouraged*** for anyone who is not an active project developer.

1. Download this repository to the local machine using the preferred method, such as git-cloning. Use one of the
   [stable releases](https://github.com/Sun-Lab-NBB/sollertia-experiment/releases).
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. ```cd``` to the root directory of the prepared project distribution.
4. Run ```python -m pip install .``` to install the project. Alternatively, if using a distribution with precompiled
   binaries, use ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file.

### pip

Use the following command to install the library using pip: ```pip install sollertia-experiment```.

___

## Data Acquisition Systems

A data acquisition (and runtime control) system can be broadly defined as a collection of hardware and software tools
used to conduct training or experiment sessions that acquire scientific data. Each data acquisition system can use one
or more machines (PCs) to acquire the data, with this library (sollertia-experiment) typically running on the **main** data
acquisition machine. Additionally, each system can use any number of long-term storage destinations, such as a remote
storage Server, a Network-Attached-Storage (NAS) volume, or both, to safely store the data after the acquisition (with
redundancy and parity).

In the Sollertia platform, each data acquisition system is built around the main tool used to acquire the brain
activity data. For example, the flagship system is the [Mesoscope-VR](#mesoscope-vr-data-acquisition-system) system,
which uses the [2-Photon Random Access Mesoscope (2P-RAM)](https://elifesciences.org/articles/14472). All other
components of that system are built around the Mesoscope to facilitate the acquisition of the brain activity data. Due
to this inherent specialization, each Sollertia platform data acquisition system is treated as an independent unit
that requires custom software to acquire, preprocess, and process the resultant data.

***Note!*** Since each data acquisition system is unique, the section below is iteratively expanded to include
system-specific assembly instructions for **each supported acquisition system**. Commonly, updates to this section
coincide with major or minor library version updates.

___

## Mesoscope-VR Data Acquisition System

This is the flagship Sollertia platform data acquisition system. The system broadly consists of four major parts:
1. The [2-Photon Random Access Mesoscope (2P-RAM)](https://elifesciences.org/articles/14472), assembled by
   [Thor Labs](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=10646) and controlled by
   [ScanImage](https://www.mbfbioscience.com/products/scanimage/) software. The Mesoscope control and data acquisition
   are performed by a dedicated computer referred to as the **'ScanImagePC'**. This PC is assembled and configured by
   [MBF Bioscience](https://www.mbfbioscience.com/). The only modification carried out during assembly was the
   configuration of a Server Message Block (SMB) protocol access to the root directory used by the ScanImage software
   to save the Mesoscope data.
2. The [Unity game engine](https://unity.com/products/unity-engine) running the Virtual Reality game world used in all
   experiments to control the task environment and resolve the task logic. The virtual environment runs on the main data
   acquisition computer referred to as the **'VRPC'** and relies on the [MQTT](https://mqtt.org/) communication protocol
   and the [sollertia-unity-tasks](https://github.com/Sun-Lab-NBB/sollertia-unity-tasks) Unity package to
   bidirectionally interface with the virtual task environment.
3. The [microcontroller-powered hardware](https://github.com/Sun-Lab-NBB/sollertia-micro-controllers) that allows the animal
   to bidirectionally interface with various physical components (modules) of the Mesoscope-VR system.
4. A set of visual and IR-range cameras, used to acquire behavior video data.

### Main Dependency
- ***Linux*** operating system. While the library *may* also work on Windows and (less likely) macOS, it has been
  explicitly written for and tested on the mainline [6.18 kernel](https://kernelnewbies.org/Linux_6.18) and
  Ubuntu 24.04 LTS distribution of the GNU Linux operating system using [Wayland](https://wayland.freedesktop.org/) 
  window system architecture.

### Software Dependencies
***Note!*** This list only includes *external dependencies*, which are installed *in addition* to all
dependencies automatically installed from pip / conda as part of library installation. The dependencies below have to
be installed and configured on the **VRPC** before calling runtime commands via the command line interface (CLI) exposed
by this library.

- [MQTT broker](https://mosquitto.org/) version **2.0.22**. The broker should be running locally and can use
  the **default** IP (127.0.0.1) and Port (1883) configuration.
- [FFMPEG](https://www.ffmpeg.org/download.html). As a minimum, the version of FFMPEG should support H265 and H264
  codecs with hardware acceleration (Nvidia GPU). This library was tested with the version **8.0.1**.
- [MvImpactAcquire](https://assets-2.balluff.com/mvIMPACT_Acquire/). This library is tested with version **2.9.2**,
  which is freely distributed. Higher GenTL producer versions likely work too, but they require purchasing a
  license.
- [Zaber Launcher](https://software.zaber.com/zaber-launcher/download) version **2025.12.30-1**.
- [Unity Game Engine](https://unity.com/products/unity-engine) version **6000.3.3f1 LTS**.

### Hardware Dependencies

**Note!** These dependencies only apply to the **VRPC**. Hardware dependencies for the **ScanImagePC** are determined
and controlled by MBF and ThorLabs. This library benefits from the **ScanImagePC** being outfitted with a 10-GB network
card, but this is not a strict requirement.

- [Nvidia GPU](https://www.nvidia.com/en-us/). This library uses GPU hardware acceleration to encode acquired video
  data. Any Nvidia GPU with hardware encoding chip(s) should work as expected. The library was tested with **RTX 4090**.
- A CPU with at least 12, preferably 16, physical cores. This library has been tested with
  [AMD Ryzen 7950X CPU](https://www.amd.com/en/products/processors/desktops/ryzen/7000-series/amd-ryzen-9-7950x.html).
  It is recommended to use CPUs with 'full' cores, instead of those using a mixture of 'efficiency' and 'performance' 
  cores for predictable performance of all library components.
- A 10-Gigabit capable motherboard or Ethernet adapter, such as [X550-T2](https://shorturl.at/fLLe9). Primarily, this is
  required for the high-quality machine vision cameras used to record videos of the animal's face and body. The 
  10-Gigabit lines are also used for transferring the data between the PCs used in the data acquisition process and 
  the destination machines used for long-term data storage 
  (see [acquired data management section](#acquired-data-structure-and-management) for more details).

### System Assembly

The Mesoscope-VR system consists of multiple interdependent components. We are constantly making minor changes to the
system to optimize its performance and facilitate novel experiments and projects. Treat this section as a general
system composition guide, but consult the original publication for each project over this section for instructions on
building specific system implementations used to acquire the data featured in different publications.

Physical assembly and mounting of ***all*** hardware components mentioned in the specific subsections below is discussed
in the [main Mesoscope-VR assembly section](#mesoscope-vr-assembly).

### Zaber Motors
All brain activity recordings with the Mesoscope require the animal to be head-fixed. To orient head-fixed animals on
the Virtual Reality treadmill (running wheel) and promote task performance, the system uses three groups of motors
controlled through Zaber motor controllers. The first group, the **HeadBar**, is used to position the animal's head in
Z, Pitch, and Roll axes. Together with the movement axes of the Mesoscope, this allows for a wide range of
motions necessary to align the Mesoscope objective with the brain imaging plane. The second group of
motors, the **LickPort**, controls the position of the water delivery port (tube) (and sensor) in X, Y, and Z axes. This
is used to ensure all animals have comfortable access to the water delivery tube, regardless of their head position.
This motor group also controls the position of the aversive air puff valve, which is mounted directly above the
LickPort. The third group of motors, the **Wheel**, controls the position of the running wheel in the X-axis relative 
to the head-fixed animal's body and is used to position the animal on the running wheel to promote good running 
behavior.

The current snapshot of Zaber motor configurations used by the Sollertia platform, alongside the motor parts list and
electrical wiring instructions, is available
[here](https://drive.google.com/drive/folders/1SL75KE3S2vuR9TTkxe6N4wvrYdK-Zmxn?usp=drive_link).

**Warning!** Zaber motors have to be configured correctly to work with this library. To (re)configure the motors to work
with the library, apply the setting snapshots from the link above via the
[Zaber Launcher](https://software.zaber.com/zaber-launcher/download) software. Read the instructions in
the 'Applying Zaber Configuration' document for the correct application procedure.

**Although highly discouraged, it is also possible to edit the motor settings manually**. To configure the motors
to work with this library, overwrite the non-volatile User Data of each motor device (controller) with
the data expected by this library:
1. **User Data 0**: Device CRC Code. This variable should store the CRC32-XFER checksum of the device's name
   (user-defined name). During runtime, the library generates the CRC32-XFER checksum of each device's name and compares
   it against the value stored inside the User Data 0 variable to ensure that each device is configured appropriately to
   work with the sollertia-experiment library. **Hint!** Use the `sle get checksum` console command to generate the CRC32-XFER
   checksum for each device during manual configuration, as it uses the same code as used during runtime and, therefore,
   guarantees that the checksums match.
2. **User Data 1**: Device ShutDown Flag. This variable tracks whether the device was properly shut down during
   previous runtimes. The library sets this variable to 0 at runtime startup and back to 1 during proper shutdown.
   This flag is only checked for devices marked as "unsafe" (see User Data 10 below). For unsafe devices that were
   not properly shut down, the system prompts for user confirmation before re-homing the motor at the beginning of the 
   next runtime.
3. **User Data 10**: Device Unsafe Flag. This variable should be set to **1** for motors that can be positioned in a
   way that is not safe to home after power cycling (e.g., motors that could collide with other components during
   homing). When a motor marked as unsafe is not properly shut down (User Data 1 = 0), the system prompts the
   user for confirmation before proceeding with the homing sequence.
4. **User Data 11**: Device Park Position. This variable should be set to the position, in native motor units, where
   the device should be moved as part of the 'park' command and the shut-down sequence. This is used to position all
   motors in a way that guarantees they can be safely 'homed' at the beginning of the next runtime. Therefore, each
   park position has to be selected so that each motor can move to their 'home' sensor without colliding with any other
   motor **simultaneously** moving towards their 'home' position. **Note!** The LickPort uses the 'park' position as
   the **default** imaging position. During runtime, it moves to the 'park' position if it has no animal-specific
   position to use during imaging. Therefore, the park position for the LickPort should always be set so that
   it cannot harm the animal mounted in the Mesoscope enclosure while moving to the park position from any other
   position.
5. **User Data 12** Device Maintenance Position. This variable should be set to the position, in native motor units,
   where the device should be moved as part of the 'maintain' command. Primarily, this position is used during water
   delivery system calibration and the running-wheel surface maintenance. Typically, this position is calibrated to
   provide easy access to all hardware components of the system by moving all motors as far away from each other as
   reasonable.
6. **User Data 13**: Device Mount Position. This variable should be set to the position, in native motor units, where
   the device should be moved as part of the 'mount' command. For the LickPort, this position is usually far away from
   the animal, which facilitates mounting and unmounting the animal from the rig. For the HeadBar and the Wheel motor
   groups, this position is used as the **default** imaging position. Therefore, the HeadBar and the Wheel 'mount'
   positions should be set so that any (new) animal can be comfortably and safely mounted in the Mesoscope enclosure.

### Behavior Cameras
To record the animal's behavior, the system uses two high-end machine-vision cameras with 2-MegaPixel resolution. The
**face_camera** records the animal's face, while the **body_camera** records the body of the animal. To interface with 
the cameras, the system leverages customized
[ataraxis-video-system](https://github.com/Sun-Lab-NBB/ataraxis-video-system) bindings.

Specific information about the cameras and related imaging hardware, as well as the snapshot of the configuration
parameters used by each camera, is available
[here](https://drive.google.com/drive/folders/1l9dLT2s1ysdA3lLpYfLT1gQlTXotq79l?usp=sharing).

### MicroControllers
To interface with all other hardware components **other** than cameras and Zaber motors, the Mesoscope-VR system uses
Teensy 4.1 microcontrollers with specialized
[ataraxis-micro-controller](https://github.com/Sun-Lab-NBB/ataraxis-micro-controller) code. Currently, The system
uses three isolated microcontroller subsystems: **Actor**, **Sensor**, and **Encoder**.

For instructions on assembling and wiring the electronic components used in each microcontroller system, as well as the
code running on each microcontroller, see the
[microcontroller repository](https://github.com/Sun-Lab-NBB/sollertia-micro-controllers).

### Virtual Reality Task Environment (Unity)
The task environment used in all Mesoscope-VR experiments is rendered and controlled by the Unity game engine. To make
Unity work with this library, each project-specific Unity task must use the bindings and assets released as part of the
[sollertia-unity-tasks repository](https://github.com/Sun-Lab-NBB/sollertia-unity-tasks). Follow the instructions from that repository to
set up Unity Game engine to interface with this library and to create new virtual task environments.

**Note!** This library does not contain tools to initialize Unity Game engine. The desired Virtual Reality task
has to be started ('armed') ***manually*** before entering the main runtime (data acquisition session) cycle. The main
Unity repository contains more details about starting the virtual reality tasks when running experiments. During
CLI-driven experiment runtimes, the library instructs the user when to 'arm' the Unity game engine.

#### Configuration Sharing via MQTT
The sollertia-experiment library and Unity communicate bidirectionally using the MQTT protocol. This communication is used to:
- **Verify scene configuration**: At startup, sollertia-experiment requests the active Unity scene name and validates it
  against the expected scene from the experiment configuration file.
- **Exchange cue sequences**: Unity sends the VR environment's wall cue sequence to sollertia-experiment, which decomposes
  it into individual trials using the trial structures defined in the experiment configuration.
- **Synchronize runtime state**: sollertia-experiment sends encoder data (animal motion) to Unity and receives stimulus
  triggers (reward delivery, gas puffs) and task guidance state updates.
- **Control task guidance**: sollertia-experiment can adjust task guidance modes based on animal performance.

Experiment configurations are stored as YAML files in the project's configuration directory and are shared between
Unity and sollertia-experiment. A copy of the experiment configuration is preserved with each session's raw data for
reproducibility.

### Google Sheets API Integration

This library interacts with the shared Google Sheet files used by the Sollertia platform to track and communicate
certain information about the animals that participate in all projects. Currently, this includes two files: the
**surgery log** and the **water restriction log**. Primarily, this integration is used to ensure that all information
about each experiment subject (animal) is stored in the same location (on the long-term storage machine(s)).
Additionally, it is used to automate certain data logging tasks.

#### Setting up Google Sheets API Access

**Skip to the next section if a service Google Sheets API account already exists.** Most lab members can safely
ignore this section, as all service accounts are managed at the acquisition-system level, rather than individual lab
members.

1. Log into the [Google Cloud Console](https://console.cloud.google.com/welcome).
2. Create a new project.
3. Navigate to APIs & Services → Library and enable the Google Sheets API for the project.
4. Under IAM & Admin → Service Accounts, create a service account. This generates a service account ID in the format
   of `service-account@gserviceaccount.com`.
5. Use Actions → Manage Keys and, if a key does not already exist, create a new key and download it in JSON format.
   This key is then used to access the Google Sheets.

#### Adding Google Sheets Access to the Service Account
To access the **surgery log** and the **water restriction log** Google Sheets as part of this library runtime, create
and share these log files with the email of the service account created above. The service account requires **Editor**
access to both files.

**Note!** This feature requires that both log files are formatted according to the available Sollertia platform
templates. Otherwise, the parsing algorithm does not behave as expected, leading to runtime failures. Additionally, both log files
have to be pre-filled in advance, as the processing code is not allowed to automatically generate new table (log) rows.
**Hint!** Currently, it is advised to pre-fill the data a month in advance. Since most experiments last for at most a
month, this usually covers the entire experiment period for any animal.

### ScanImage PC Assets
As mentioned above, the ScanImagePC is largely assembled and configured by external contractors. However, the PC
requires additional assets and configuration post-assembly to make it compatible with sollertia-experiment-managed runtimes.

#### File System Access
To support the sollertia-experiment runtime, the ScanImagePC's filesystem must be accessible to the **VRPC** via the Server
Message Block version 3 (SMB3) or equivalent protocol. Since ScanImagePC uses Windows, it is advised to use the SMB3
protocol, as all Windows machines support it natively with minimal configuration. As a minimum, the ScanImagePC must be
configured to share the root Mesoscope output directory with the VRPC over the local network. This is required to both
fetch the data acquired by the Mesoscope during preprocessing and to control the Mesoscope during runtime.

#### Default Screenshot Directory
During runtime, the sollertia-experiment library prompts the user to generate a screenshot of the ScanImagePC desktop and
place it in the network-shared mesoscope data directory (see above). The screenshot is used to store the information 
about the red-dot alignment, the acquisition parameters, and the state of the imaging plane at the beginning of each 
session. The library is statically configured to fetch the screenshot from the shared directory and does not look in 
any other directories. Therefore, it is advised to reconfigure the default output directory used by the 'Win + PrntSc' 
command on the ScanImagePC to save the screenshots into the shared Mesoscope output directory.

#### MATLAB Assets
ScanImage software is written in MATLAB and controls all aspects of Mesoscope data acquisition. While each runtime
requires the experimenter to manually interface with the ScanImage GUI during Mesoscope preparation, all data
acquisition runtimes using the sollertia-experiment library require the user to call the **setupAcquisition** MATLAB function
available from [mesoscope assets repository](https://github.com/Sun-Lab-NBB/sollertia-mesoscope-assets). This function carries
out multiple runtime-critical tasks, including setting up the acquisition, generating and applying the online motion
correction algorithm, and allowing the VRPC to control the Mesoscope via creating or removing binary marker files.

To configure MATLAB to access the mesoscope assets, git-clone the entire repository to the ScanImagePC. Then, follow the
tutorials [here](https://www.mathworks.com/help/matlab/matlab_env/add-remove-or-reorder-folders-on-the-search-path.html)
and add the path to the root mesoscope assets directory to MATLAB's search path. MATLAB can then use all
functions from that repository, including the setupAcquisition function. The repository also contains the online
motion estimation and correction assets developed in the [Pachitariu and Stringer lab](https://mouseland.github.io/),
which are required for the setupAcquisition function to work as expected.

### Mesoscope-VR Assembly
***This section is currently a placeholder. Since the final Mesoscope-VR system design is still a work in progress, it
will be populated once the final design implementation is constructed and tested in the lab.***

The Mesoscope-VR assembly mostly consists of two types of components. First, it includes custom components manufactured
via 3D-printing or machining (for metalwork). Second, it consists of generic components available from vendors such as
ThorLabs, which are altered in workshops to fit the specific requirements of the Mesoscope-VR system. The blueprints and
CAD files for all components of the Mesoscope-VR systems, including CAD renders of the assembled system, are available
[here](https://drive.google.com/drive/folders/1Oz2qWAg3HkMqw6VXKlY_c3clcz-rDBgi?usp=sharing).

___

## Acquired Data Structure and Management

The library defines a fixed structure for storing all acquired data which uses a 4-level directory tree hierarchy:
**root** (volume), **project**, **animal**, and **session**. This structure is reused by all acquisition systems, and
it is maintained across all long-term storage destinations. After each data acquisition runtime (session), all
raw data is stored under the **root/project/animal/session/raw_data** directory stored on one or more machines mentioned
below.

Currently, each Sollertia platform data acquisition system uses a **main data acquisition PC** and any number of
**long-term storage destinations** (zero, one, or more):
1. The **main data acquisition PC** is used to acquire and preprocess the data. For example, the *VRPC* of the
   *Mesoscope-VR* system is the main data acquisition PC for that system. This PC is used to both **acquire** the data
   and, critically, to **preprocess** the data before it is moved to the long-term storage destinations.
2. Each **long-term storage destination** is a machine or storage volume to which the preprocessed data is transferred
   for long-term storage. The Mesoscope-VR system anticipates two common destinations by default: a **Server** (a
   high-performance compute server used as the primary long-term storage and analysis destination) and a **NAS** (a
   Network-Attached-Storage volume used for redundant 'cold' backup storage, typically located in a different physical
   location to provide data storage redundancy). Neither is required: a system can be configured with any number of
   destinations under arbitrary names, or with none at all.

***Critical!*** Each configured long-term storage destination is expected to be **mounted to the main acquisition PC
filesystem** (for example, using the Server Message Block 3 (SMB3) protocol). Therefore, each data acquisition system
operates on the assumption that all configured storage destination filesystems are used contiguously and can be freely
accessed by the main acquisition PC's Operating System.

***Note,*** the library transfers the preprocessed data to every configured long-term storage destination and purges
the redundant local copy from the acquisition machine once the transfer succeeds. If no destinations are configured,
the data is retained on the acquisition machine and the preprocessing is limited to the on-premises data conversion and
aggregation steps.

### Root Directory (Volume)
All data acquisition systems and all long-term storage destinations keep **ALL** Sollertia platform projects under
the same **root** directory. The exact location and name of the root directory on each machine is arbitrary but should
generally remain fixed (unchanging) over the entire lifetime of that specific machine.

### Project Directory
When a new project is created, a **project** directory **named after the project** is created under the **root**
directory of the main data acquisition machine. When the data is moved to the configured long-term storage destinations
as part of preprocessing, the project directory is also created on these destinations, if it does not already exist.

All data acquisition systems also create a **configuration** subdirectory under the root project directory. This
directory stores all supported experiment configurations for the project. The `sle run session` command searches the
configuration directory for the .yaml file with the name of the target experiment to load the experiment data.

### Animal Directory
When the library is used to acquire data for a new animal, it generates a new **animal** directory under the **root**
and **project** directory combination. The directory uses the ID of the animal as its name.

All data acquisition systems also create a **persistent_data** subdirectory under the root animal directory, which is
used to store data that is reused between data acquisition sessions.

***Critical!*** The current Sollertia platform convention stipulates that all animal IDs should be numeric. While some library
components do accept strings as inputs, it is expected that all animal IDs only consist of positive integers. Failure to
adhere to this naming convention can lead to runtime errors and unexpected behavior of all library components!

### Session Directory
Each time the library is used to acquire data, a new session directory is created under the **root**, **project** and
**animal** directory combination. The session name is derived from the current ***UTC*** timestamp at the time of the
session directory creation, accurate to ***microseconds***. Primarily, this naming format was chosen to make all
sessions acquired by the same acquisition system have unique and chronologically sortable names. The session name format
follows the order of **YYYY-MM-DD-HH-MM-SS-US**.

### Raw Data
All data acquired by this library is stored under the **raw_data** subdirectory, generated for each session. Overall,
an example path to the acquired (raw) data can therefore look like this:
`/media/Data/Experiments/Template/666/2025-11-11-05-03-234123/raw_data/`.

***Note!*** This library treats **both** newly acquired and preprocessed data as **raw**. This is because preprocessing
**does not change the content of the data**. Instead, preprocessing uses lossless compression to more efficiently
package the data for transmission and can at any time be converted back to the original format.

### Shared Raw Data

The section below briefly lists the data acquired by **all** Sollertia platform data acquisition systems. Note, each acquisition
system also generates **system-specific** data, which is listed under acquisition-system-specific sections available
after this section.

**Note!** For information about the **processed** data, see the
[main data processing library](https://github.com/Sun-Lab-NBB/sollertia-forgery).

After acquisition and preprocessing, the **raw_data** directory of each acquisition system contains, as a minimum, the
following files and subdirectories:
1. **ax_checksum.txt**: Stores the xxHash-128 checksum used to verify data integrity when it is transferred to the
   long-term storage destination. The checksum is generated before the data leaves the main data acquisition system PC
   and, therefore, accurately captures the final state of the raw data before it enters storage.
2. **hardware_state.yaml**: Stores the snapshot of the dynamically calculated parameters used by the data acquisition
   system modules during runtime. These parameters are recalculated at the beginning of each data acquisition
   session and are rounded and stored using the appropriate floating point type (usually fp64) to minimize floating
   point rounding errors. This file is also used to determine which modules were used during runtime and, consequently,
   which data can be parsed from the .npz log files generated at runtime (see below).
3. **session_data.yaml**: Stores information necessary to maintain the same session data structure across all machines
   used during data acquisition and long-term storage. This file is used by all Sollertia platform libraries as an
   entry point for working with session's data. The file also includes all available information about the identity and
   purpose of the session and can be used by human experimenters to identify the session. Since version 3.0.0, the file
   also stores the version of the sollertia-experiment and Python that were used to acquire the data.
4. **session_descriptor.yaml**: Stores session-type-specific information, such as the task parameters and experimenter
   notes. The contents of the file are different for each session type, although some fields are reused by all
   sessions. The contents for this file are partially written by the library (automatically) and, partially, by the
   experimenter (manually, at the end of each session). At the end of each runtime, a copy of the descriptor file is
   cached inside the *persistent_data* directory of the animal, replacing any already existing copy. This is used to
   optionally restore certain runtime configuration parameters between session types that support this functionality.
5. **surgery_metadata.yaml**: Stores the data on the surgical intervention(s) performed on the animal that participated
   in the session. This data is extracted from the **surgery log** Google Sheet and, for most animals, should be the
   same across all sessions.
6. **system_configuration.yaml**: Stores the configuration parameters of the data acquisition system that generated the
   session data. This is a snapshot of **all** dynamically addressable configuration parameters used by the system.
   When combined with the assembly instructions and the appropriate sollertia-experiment library version, it allows completely
   replicating the data acquisition system used to acquire the session's data.
7. **behavior_data**: Stores compressed .npz log files that contain all non-video behavior data acquired by the system.
   This includes all messages sent or received by each microcontroller, the timestamps for the frames acquired by
   each camera, and (if applicable) the main brain activity recording device (e.g.: Mesoscope). These logs also include
   session metadata, such as trials, task conditions, and system and runtime state transitions. Although the exact
   content of the behavior data directory can differ between acquisition systems, all systems used in the lab generate 
   some form of non-video behavior data.
8. **camera_data**: Stores the behavior videos recorded by video cameras used by the acquisition system. Videos are
   named using the session name and camera identifier (e.g., `{session_name}_face_camera.mp4`,
   `{session_name}_body_camera.mp4`).
9. **experiment_configuration.yaml**: This file is only created for **experiment** sessions. It stores the
   configuration of the experiment task performed by the animal during runtime. The contents of the file differ for each
   data acquisition system, but each system generates a version of this file. The file contains enough information to
   fully replicate the experiment runtime on the same acquisition system and to process and analyze the acquired data.
   Since version 4.0.0, the experiment configuration include the necessary data to fully replace the Virtual Reality 
   environment used during the experiments using Unity game engine and the assets available from the 
   [sollertia-unity-tasks](https://github.com/Sun-Lab-NBB/sollertia-unity-tasks) library.

### Shared Temporary Data
The sollertia-experiment library additionally uses the following temporary marker files and directories which are cleared
before the raw data is transmitted to the long-term storage destinations:
1. **nk.bin**: This marker is automatically cached to disk as part of creating a new session data hierarchy. Each
   runtime removes this marker file when it successfully completes its runtime preparation. If this marker exists when 
   the runtime enters the shutdown cycle, this indicates that the runtime encountered a fatal error during startup and 
   had to be terminated early. In this case, the session's data is silently deleted, as uninitialized sessions 
   necessarily do not contain any valid data.
2. **behavior_data_log**: All behavior log entries are initially saved as individual .npy files. Each .npy file stores
   a serialized log message in the form of an uint8 (byte) NumPy array. Since messages are cached to disk as soon as
   they are received by the DataLogger to minimize data loss in case of emergency shutdowns, the temporary
   behavior_data_log directory is used to store these messages during runtime. Frequently, the directory accumulates
   millions of .npy files at runtime, making it challenging for human operators to work with the data. During
   preprocessing, individual .npy files are grouped by their source (what made the log entry, e.g.: VideoSystem,
   MicroController, Data Acquisition System, etc.) and are compressed into .npz archives, one for each source. The
   .npz archives are then moved to the *behavior_data* directory, and the *behavior_data_log* with the individual
   .npy files is deleted to conserve disk space.

### Mesoscope-VR System Data

The Mesoscope-VR system instantiates a directory hierarchy both on the VRPC and the ScanImagePC. Below is the list of
files and directories found on each of these machines.

#### Raw Data

The Mesoscope-VR system generates the following files and directories, in addition to those discussed in the shared
raw data section on the VRPC:
1. **mesoscope_data**: Stores all Mesoscope-acquired data. Since Mesoscope data is only acquired for **experiment**
   and **window checking** sessions, this directory is kept empty for training session types. During preprocessing,
   the directory contents are organized to automatically work with the
   [cindra](https://github.com/Sun-Lab-NBB/cindra) processing library. The directory contains:
   - **mesoscope_XXXXXX_XXXXXX.tiff**: Recompressed Mesoscope frame stacks using LERC lossless compression.
   - **frame_invariant_metadata.json**: ScanImage metadata constant across all frames (frame rate, plane/channel
     count, and ROI information).
   - **frame_variant_metadata.npz**: Per-frame metadata (frame numbers, timestamps, acquisition triggers).
   - **cindra_parameters.json**: cindra-compatible configuration used for further data processing.
   - **MotionEstimator.me**: Reference motion estimator file.
   - **fov.roi**: Region-of-interest file defining the imaging field.
   - **zstack.tiff**: A high-resolution z-stack reference image for future alignment with histology data.
2. **zaber_positions.yaml**: Stores the snapshot of the positions used by the HeadBar, LickPort, and Wheel Zaber motor
   groups, taken at the end of the session's data acquisition. All positions are stored in native motor units. This
   file is created for all session types supported by the Mesoscope-VR system. As a backup, a copy of this file is also
   generated at the beginning of each session's runtime. This allows recovering from critical runtime failures, where 
   the runtime may not be able to generate this snapshot. During both snapshot generation timepoints, a copy of the
   generated snapshot file is also cached inside the *persistent_data* directory of the animal to support restoring the
   motors to the same position during the next session.
3. **mesoscope_positions.yaml**: Stores the snapshot of the Mesoscope objective position in the physical axes (X, Y, Z, 
   and Roll), the virtual ScanImage axes (Fast Z, Tip, and Tilt), and the laser power at the sample, taken at the end of
   the session's data acquisition. **Note!** This file relies on the experimenter updating the stored positions. It is 
   only created for window checking and experiment sessions. A copy of this snapshot file is also saved to the 
   *persistent_data* directory of the animal to support restoring the Mesoscope to the same imaging field during the 
   next session.
4. **window_screenshot.png**: Stores the screenshot of the ScanImagePC screen. The screenshot should contain the image
   of the red-dot alignment, the view of the target cell layer, the Mesoscope position information, and the data
   acquisition parameters. Primarily, the screenshot is used by experimenters to quickly reference the imaging quality
   from each experiment session. This file is only created for window checking and experiment sessions. A copy of this
   file is saved to the *persistent_data* directory of the animal to help the user to realign the red-dot to a similar
   position during the next session.

### ScanImagePC

All Mesoscope-VR system data on the ScanImagePC is stored under the user-defined ScanImagePC root directory, which is
expected to be mounted to the VRPC via the SMB or similar protocol. Under that root directory, the system creates the
following directories and files:
1. **mesoscope_data**: This directory stores all Mesoscope-acquired data for the currently running session. The
   *setupAcquisition* MATLAB function configures ScanImage software to output all data to the mesoscope_data directory,
   which is shared by all sessions, animals, and projects. This allows using the same static output path for all
   ScanImage acquisitions.
2. **session-specific mesoscope_data**: At the end of each runtime, the Mesoscope-VR system ***renames*** the
   mesoscope_data directory to instead use the session name (id). Then, it generates an empty mesoscope_data directory 
   for the next runtime. This way, all data of each completed session is stored under a unique directory named after 
   that session. This step is crucial for data preprocessing, which identifies the session data directory and pulls it 
   over to the VRPC based on the session name (id).
3. **persistent_data**. This directory is created for each unique **project** and **animal** combination,
   similar to the data structure created by sollertia-experiment on the main acquisition system PC. This directory contains
   the **first experiment day's** MotionEstimator.me and fov.roi files. These files are typically reused by all 
   following data acquisition sessions to restore the imaging field to the same location as used on the first day. The 
   full path to the persistent_data directory would typically look like **root/project/animal/persistent_data**.

#### Mesoscope-VR Temporary Data

The Mesoscope-VR system also generates the following temporary files and directories during runtime:
1. **raw_mesoscope_frames**: Stores uncompressed .TIFF stacks fetched by the VRPC from the ScanImagePC. This is done
   as part of data preprocessing to collect all data on the VRPC before executing individual preprocessing subroutines.
   The .TIFF stacks are then re-compressed using the Limited Error Raster Compression (LERC) scheme and are saved to the
   *mesoscope_data* directory. Once this process completes successfully, the *raw_mesoscope_frames* directory with all
   raw files is deleted to conserve disk space.
2. **kinase.bin**: This marker is created in the *mesoscope_data* ScanImagePC directory. During runtime, the
   *setupAcquisition* MATLAB function monitors the *mesoscope_data* directory for the presence of this marker file. If
   the file is present, the function triggers the Mesoscope data acquisition. If the file is absent, the function stops
   the Mesoscope data acquisition until the file is re-created. As such, the VRPC uses this marker file to start
   and stop the Mesoscope data acquisition during normal operation.
3. **phosphatase.bin**: This marker works similar to the *kinase.bin* marker, but is used by the VRPC to abort the
   ScanImagePC runtime at any point. When ScanImagePC is waiting for the *kinase.bin* marker to be created for the first
   time, stopping the Mesoscope acquisition necessarily requires the kinase marker to be first created and then removed,
   triggering a brief Mesoscope frame acquisition. Creating the *phosphatase.bin* marker instead triggers the
   ScanImagePC to end the runtime without waiting for the *kinase.bin* marker, effectively aborting the runtime without
   acquiring any frames.

___

## Acquiring Data

All user-facing library functionality is realized through a set of Command Line Interface (CLI) commands automatically
exposed when the library is pip-installed into a python environment. The library exposes three main CLI command groups:
`sle get`, `sle manage`, and `sle run`. Each group contains subcommands that allow further configuring their runtime.
Use `--help` argument when calling any of the commands described below to see the list of supported arguments together
with their descriptions and default values.

To use any of the commands described below, activate the python environment where the library is installed, e.g., with
`conda activate MYENV` and type one of the commands described below.

***Warning!*** Most commands described below use the terminal to communicate important runtime information to the user
or request user feedback. **Carefully read every message printed to the terminal during runtime**.
Failure to do so may damage the equipment or harm the animal!

### CLI Command Overview

The sollertia-experiment library exposes three main CLI command groups:

| Command Group | Purpose                                                  |
|---------------|----------------------------------------------------------|
| `sle get`      | Discover and evaluate data acquisition system components |
| `sle manage`   | Manage session data (preprocessing, deletion, migration) |
| `sle run`      | Execute data acquisition and maintenance sessions        |

### Step 0: Configuring the Data Acquisition System

Before acquiring data, each acquisition system has to be configured. This step is done in addition to assembling
the system and installing the required hardware components. Typically, this only needs to be done when the acquisition
system configuration or hardware changes, so most lab members can safely skip this step.

Use `sle configure system` command (from the
[sollertia-shared-assets](https://github.com/Sun-Lab-NBB/sollertia-shared-assets) library) to generate the system configuration file.
As part of its runtime, the command configures the host machine to remember the path to the generated configuration
file, so all future sollertia-experiment runtimes on that machine automatically load and use the appropriate
acquisition-system configuration parameters.

***Note!*** Each acquisition system uses unique configuration parameters. Additionally, the sollertia-experiment library always
assumes that any machine (PC) can only be used by a single data-acquisition system (is permanently a part of that
acquisition system). Only the **main** PC of the data acquisition system (e.g.: the VRPC of the Mesoscope-VR system)
that runs the sollertia-experiment library should be configured via this command.

For information about the available system configuration parameters, read the *API documentation* of the appropriate
data-acquisition system available from the [sollertia-shared-assets](https://github.com/Sun-Lab-NBB/sollertia-shared-assets) library.
Specifically, all data acquisition system configuration parameters are defined in the *SystemConfiguration* class named
after that system.

### Step 1: Creating a Project

All data acquisition sessions require a valid project to run. To create a new project, use the `sle configure project`
command (from the [sollertia-shared-assets](https://github.com/Sun-Lab-NBB/sollertia-shared-assets) library). This command can only
be called on the main PC of a properly configured data-acquisition system (see Step 0 above). As part of its runtime,
this command generates the root project directory on all machines that make up the data acquisition system.

### Step 2: Creating an Experiment

All projects that involve scientific experiments also need to define at least one **experiment configuration**.
Experiment configurations are unique for each data acquisition system and are stored inside .yaml files named after the
experiment. To generate a new experiment configuration file, use the `sle configure experiment` command (from the
[sollertia-shared-assets](https://github.com/Sun-Lab-NBB/sollertia-shared-assets) library). This command generates a **precursor**
experiment configuration file inside the **configuration** subdirectory, stored under the root project directory on the
main PC of the data acquisition system.

For information about the available experiment configuration parameters in the precursor file, read the
*API documentation* of the appropriate data-acquisition system available from the
[sollertia-shared-assets](https://github.com/Sun-Lab-NBB/sollertia-shared-assets) library.

**Mesoscope-VR Note:** Mesoscope-VR experiments also require a valid Virtual Reality environment generated through the
[sollertia-unity-tasks](https://github.com/Sun-Lab-NBB/sollertia-unity-tasks) package. The same experiment configuration file is used
by both sollertia-experiment and Unity to define the VR environment (cues, segments, and trials). After creating the
experiment configuration, use the Unity package to generate the corresponding VR scene before running experiment
sessions.

### Step 3: Discovering System Components

Before running sessions, use `sle get` subcommands to discover and verify the components accessible to the data
acquisition system:

| Command                          | Description                                                                   |
|----------------------------------|-------------------------------------------------------------------------------|
| `sle get zaber`                  | Discovers Zaber motor devices connected to the system                         |
| `sle get projects`               | Lists projects configured for the data acquisition system                     |
| `sle get experiments -p PROJECT` | Lists experiment configurations for the specified project                     |
| `sle get cameras`                | Discovers cameras accessible to the system (OpenCV and Harvesters interfaces) |
| `sle get controllers`            | Discovers microcontrollers accessible to the system                           |
| `sle get ports`                  | Lists available serial communication ports                                    |
| `sle get checksum -i STRING`     | Calculates the CRC32-XFER checksum for the input string                       |

### Step 4: Maintaining the Acquisition System

All acquisition systems contain modules that require frequent maintenance. Most of these modules are unique to each
acquisition system. Therefore, this section is further broken into acquisition-system-specific subsections.

#### Mesoscope-VR

The Mesoscope-VR system contains two modules that require frequent maintenance: the **water delivery system** and the
**running wheel**. To facilitate the maintenance of these modules, the sollertia-experiment library exposes the
`sle run maintenance` command.

This command is typically called at least twice during each day the system is used to acquire data. First, it is used
at the beginning of the day to prepare the Mesoscope-VR system for runtime by filling the water delivery system. Second,
it is used at the end of each day to empty the water delivery system.

Less frequently, this command is used to re-calibrate the water delivery system, typically, as a result of replacing
system components, such as tubing or the valve itself. The command is also occasionally used to replace the
surface material of the running wheel, which slowly deteriorates as the wheel is used.

This command can also facilitate cleaning the wheel, which is typically done before and after each runtime to remove
any biological contaminants left by each animal participating in experiment or training runtimes.

***Note!*** This runtime uses a GUI-based interface for issuing commands and monitoring the system state.

### Step 5: Acquiring Data

Each acquisition system supports one or more distinct types of data-acquisition sessions (runtimes). As a minimum, this
includes an 'experiment' session type, which is the primary use case for all acquisition systems in the lab. Some
systems may also support one or more training session types, which often do not acquire any brain activity data, but
otherwise behave similar to experiment sessions.

All session commands require common parameters: user ID, project name, animal ID, and animal weight. These are provided
to the parent `sle run session` command before specifying the session type.

#### Mesoscope-VR Session Commands

The Mesoscope-VR system supports four types of runtime sessions:

**1. Window Checking Session**
```bash
sle run session -u USER -p PROJECT -a ANIMAL -w WEIGHT check-window
```

This session guides the user through finding the imaging plane and generating the reference MotionEstimator.me and
zstack.tiff files for the checked animal. This session is typically used ~2–3 weeks after the surgical intervention and
before any training or experiment sessions to assess the quality of the intervention and the suitability of including
the animal in experiment cohorts.

**2. Lick Training Session**
```bash
sle run session -u USER -p PROJECT -a ANIMAL -w WEIGHT lick-training [OPTIONS]
```

All animals that participate in Mesoscope-VR experiments undergo a two-stage training protocol, with lick training
being the first stage. During this runtime, the animals are head fixed in the Mesoscope enclosure for ~20 minutes. The
primary goal of the runtime is to teach the animals to lick at the water tube to consume water rewards and associate
the sound tone emitted at reward delivery with water coming out of the tube. During runtime, the running wheel is
locked, so the animals cannot run.

Optional arguments include maximum training time, reward delay parameters, maximum water volume, and unconsumed reward
tracking threshold.

**3. Run Training Session**
```bash
sle run session -u USER -p PROJECT -a ANIMAL -w WEIGHT run-training [OPTIONS]
```

This is the second stage of the mandatory two-stage Mesoscope-VR training protocol. During this runtime, the animals
are head fixed in the Mesoscope enclosure for ~40 minutes. The primary goal of the runtime is to teach the animals to
run on the wheel while head-fixed and to associate running with receiving water rewards. During runtime, the running
wheel is unlocked, but the Virtual Reality screens are kept off, ensuring that the animal is not exposed to any visual
cues until the first experiment day.

Optional arguments include maximum training time, speed and duration thresholds, threshold increase parameters, maximum
water volume, idle time allowance, and unconsumed reward tracking threshold.

**4. Experiment Session**
```bash
sle run session -u USER -p PROJECT -a ANIMAL -w WEIGHT experiment -e EXPERIMENT [OPTIONS]
```

This session type is designed to execute the experiment specified in the target *experiment_configuration.yaml* file
(see above). The system supports varying experiment configurations and Virtual Reality environments, offering
experimenters the flexibility to run different projects using the same set of APIs and system hardware.

The `-e/--experiment` argument specifies the name of the experiment configuration to use.

### Step 6: Preprocessing and Managing Data

All acquisition systems support several ways of handling the session's data acquired at runtime. For most runtimes, the
choice of how to handle the data is made as part of the acquisition system shutdown sequence. However, in the case of
unexpected runtime terminations, all data preprocessing steps can also be executed manually by calling the appropriate
CLI command.

#### Preprocessing Data

The most commonly used operation is to **preprocess** the acquired data. This can be done manually by calling:

```bash
sle manage preprocess -sp SESSION_PATH
```

Preprocessing consists of two major steps. The first step pulls all available data to the main data acquisition system
machine (PC) and re-packages (re-compresses) the data to reduce its size without loss. The second step distributes
(pushes) the data to all configured long-term storage destinations, such as a Server and a NAS.

**Critical!** It is imperative that **all** valid data acquired in the lab undergoes preprocessing
**as soon as possible**. Only preprocessed data is stored in a way that maximizes its safety by using both
redundancy and parity. Data that is not preprocessed may be **lost** in the case of emergency, which is considerably
less likely for the preprocessed data.

When preprocessed data is successfully and safely pushed to long-term storage destinations, the preprocessing runtime
removes the local copy of the session's data from the data acquisition system machine to conserve disk space.

#### Deleting Session Data

The second way of managing the data is primarily used during testing and when handling interrupted sessions that did
not generate any valid data. This involves removing all session data from **both** the data acquisition system and all
long-term storage destinations. This runtime is extremely dangerous and, if not used carefully, can
***permanently delete valid data***. This mode can be triggered using:

```bash
sle manage delete -sp SESSION_PATH
```

**Warning!** This command is not recommended for most users.

#### Migrating Animal Data Between Projects

To transfer all sessions for an animal from one project to another, use:

```bash
sle manage migrate -s SOURCE_PROJECT -d DESTINATION_PROJECT -a ANIMAL_ID
```

This moves the animal's data across all accessible storage destinations.

### MCP Servers

This library provides two MCP servers that expose CLI functionality for AI agent integration.

#### Starting the Servers

Start the MCP servers using the CLI:

```bash
sle get mcp      # Discovery and evaluation tools
sle manage mcp   # Data management tools
```

#### Available Tools (sle get MCP)

| Tool                                | Description                                                      |
|-------------------------------------|------------------------------------------------------------------|
| `get_zaber_devices_tool`            | Discovers Zaber motor devices connected to the system            |
| `get_projects_tool`                 | Lists all projects configured for the acquisition system         |
| `get_experiments_tool`              | Lists experiment configurations for a specified project          |
| `get_experiment_info_tool`          | Returns detailed experiment configuration summary                |
| `get_checksum_tool`                 | Calculates CRC32-XFER checksum for an input string               |
| `get_zaber_device_settings_tool`    | Reads configuration from a Zaber device's non-volatile memory    |
| `set_zaber_device_setting_tool`     | Writes configuration to a Zaber device's non-volatile memory     |
| `validate_zaber_configuration_tool` | Validates a Zaber device's configuration for binding library use |
| `check_mount_accessibility_tool`    | Verifies a filesystem path is accessible and writable            |
| `check_system_mounts_tool`          | Verifies all configured filesystem paths are accessible          |

#### Available Tools (sle manage MCP)

| Tool                            | Description                                              |
|---------------------------------|----------------------------------------------------------|
| `preprocess_session_tool`       | Preprocesses a session's data on the host machine        |
| `delete_session_tool`           | Removes a session from all storage locations             |
| `migrate_animal_tool`           | Transfers all sessions for an animal between projects    |
| `create_project_tool`           | Creates a new project directory structure                |
| `create_experiment_config_tool` | Creates an experiment configuration from a task template |

#### Claude Desktop Configuration

Add the following to the Claude Desktop configuration file:

```json
{
  "mcpServers": {
    "sollertia-experiment-get": {
      "command": "sle",
      "args": ["get", "mcp"]
    },
    "sollertia-experiment-manage": {
      "command": "sle",
      "args": ["manage", "mcp"]
    }
  }
}
```

___

## API Documentation

See the [API documentation](https://sollertia-experiment-api-docs.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library, as well as all available
CLI commands with their arguments.

___

## Recovering from Interruptions
While it is not typical for the data acquisition or preprocessing pipelines to fail during runtime, it is possible. The
library can recover or gracefully terminate the runtime for most code-generated errors, so this is usually not a
concern. However, if a major interruption (i.e., power outage) occurs or one of the hardware assets malfunctions during
runtime, manual intervention is typically required to recover the session's data and reset the acquisition system.

### Data acquisition interruption

Data acquisition can be interrupted in two main ways, the first being due to an external asset failure, for example, if
the ScanImagePC unexpectedly shuts down during Mesoscope-VR system runtime. In this case, the runtime pauses and
instructs the user to troubleshoot the issue and then resume the runtime. This type of *soft* interruption is handled
gracefully during runtime to exclude the data collected during the interruption from the output dataset. Generally,
soft interruptions are supported for most external assets, which includes anything not managed directly by the
sollertia-experiment library and the main data acquisition system PC. While inconvenient, these
interruptions do not typically require specialized handling other than recovering and restoring the failed asset.

**Note!** While most soft interruptions typically entail resuming the interrupted runtime, it is also possible to
instead terminate the runtime. To do so, execute the `terminate` command via the GUI instead of trying to resume
the runtime. In this case, the system attempts to execute a graceful shutdown procedure, saving all valid data in the
process.

The second way involves interruption due to sollertia-experiment runtime failures or unexpected shut-downs of the main
acquisition system PC. In these cases, manual user intervention is typically required to recover the useful data and
reset the system before the acquisition can be restarted. The handling of such cases often consists of specific steps
for each supported acquisition system. Typically, these *hard* interruptions are related to major issues, such as
global facility power loss or severe malfunction of sensitive acquisition system components, such as microcontrollers
and communication cables.

#### Mesoscope-VR
If the VRPC runtime unexpectedly interrupts at any point without executing the graceful shutdown, follow these
instructions:
1. If the session involved Mesoscope imaging, shut down the Mesoscope acquisition process and make sure all required
   files (frame stacks, motion estimator data, and cranial window screenshot) have been generated and saved to the
   **mesoscope_data** directory.
2. If necessary, **manually** edit the session_descriptor.yaml, the mesoscope_positions.yaml, and the
   zaber_positions.yaml files to include actual runtime information. Estimate the volume of water delivered at runtime
   by manually reading the water tank level gauge.
3. Remove the animal from the Mesoscope enclosure. If necessary, use the *Zaber Launcher* app to directly interface with
   Zaber motors and move them in a way that allows the animal to be recovered from the enclosure.
4. Use Zaber Launcher to **manually move the HeadBar group's Roll axis to have a positive angle** (> 0 degrees). This is
   critical! If this is not done, the motor cannot home during the next session and instead collides
   with the movement guard, at best damaging the motor and, at worst, the Mesoscope.
5. If the session involved Mesoscope imaging, **rename the mesoscope_data directory to use the session name**.
   For example, from mesoscope_data → 2025-11-11-05-03-234123. **Critical!** if this is not done, the library may
   **delete** any leftover Mesoscope files during the next runtime and cannot properly preprocess the frames for the
   interrupted session during the next step.
6. Call `sle manage preprocess -sp SESSION_PATH` and provide the path to the session directory of the interrupted
   session. This preprocesses and transfers all collected data to the long-term storage destinations. This preserves
   any data acquired before the interruption and prepares the system for running the next session.

### Data preprocessing interruption
To recover from an error encountered during preprocessing, call `sle manage preprocess -sp SESSION_PATH` and provide
the path to the session directory of the interrupted session. The preprocessing pipeline automatically resumes an
interrupted runtime from the nearest checkpoint.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). See the
[tags on this repository](https://github.com/Sun-Lab-NBB/sollertia-experiment/tags) for the available project releases.

___

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))
- Kushaan Gupta ([kushaangupta](https://github.com/kushaangupta))
- Natalie Yeung
- Katlynn Ryu ([katlynn-ryu](https://github.com/KatlynnRyu))
- Jasmine Si

___

## License

This project is licensed under the Apache 2.0 License: see the [LICENSE](LICENSE) file for details.

___

## Acknowledgments

- All Sun (NeuroAI) lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments
  during the development of this library.
- The creators of all other dependencies and projects listed in the [pyproject.toml](pyproject.toml) file.

___
