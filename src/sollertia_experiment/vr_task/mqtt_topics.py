"""Provides the MQTT topic catalog used to communicate with the Unity game engine that runs the Virtual Reality
task environment.
"""

from enum import StrEnum


class VRTaskMQTTTopics(StrEnum):
    """Defines the set of MQTT topics used to communicate with the Unity game engine that runs the Virtual Reality
    task environment.

    Notes:
        The topics defined in this enumeration are used in addition to the topic defined by the hardware module
        interfaces of the acquisition system that drives the Virtual Reality task.
    """

    UNITY_TERMINATION = "Gimbl/Session/Stop"
    """Stops the Unity game session."""
    UNITY_STARTUP = "Gimbl/Session/Start"
    """Starts the Unity game session."""
    CUE_SEQUENCE = "CueSequence/"
    """The topic to which Unity sends the sequence of Virtual Reality cues used by the current game session."""
    CUE_SEQUENCE_REQUEST = "CueSequenceTrigger/"
    """Requests Unity to send the sequence of Virtual Reality cues used by the current game session."""
    DISABLE_LICK_GUIDANCE = "RequireLick/True/"
    """Disables lick guidance for reinforcing trials (animal must lick to trigger reward)."""
    ENABLE_LICK_GUIDANCE = "RequireLick/False/"
    """Enables lick guidance for reinforcing trials (reward on collision without lick)."""
    DISABLE_OCCUPANCY_GUIDANCE = "RequireWait/True/"
    """Disables occupancy guidance for aversive trials (animal must meet duration requirement)."""
    ENABLE_OCCUPANCY_GUIDANCE = "RequireWait/False/"
    """Enables occupancy guidance for aversive trials (brake pulse on early exit)."""
    SHOW_REWARD_ZONE_BOUNDARY = "VisibleMarker/True/"
    """Requests Unity to show the task guidance mode collision box to the animal."""
    HIDE_REWARD_ZONE_BOUNDARY = "VisibleMarker/False/"
    """Requests Unity to hide the task guidance mode collision box from the animal."""
    UNITY_SCENE_REQUEST = "SceneNameTrigger/"
    """Requests Unity to send the name of the currently used game scene."""
    UNITY_SCENE = "SceneName/"
    """The topic to which Unity sends the name of the currently used game scene."""
    STIMULUS = "Gimbl/Stimulus/"
    """The topic used by Unity to notify the runtime when the animal triggers a stimulus (water reward or gas puff)."""
    TRIGGER_DELAY = "Gimbl/TriggerDelay/"
    """The topic to which Unity sends the occupancy delay to enforce by briefly pulsing the brake."""
    ENCODER_DATA = "LinearTreadmill/Data"
    """Sends animal motion (distance) updates to Unity."""
    LICK_EVENT = "LickPort/"
    """Sends lick event notifications to Unity."""
