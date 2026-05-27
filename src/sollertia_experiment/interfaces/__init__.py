"""This package provides the Command Line Interfaces (CLIs) for interfacing with all user-facing library components,
exposed by installing the library into a Python environment.

The interfaces are organized into two layers: a general, hardware-agnostic discovery layer ('sle get') shared by all
acquisition systems, and a per-system layer that combines configuration, acquisition, and data management for a single
acquisition system ('sle mesoscope' for the Mesoscope-VR system).
"""
