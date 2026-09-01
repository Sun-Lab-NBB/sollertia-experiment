"""Provides data acquisition and preprocessing runtimes for Sollertia platform data acquisition systems.

See the `API documentation <https://sollertia-experiment-api-docs.netlify.app/>`_ for the description of available
assets. See the `source code repository <https://github.com/Sun-Lab-NBB/sollertia-experiment>`_ for more details.

Authors: Ivan Kondratyev (Inkaros), Kushaan Gupta, Natalie Yeung, Katlynn Ryu, Jasmine Si
"""

# This library's runtimes are driven through the click-based CLI commands automatically exposed by installing the
# library into a conda environment. The Python-level API it documents lives in the cross_system, mesoscope_vr, and
# vr_task subpackages, each of which declares its own __all__.

from ataraxis_base_utilities import console

# Ensures the console is enabled whenever this library is imported. Progress display is also enabled so that
# console.track() and console.progress() bars used by long-running runtimes remain visible to the user.
if not console.enabled:
    console.enable()
if not console.progress_enabled:
    console.enable_progress()

# The root package re-exports nothing, so its public namespace is intentionally empty. Import the API symbols from
# the subpackage that declares them.
__all__: list[str] = []
