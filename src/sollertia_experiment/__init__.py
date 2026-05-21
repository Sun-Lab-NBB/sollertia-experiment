"""Provides data acquisition and preprocessing runtimes for Sollertia platform data acquisition systems.

See the `API documentation <https://sollertia-experiment-api-docs.netlify.app/>`_ for the description of available
assets. See the `source code repository <https://github.com/Sun-Lab-NBB/sollertia-experiment>`_ for more details.

Authors: Ivan Kondratyev (Inkaros), Kushaan Gupta, Natalie Yeung, Katlynn Ryu, Jasmine Si
"""

# Unlike most other libraries, all of this library's features are realized via the click-based CLI commands
# automatically exposed by installing the library into a conda environment. Therefore, it currently does not contain
# any explicit API exports.

from ataraxis_base_utilities import console

# Ensures the console is enabled whenever this library is imported.
if not console.enabled:
    console.enable()
