.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (automodule, etc.).

Command Line Interfaces
=======================
.. click:: sl_experiment.command_line_interfaces.execute:run
   :prog: sl-run
   :nested: full

.. click:: sl_experiment.command_line_interfaces.manage:manage
   :prog: sl-manage
   :nested: full

.. click:: sl_experiment.command_line_interfaces.get:get
   :prog: sl-get
   :nested: full

.. click:: sl_experiment.command_line_interfaces.configure:configure_cli
   :prog: sl-configure
   :nested: full

Mesoscope-VR Acquisition System
===============================
.. automodule:: sl_experiment.mesoscope_vr
   :members:
   :undoc-members:
   :show-inheritance:

Shared Acquisition Tools And Assets
===================================
.. automodule:: sl_experiment.shared_components
   :members:
   :undoc-members:
   :show-inheritance:
