.. This file provides the instructions for how to display the API documentation generated using sphinx autodoc
   extension. Use it to declare Python documentation sub-directories via appropriate modules (automodule, etc.).

Mesoscope-VR Acquisition System
===============================

.. automodule:: sollertia_experiment.mesoscope_vr
   :members:
   :undoc-members:
   :show-inheritance:

Virtual Reality Task Interface
==============================

.. automodule:: sollertia_experiment.vr_task
   :members:
   :undoc-members:
   :show-inheritance:

Command Line Interface
======================

.. click:: sollertia_experiment.interfaces.entry_points:sle_cli
   :prog: sle
   :nested: full

Cross-System Acquisition Tools
==============================

.. automodule:: sollertia_experiment.cross_system
   :members:
   :undoc-members:
   :show-inheritance:

.. The automodule directive above discovers module-level data through the source of the module it documents, so it
   skips a constant the package re-exports, and the constant never reaches the rendered page. This directive names the
   defining module rather than the re-exporting package, because autodoc reads the attribute docstring from that
   module's source and otherwise falls back to the docstring of the value's own type.

.. autodata:: sollertia_experiment.cross_system.data_preprocessing.BEHAVIOR_LOGGER_NAME
