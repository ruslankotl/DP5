Anatomy of a config file
========================

New config file is available in `.toml` and `.json` formats. 
This page outlines the configurable parameters.


structure
---------

List of paths to the structure files used in the workflow. 
If structure cleanup or diastereomer genereation is requested, the new molecules are saved to the config file instead.

input_type
----------

Input type of the structure file as text. Default value is ``sdf``.Handled by :py:func:`~dp5.run.mol_file_preparation`.

nmr_file
-----------

List of paths to NMR data. Can be a folder containing FID data in (Bruker or JCAMP-DX format) or NMR description file.

.. _cfgstereocentres:

stereocentres
--------------
List of atom indices where stereochemistry is changed. Empty by default. Useful when :ref:`generating stereoisomers <cfggenerate>`.

solvent
--------
Name of the solvent (as in Gaussian) used in subsequent calculations.

log_level
------------
Logging level in the run. Can be set to ``info``, ``debug``, and other options specified in `Python documentation <https://docs.python.org/3/library/logging.html#logging-levels>`_

output_folder
-------------
Path to the output folder.

gui_running
-----------
Boolean flag (``true`` or ``false``). *Currently not in use.*

workflow
--------

Set of the boolean workflow flags.

cleanup
^^^^^^^

Performs MMFF optimisation of the structure candidates. 
Triggers automatically if both ``dft_opt`` and ``conf_search`` are set to `false`.

.. _cfggenerate:

generate
^^^^^^^^

Generates diastereomers of the input molecule. Works on all stereocentres by default, 
but can be limited to selected atoms by :ref:`stereocentres <cfgstereocentres>` option.

conf_search
^^^^^^^^^^^^
Runs conformational search using method provided. Default is ``macromodel``.
Currently supports ``macromodel``, ``etkdg``. ``tinker`` is not tested on the current version.
It is possible to add support for the new methods (will be shown in the appropriate section).