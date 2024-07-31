Usage
=====

.. _installation:

Installation
------------

To use DP5, clone this repository:

.. code-block:: console

    git clone whatever


Then install the repository using

.. code-block:: console

    pip install -e .


.. _configuration:

Configuration
-------------

This version adds human-editable configuration files. They can be supplied in .json and .toml formats.
For backwards compatibility, the basic elements of former command-line interface are retained:

Command line arguments
^^^^^^^^^^^^^^^^^^^^^^

.. attention::
    Arguments from command line override arguments from configuration file!

================================== ==========================
 Command Line Flags                  What they mean 
================================== ==========================
``-s``, ``--structure_files``          Paths to structure files.
   ``-n``, ``--nmr_file``            Paths to NMR spectra or their description.
   ``-c``, ``--config``              Path to configuration file for the run. Defaults to ``dp5/config/default_config.toml``.
   ``-o``, ``--output``             Path to output directory. Defaults to current working folder.
   ``-i``, ``--input_type``         Input file type. Can be ``sdf``, ``smiles``, ``smarts``, or ``inchi``. Default is ``sdf``.
   ``-w``, ``--workflow``            Workflow type. Must be followed by :ref:`workflow flags<workflowflags>` without spaces. 
   ``--stereocentres``               When generating diastereomers, limit generation to specified stereocentres.
   ``-l``, ``--log_filename``       Log file name.
   ``--log_level``                   Logging levels. Can be ``warning``, ``info``, or ``debug``. Default level is ``info``.
================================== ==========================


Workflow arguments
^^^^^^^^^^^^^^^^^^

Specifies workflow actions. 

.. _workflowflags:


=============== ==============
Workflow Flags  What they mean
=============== ==============
c               generate 3D structure, optimise using MMFF
g               generate diastereomers
m               perform conformational search
o               optimise geometries using DFT
e               calculate single point energies using DFT
n               calculate NMR spectra using DFT-GIAO method
a               assignment only **(currently not supported)**
s               perform DP4 analysis
w               perform DP5 analysis
=============== ==============

Default DP4 workflow for establishing stereochemistry would be specified as ``-w gnms``, 
the best results would be produced using ``-w gnomes``.

In general, conformational search should provide representative geometries, 
DFT optimisation would provide accurate geometries, 
and single-point energies would increase precision 
by re-weighting the conformers.