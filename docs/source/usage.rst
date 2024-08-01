Usage
=====

.. _installation:

Installation
------------

To use DP5, clone this repository:

.. code-block:: console

    git clone https://github.com/ruslankotl/DP5.git


Then change to the top-level directory of the repository and install the repository using

.. code-block:: console

    pip install -e .


.. _configuration:

Configuration
-------------

This version adds :doc:`human-editable configuration files <config_anatomy>`. They can be supplied in `.json` and `.toml` formats.
The basic elements of former command-line interface are retained for experienced user convenience.

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
   ``-w``, ``--workflow <flags>``    Workflow type. Must be followed by :ref:`workflow flags <workflowflags>` without spaces. 
   ``--stereocentres``               When generating diastereomers, limit generation to specified stereocentres.
   ``-l``, ``--log_filename``       Log file name.
   ``--log_level``                   Logging levels. Can be ``warning``, ``info``, or ``debug``. Default level is ``info``.
================================== ==========================


Workflow arguments
^^^^^^^^^^^^^^^^^^

Specifies workflow actions. Will load the values from the :ref:`configuration file <cfg_workflowflags>` if left unset.

.. _workflowflags:


=============== ======================== ===============
Workflow Flags   Config file equivalents What they mean
=============== ======================== ===============
``c``           ``cleanup``              generate 3D structure, optimise using MMFF
``g``           ``generate``             generate diastereomers
``m``           ``conf_search``          perform conformational search
``o``           ``dft_opt``              optimise geometries using DFT
``e``           ``dft_energies``         calculate single point energies using DFT
``n``           ``dft_nmr``              calculate NMR spectra using DFT-GIAO method
``a``           ``assign_only``          assignment only **(currently not supported)**
``s``           ``dp4``                  perform DP4 analysis
``w``           ``dp5``                  perform DP5 analysis
=============== ======================== ===============

Default DP4 workflow for establishing stereochemistry would be specified as ``-w gnms``, 
the best results would be produced using ``-w gnomes``.

In general, conformational search should provide representative geometries, 
DFT optimisation would provide accurate geometries, 
and single-point energies would increase precision 
by re-weighting the conformers.