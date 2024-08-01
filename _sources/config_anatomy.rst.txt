Anatomy of a config file
========================

New config file is available in `.toml` and `.json` formats. Keywords are now grouped together for the ease of handling.
This page outlines the configurable parameters. :download:`Click to download an example <../../dp5/config/default_config.toml>`


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

.. _stereocentres:

stereocentres
--------------
List of atom indices where stereochemistry is changed. Empty by default. Useful when :ref:`generating stereoisomers <generate>`.

.. _solvent:

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

.. _cfg_workflowflags:

workflow
--------

Set of the boolean workflow flags.

.. _cleanup:

cleanup
^^^^^^^

.. tip::
    Setting ``c`` in the workflow flags (e.g., ``pydp4 ... -w c<other flags>``) will override the configuration file value.

Performs MMFF optimisation of the structure candidates. Default is ``false`` 
If both ``dft_opt`` and ``conf_search`` are set to ``false``, automatically set to ``true``.

.. _generate:

generate
^^^^^^^^

.. tip::
    Setting ``g`` in the workflow flags (e.g., ``pydp4 ... -w g<other flags>``) will override the configuration file value.

Generates diastereomers of the input molecule. Default is ``false``. Works on all stereocentres by default, 
but can be limited to selected atoms by :ref:`stereocentres <stereocentres>` option.

.. _conf_search:

conf_search
^^^^^^^^^^^^

.. tip::
    Setting ``m`` in the workflow flags (e.g., ``pydp4 ... -w m<other flags>``) will override the configuration file value.

Runs conformational search using method provided. Default is ``true``.
Recommended for accurate estimation of spectra, especially for flexible molecules.

.. _dft_energies:

dft_energies
^^^^^^^^^^^^

.. tip::
    Setting ``e`` in the workflow flags (e.g., ``pydp4 ... -w e<other flags>``) will override the configuration file value.

Runs single-point energy DFT calculations. Default value is ``false``, but we recommend setting it to ``true`` for more accurate results.
The re-ranked ensemble of conformers should lead to more precise NMR spectrum estimation.

dft_opt
^^^^^^^

.. tip::
    Setting ``o`` in the workflow flags (e.g., ``pydp4 ... -w o<other flags>``) will override the configuration file value.

Runs geometry optimisation using DFT. Default value is ``false``, but we recommend setting it to ``true``. 
DFT-optimised geometries tend to produce better match to experimental NMR spectra at the increased computational cost.

dft_nmr
^^^^^^^

.. tip::
    Setting ``n`` in the workflow flags (e.g., ``pydp4 ... -w n<other flags>``) will override the configuration file value.

Default value is ``false``. If set to ``true``, calculates NMR shifts by DFT-GIAO method. Otherwise, if set to ``false``, uses a neural net to predict the shifts.

.. warning::
    Neural network is implemented for carbon NMR only

dp4 
^^^^

.. tip::
    Setting ``s`` in the workflow flags (e.g., ``pydp4 ... -w s<other flags>``) will override the configuration file value.


Performs DP4 analysis of structure candidates. Default is ``true``.

dp5
^^^^


.. tip::
    Setting ``w`` in the workflow flags (e.g., ``pydp4 ... -w w<other flags>``) will override the configuration file value.


Performs DP5 analysis of structure candidates. Default is ``false``.

.. warning::
    DP5 analysis is implemented for carbon NMR only

assign_only
^^^^^^^^^^^

.. error::
    The method is not implemented.

Assigns the structures to the provided spectra and nothing else. Default is ``false``.

calculations_complete
^^^^^^^^^^^^^^^^^^^^^

Assumes the calculations are complete. Default is ``false``.

optimisation_converged
^^^^^^^^^^^^^^^^^^^^^^^

Assumes all optimisations have converged. Default is ``false``.

restart_dft
^^^^^^^^^^^
Use when conformational search has completed, but the DFT calculations must be restarted. Default is ``false``.

mm_complete
^^^^^^^^^^^
Does nothing. Default is ``false``.

dft_complete
^^^^^^^^^^^^
Does nothing. Default is ``false``.

.. _conformer_search:

conformer_search
----------------
Configuration of conformational search.

method
^^^^^^
Currently supports ``macromodel``, ``etkdg``. Default is ``macromodel``. ``tinker`` is not tested on the current version.
It is possible to add support for the new methods (will be shown in the appropriate section).

force_field
^^^^^^^^^^^
.. note::
    Only has an effect on :py:mod:`dp5.conformer_search.macromodel`.

Force field to use. Currently supports ``mmff`` or ``opls``. Default is ``mmff``. 


step_count
^^^^^^^^^^^
.. note::
    Only has an effect on :py:mod:`dp5.conformer_search.macromodel`.

Maximum number of steps to use in the conformational search. Default is ``10000``. 


steps_per_rotatable_bond
^^^^^^^^^^^^^^^^^^^^^^^^^
.. note::
    Only has an effect on :py:mod:`dp5.conformer_search.macromodel`.

Maximum number of steps per bond to use in the conformational search. Default is ``2500``. 


manual_five_membered_rings
^^^^^^^^^^^^^^^^^^^^^^^^^^
Use special method :py:mod:`dp5.conformer_search.five_conf` to generate conformers for five-membred rings. Default is ``false``.

five_membered_ring_atoms
^^^^^^^^^^^^^^^^^^^^^^^^^
List of 1-based indices for atoms in five-membred rings to be handled by :py:mod:`dp5.conformer_search.five_conf`. Default is an empty list.

conf_prune
^^^^^^^^^^^
.. warning::
    This method is not aware of molecular symmetry

If number of conformers per molecule exceeds ``conf_per_structure``, 
remove redundant conformers if their RMSD is below ``rmsd_cutoff`` Ångström. Default value is ``true``.

conf_limit
^^^^^^^^^^

.. note::
    This method does nothing.

Total limit of conformers across all candidates. Default is ``1000``.

rmsd_cutoff
^^^^^^^^^^^
Cutoff threshold (in Ångström) to use when ``conf_prune`` is set to ``true``.

energy_cutoff
^^^^^^^^^^^^^^
Energy cutoff (in kJ/mol). Default is ``10``.

executable
^^^^^^^^^^^^
Path to your favourite executable. Add your new methods as appropriate.

schrodinger
```````````
Path to root folder for Schrodinger Suite (Maestro, MacroModel).

tinker
```````
Path to root folder for tinker.


dft
----
Configuration of DFT workflows.

.. note::
    Uses Gaussian keywords as default. 

method
^^^^^^
Accepts ``gaussian``, ``nwchem``, ``orca``. Default is ``gaussian``.
It is possible to add support for the new methods (will be shown in the appropriate section).

charge
^^^^^^
Leave as ``nan`` if charges are computed from structure input, set manually otherwise.

solvent
^^^^^^^
Solvent to use in implicit solvation DFT calculations. 
Automatically takes the value specified :ref:`previously <solvent>`.

n_basis_set
^^^^^^^^^^^
Basis set for DFT NMR calculations

n_functional
^^^^^^^^^^^^
Functional for DFT NMR calculations

o_basis_set
^^^^^^^^^^^
Basis set for DFT geometry optimisation

o_functional
^^^^^^^^^^^^
Functional for DFT geometry optimisation

e_basis_set
^^^^^^^^^^^
Basis set for DFT single-point energy calculation

e_functional
^^^^^^^^^^^^
Functional for DFT single-point energy calculation

optimisation_converged
^^^^^^^^^^^^^^^^^^^^^^
Use to skip optimisation convergence check. Default is ``false``

dft_complete
^^^^^^^^^^^^^
Use if DFT calculations are complete adn new calculations are not required. Default is ``false``.

max_opt_cycles
^^^^^^^^^^^^^^
Number of optimisation cycles allowed. Default is ``50``.

calc_force_constants
^^^^^^^^^^^^^^^^^^^^^
Use to include Hessian calculations in DFT optimisation. Speeds up convergence.

opt_step_size
^^^^^^^^^^^^^^
Optimisation step size (in Bohr or radians). Default value is ``0.3``

num_processors
^^^^^^^^^^^^^^
Number of processors for each DFT calculation.

memory
^^^^^^^
Allocated memory in megabytes. Default is ``2000``.

c13_tms
^^^^^^^
Total isotropic shielding tensor for carbon in tetramethylsilane. Default value ``191.69255`` is calculated using B3LYP/6-31G** in gas phase.

h1_tms
^^^^^^
Total isotropic shielding tensor for proton in tetramethylsilane. Default value ``31.7518583`` is calculated using B3LYP/6-31G** in gas phase.


executable
^^^^^^^^^^^^
Path to your favourite executable. Add your new methods as appropriate.

gaussian
```````````
Path to Gaussian executable.

nwchem
```````
Path to NWChem executable.

orca
``````
Path to ORCA executable.

cluster
^^^^^^^
Support for running DFT jobs on the remote cluster yet to be implemented.

dp4
----
Parameters for DP4 analysis.

stats_model
^^^^^^^^^^^
Accepts ``g`` for single gaussian, ``m`` for multuple gaussian. 
Type of statictical model to use.

param_file
^^^^^^^^^^^
Path to DP4 parameters file. If left as ``none``, uses default parameters.



