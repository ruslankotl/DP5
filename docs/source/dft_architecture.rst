DFT Architecture And Class Logic
================================

Overview
--------

The DFT workflow in :mod:`dp5.dft` is designed around one abstract base class
and several concrete backend implementations:

* :class:`dp5.dft.base_dft_method.BaseDFTMethod`
* :class:`dp5.dft.gaussian.DFTMethod`
* :class:`dp5.dft.nwchem.DFTMethod`
* :class:`dp5.dft.orca.DFTMethod`

The module-level entrypoint :func:`dp5.dft.run_dft.dft_calculations` loads the
backend selected in config and executes the requested DFT stages.

Class Inheritance
-----------------

Inheritance structure:

* ``BaseDFTMethod`` defines shared orchestration logic and abstract methods.
* Each backend ``DFTMethod`` subclass implements engine-specific input writing,
  command format, and output parsing.

The following directives show the class hierarchy and inherited members:

.. autoclass:: dp5.dft.base_dft_method.BaseDFTMethod
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dp5.dft.gaussian.DFTMethod
   :members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: dp5.dft.nwchem.DFTMethod
   :members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: dp5.dft.orca.DFTMethod
   :members:
   :show-inheritance:
   :inherited-members:

Execution Flow
--------------

1. :func:`dp5.dft.run_dft.dft_calculations` imports the backend module named by
   ``config["method"]`` and instantiates ``DFTMethod(config)``.
2. Workflow flags (``dft_opt``, ``dft_energies``, ``dft_nmr``) control which
   stage methods are called.
3. Stage methods in ``BaseDFTMethod`` call :meth:`BaseDFTMethod.get_files`.
4. ``get_files`` either loads pre-run outputs or creates/runs missing jobs.
5. Each output is parsed through backend :meth:`read_file` and mapped into DP5
   molecule attributes.

Core Base Class Logic
---------------------

The base class isolates common logic so backends only provide engine-specific
behavior:

* :meth:`dp5.dft.base_dft_method.BaseDFTMethod.get_files`
  Selects between precomputed data and active execution.
* :meth:`dp5.dft.base_dft_method.BaseDFTMethod._get_files`
  Builds file stems per conformer, reuses valid outputs, and schedules missing
  calculations.
* :meth:`dp5.dft.base_dft_method.BaseDFTMethod._run_calcs`
  Runs backend commands and checks completion status.
* :meth:`dp5.dft.base_dft_method.BaseDFTMethod.opt`
  Enforces optimisation convergence before returning geometries.
* :meth:`dp5.dft.base_dft_method.BaseDFTMethod.energy`
  Collects single-point energies and associated coordinates.
* :meth:`dp5.dft.base_dft_method.BaseDFTMethod.nmr`
  Collects shielding tensors and labels in addition to energies.

Backend Responsibilities
------------------------

Each backend subclass implements these abstract integration points:

* :meth:`dp5.dft.base_dft_method.BaseDFTMethod.write_file`
  Serialize one conformer and settings to an engine input file.
* :meth:`dp5.dft.base_dft_method.BaseDFTMethod.prepare_command`
  Return the shell command used to run one calculation.
* :meth:`dp5.dft.base_dft_method.BaseDFTMethod.read_file`
  Parse one output file and return the standard DP5 tuple contract.

This contract keeps downstream DP5 logic independent from the DFT engine in
use.
