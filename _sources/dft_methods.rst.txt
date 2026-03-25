DFT Module
==========

The :mod:`dp5.dft` package provides a backend-agnostic workflow for running
quantum chemistry calculations and parsing outputs into the data structures used
by DP5.

This section documents:

* the class inheritance model used by DFT backends,
* the orchestration logic from dispatch to parsed outputs,
* the backend class methods and helper functions.

.. toctree::
    :maxdepth: 2

    dft_architecture
    dft_api