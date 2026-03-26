NMR Processing
==============

The :mod:`dp5.nmr_processing` package implements the automated 1D NMR workflow
that underpins DP4-AI and the current DP5 preprocessing pipeline. Its job is to
turn either raw FID data or legacy text descriptions into the peak lists,
integrals, and solvent-corrected spectra needed for probabilistic structure
assignment.

This section documents:

* how the module maps onto the DP4-AI paper and its supplementary methods,
* the end-to-end logic of the proton and carbon processing pipelines,
* the public classes and functions that coordinate assignment-ready outputs.

.. toctree::
   :maxdepth: 2

   nmr_processing_methods
   nmr_processing_api
