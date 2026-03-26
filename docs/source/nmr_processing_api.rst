API Reference
=============

This page references the public classes and functions that define the
``dp5.nmr_processing`` workflow.

Orchestration
=============

.. autoclass:: dp5.nmr_processing.nmr_ai.NMRData
   :members: search_files, process_proton, process_carbon, process_description, assign

Processing Entrypoints
======================

.. autofunction:: dp5.nmr_processing.proton.process.proton_processing

.. autofunction:: dp5.nmr_processing.carbon.process.carbon_processing

Assignment Algorithms
=====================

.. autofunction:: dp5.nmr_processing.proton.assign.iterative_assignment

.. autofunction:: dp5.nmr_processing.carbon.assign.iterative_assignment

Peak Modelling and Manual Inputs
================================

.. autofunction:: dp5.nmr_processing.proton.bic_minimisation.BIC_minimisation_region_full

.. autofunction:: dp5.nmr_processing.proton.bic_minimisation.multiproc_BIC_minimisation

.. autofunction:: dp5.nmr_processing.description_files.process_description
