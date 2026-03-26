Method Summary
==============

This page records the implementation-level understanding used to document the
``dp5.nmr_processing`` package. It is based on the DP4-AI paper,
``DP4-AI automated NMR data analysis: straight from spectrometer to structure``
(Chem. Sci., 2020, 11, 4351-4359), together with the supplementary information
sections on automated NMR processing and assignment.

Paper Context
=============

DP4-AI was introduced to remove the most labour-intensive part of classical DP4
analysis: manual extraction of peak positions, integrals, and assignments from
1D proton and carbon NMR spectra. The method couples two ideas:

* robust automated processing of raw spectra,
* probabilistic matching of calculated GIAO shifts to experimental peaks.

The paper's central claim is not only that these steps can be automated, but
that they can be automated without materially reducing stereochemical
assignment performance relative to expert-curated DP4 input files.

In the codebase, that automation is concentrated in
:mod:`dp5.nmr_processing`. The package provides the experimental side of the
workflow, while :mod:`dp5.dft` and :mod:`dp5.analysis` consume the processed
peak lists and calculated shifts.

Implementation Map
==================

The current implementation divides the workflow into four layers.

Input handling
--------------

The :class:`dp5.nmr_processing.nmr_ai.NMRData` class is the orchestration layer.
It detects whether the user supplied Bruker data, JCAMP-DX data, or a manual
description file. Raw FID inputs are converted into normalised frequency-domain
spectra by :mod:`dp5.nmr_processing.helper_functions`, while text descriptions
are parsed by :mod:`dp5.nmr_processing.description_files`.

Proton pipeline
---------------

The proton path follows the DP4-AI methodology most closely.

1. The FID is Fourier transformed, phased, baseline corrected, and analysed to
   estimate a correlation distance and noise level.
2. Candidate peaks are identified from minima in the second derivative while
   retaining a deliberately low threshold so that weak signal peaks are not
   missed.
3. Nearby peaks are grouped into provisional multiplets and each region is fit
   with a Pearson-VII line-shape model.
4. Individual component peaks are removed if doing so lowers the Bayesian
   Information Criterion enough to justify the simpler model.
5. Known solvent multiplets are identified from a solvent database, removed, and
   used to reference the spectrum.
6. The fitted line-shape model is integrated, a plausible total proton count is
   selected by maximising an integer-likeness score, and low-integral regions
   are removed as impurities.
7. The final multiplet centres and rounded integrals are passed to the proton
   assignment algorithm.

This logic is implemented mainly in
:func:`dp5.nmr_processing.proton.process.proton_processing`,
:func:`dp5.nmr_processing.proton.bic_minimisation.multiproc_BIC_minimisation`,
and :func:`dp5.nmr_processing.proton.assign.iterative_assignment`.

Carbon pipeline
---------------

The carbon path is simpler because integrals are not assignment constraints in
the same way they are for proton spectra.

1. The spectrum is corrected and its noisy edges are zeroed.
2. An iterative peak-picking loop repeatedly fits the most intense remaining
   peak and keeps any maxima that still rise sufficiently above the fitted
   background.
3. Solvent peaks are removed using expected solvent patterns.
4. The assignment step uses peak position and amplitude together, because
   carbon spectra often contain low-intensity noise peaks and may have fewer
   experimental peaks than atoms.

This logic is implemented mainly in
:func:`dp5.nmr_processing.carbon.process.carbon_processing` and
:func:`dp5.nmr_processing.carbon.assign.iterative_assignment`.

Assignment Logic
================

Shared idea
-----------

Both nuclei use an assignment probability matrix whose element ``M[i, j]``
measures how plausible it is for calculated shift ``i`` to correspond to
experimental peak ``j``. The final assignment is obtained with a Hungarian
linear-sum optimisation, which is the same core strategy described in the
paper.

External and internal scaling
-----------------------------

The paper emphasises that internal DP4-style scaling cannot be used at the
start because the assignments are not known yet. The code therefore performs an
initial pass with fixed empirical external scaling factors and then recalculates
the scaling relation from the provisional assignment. This two-stage logic is
present in both the proton and carbon assignment modules.

Proton-specific constraints
---------------------------

The proton assignment implementation uses integral information explicitly.
Expanded multiplet centres allow a peak to be assigned as many times as its
rounded integral permits. Methyl protons are handled first as grouped units,
which reflects the paper's observation that methyl groups behave as equivalent
signals and should be assigned to the same peak before the remaining protons are
optimised.

Carbon-specific weighting
-------------------------

The carbon assignment implementation follows the DP4-AI strategy of using peak
amplitudes to distinguish more reliable signal peaks from likely noise. A KDE of
peak heights is used to derive amplitude groups and weights. The code then
duplicates assignment columns so that one experimental peak may explain multiple
equivalent carbons, while progressively penalising repeated use of the same
peak. After two assignment rounds, a bias-driven reassignment step checks for
cases where a nearby intense unassigned peak is chemically more plausible than a
weak peak chosen in the initial optimisation.

Why the Proton Pipeline Is More Elaborate
=========================================

The supplementary methods make clear that automated proton processing is harder
than automated carbon processing because the final data product is not just a
list of peak positions. The pipeline must also recover multiplet boundaries,
remove over-picked noise peaks, estimate integer-like integrals, and preserve
equivalence information needed by the proton assignment algorithm.

That distinction is visible in the code:

* proton processing has dedicated modules for gradient peak picking, BIC-based
  deconvolution, analytic integration, and impurity filtering,
* carbon processing delegates more of the complexity to the assignment stage,
  where amplitude weighting and repeat assignments compensate for noisier peak
  lists.

Relationship to Legacy Description Files
========================================

The package still supports hand-written NMR description files because DP4-style
workflows pre-date automated raw-data processing. When a description file is
used, the code bypasses the FID pipelines and instead parses manually supplied
shifts, equivalence groups, and omitted atoms. This is why
:mod:`dp5.nmr_processing.description_files` remains part of the public module:
it preserves compatibility with curated inputs while the automated pipelines are
used whenever raw spectra are available.

Practical Reading Guide
=======================

For code readers, the most useful entry points are:

* :class:`dp5.nmr_processing.nmr_ai.NMRData` for orchestration,
* :func:`dp5.nmr_processing.proton.process.proton_processing` for the proton
  pipeline,
* :func:`dp5.nmr_processing.carbon.process.carbon_processing` for the carbon
  pipeline,
* :func:`dp5.nmr_processing.proton.assign.iterative_assignment` for
  integral-constrained proton assignment,
* :func:`dp5.nmr_processing.carbon.assign.iterative_assignment` for
  amplitude-aware carbon assignment.

The API reference page links these functions directly to their source
docstrings.
