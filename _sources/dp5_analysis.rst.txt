DP5 Analysis
============

.. currentmodule:: dp5.analysis.dp5

.. autoclass:: DP5

   
   .. automethod:: __init__

   .. automethod:: __call__


.. autoclass:: DP5ProbabilityCalculator

    .. automethod:: __init__

    .. automethod:: __call__

    .. automethod:: get_shifts_and_labels


Notes
-----
DP5 analysis generates vector-based representations for each atom and compares them with reference representations using Kernel Density Estimation.
It runs in two modes, denoted as `Error` and `Experimental`.

In the `Error` mode, for each conformer, the scaled errors of calculation shifts are concatenated with condensed atom representations and compared with scaled errors for `Exp5K` NMR data set.
In the `Experimental` mode, the Boltzmann-weighted calculated shifts are compared directly with experimental data.
