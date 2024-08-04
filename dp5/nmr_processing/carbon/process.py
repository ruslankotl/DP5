import logging
import time

import numpy as np
import nmrglue as ng

from dp5.nmr_processing.carbon import *
from .fid_corrections import spectral_processing
from .peak_picking import edge_removal, iterative_peak_picking
from .solvent_removal import solvent_removal
from .assign import iterative_assignment

from ..helper_functions import normalise_intensities


def carbon_processing(total_spectral_ydata, uc, solvent):
    total_spectral_ydata, spectral_ydata, threshold, corr_distance = (
        spectral_processing(total_spectral_ydata, uc)
    )

    spectral_xdata_ppm = uc.ppm_scale()

    total_spectral_ydata = edge_removal(total_spectral_ydata)

    picked_peaks, simulated_ydata = iterative_peak_picking(
        total_spectral_ydata, 5, corr_distance
    )

    picked_peaks = sorted(list(set(picked_peaks)))

    picked_peaks, removed = solvent_removal(
        simulated_ydata, spectral_xdata_ppm, solvent, uc, picked_peaks
    )

    return (
        spectral_xdata_ppm,
        total_spectral_ydata,
        picked_peaks,
        simulated_ydata,
        removed,
    )


def carbon_assignment(nmr_data, molecule, shifts, labels):
    """
    Arguments:
        nmr_data (:py:class:`~dp5.nmr_processing.nmr_ai.NMRData`:): NMR Data class
        molecule (:py:object:`rdkit.Chem.Mol`): rdkit molecule used to establish connectivity. Currently unused.
        shifts (:py:object:`numpy.ndarray`): numpy array of shifts
        label (:py:object:`numpy.ndarray`): numpy array of shifts
    """
    assigned_shifts, assigned_peaks, assigned_labels, scaled_shifts = (
        iterative_assignment(
            picked_peaks=nmr_data["exppeaks"],
            spectral_xdata_ppm=nmr_data["xdata"],
            total_spectral_ydata=nmr_data["ydata"],
            calculated_shifts=shifts,
            C_labels=labels,
        )
    )
    C_exp = [None] * len(shifts)
    for label, peak in zip(assigned_labels, assigned_peaks):

        w = labels.tolist().index(label)

        C_exp[w] = peak

    return C_exp
