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
    """Process a carbon FID into the peak list used by DP5 assignment.

    The carbon pipeline follows the DP4-AI design more loosely than the proton
    path. It performs spectral correction, removes edge artefacts, iteratively
    fits strong peaks to suppress noise, removes solvent signals, and returns
    the experimental peak positions required by the carbon assignment algorithm.

    :param total_spectral_ydata: Complex frequency-domain spectrum.
    :type total_spectral_ydata: numpy.ndarray
    :param uc: ``nmrglue`` unit-conversion object for converting point indices
        into ppm.
    :type uc: object
    :param solvent: Solvent identifier used by the solvent-removal routine.
    :type solvent: str
    :returns: PPM axis, processed spectrum, picked peak indices, simulated
        spectrum used during iterative picking, and indices removed as solvent.
    :rtype: tuple
    """
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
    """Assign calculated carbon shifts to processed experimental peaks.

    :param nmr_data: Dictionary produced by :func:`carbon_processing` and
        stored on :class:`dp5.nmr_processing.nmr_ai.NMRData`.
    :type nmr_data: dict
    :param molecule: RDKit molecule used to preserve the public API. The carbon
        assignment implementation does not currently consult connectivity.
    :type molecule: object
    :param shifts: Calculated carbon shifts for the candidate structure.
    :type shifts: numpy.ndarray
    :param labels: Carbon atom labels corresponding to ``shifts``.
    :type labels: numpy.ndarray
    :returns: Experimental carbon shifts ordered to match ``shifts``.
    :rtype: numpy.ndarray
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

    return np.array(C_exp)
