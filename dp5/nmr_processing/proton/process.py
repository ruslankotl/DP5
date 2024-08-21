import logging
import time

import numpy as np

from dp5.nmr_processing.proton import (
    find_integrals,
    gradient_peak_picking,
    spectral_processing,
    solvent_removal,
    multiproc_BIC_minimisation,
    simulate_regions,
    weighted_region_centres,
    iterative_assignment,
)


logger = logging.getLogger(__name__)


def proton_processing(total_spectral_ydata, uc, solvent):
    """
    Handles the initial processing, peak picking, solvent removal and integration.
    Should not handle impurities until assignment
    arguments:
        - total_spectral_ydata: complex array for FID data
        – uc: nmrglue unit conversion object
        - solvent: required for solvent removal

    returns:
        - spectral_xdata_ppm
        - total_spectral_ydata
        - peak_regions
        - grouped_peaks
        - solvent_region_ind
        - picked_peaks
        - total_params
        - sim_regions
    """
    (
        total_spectral_ydata,
        corr_distance,
        noise_std,
        peak_regions,
    ) = spectral_processing(total_spectral_ydata)

    gradient_peaks, gradient_regions, gradient_groups, std = gradient_peak_picking(
        total_spectral_ydata, corr_distance, uc, noise_std, peak_regions
    )

    start = time.time()

    picked_peaks, grouped_peaks, peak_regions, sim_y, total_params = (
        multiproc_BIC_minimisation(
            gradient_regions,
            gradient_groups,
            total_spectral_ydata,
            corr_distance,
            uc,
            noise_std,
        )
    )

    end = time.time()

    logger.info(f"minimisation time = {(end - start) / 60:.2f} mins")

    spectral_xdata_ppm = uc.ppm_scale()

    (
        peak_regions,
        picked_peaks,
        grouped_peaks,
        spectral_xdata_ppm,
        solvent_region_ind,
    ) = solvent_removal(
        solvent,
        total_spectral_ydata,
        spectral_xdata_ppm,
        picked_peaks,
        peak_regions,
        grouped_peaks,
        total_params,
        uc,
    )

    sim_regions, full_sim_data = simulate_regions(
        total_params,
        peak_regions,
        grouped_peaks,
        total_spectral_ydata,
        spectral_xdata_ppm,
    )

    return (
        spectral_xdata_ppm,
        total_spectral_ydata,
        peak_regions,
        grouped_peaks,
        picked_peaks,
        total_params,
        sim_regions,
    )


def integrate_and_remove_impurities(
    mol,
    spectral_xdata_ppm,
    total_spectral_ydata,
    peak_regions,
    grouped_peaks,
    picked_peaks,
    total_params,
    sim_regions,
):
    """Takes the molecular structure, identifies relevant peaks"""
    (
        peak_regions,
        grouped_peaks,
        sim_regions,
        integral_sum,
        cummulative_vectors,
        integrals,
        number_of_protons_structure,
        optimum_proton_number,
        total_integral,
    ) = find_integrals(
        mol,
        peak_regions,
        grouped_peaks,
        sim_regions,
        picked_peaks,
        total_params,
        total_spectral_ydata,
    )

    # find region centres

    centres = weighted_region_centres(peak_regions, total_spectral_ydata)

    ################

    exp_peaks = []

    integrals = np.array([int(i) for i in integrals])

    for ind, peak in enumerate(centres):
        exp_peaks += [peak] * integrals[ind]

    integrals = integrals[integrals > 0.5]

    exp_peaks = spectral_xdata_ppm[exp_peaks]

    exp_peaks = np.array([round(i, 4) for i in exp_peaks])
    # it is best we standardise everyting as array of arrays for compatibility with subsequent steps
    return (
        exp_peaks,
        spectral_xdata_ppm,
        total_spectral_ydata,
        integrals,
        peak_regions,
        centres,
        cummulative_vectors,
        integral_sum,
        picked_peaks,
        total_params,
        sim_regions,
    )


def proton_assignment(nmr_data, molecule, shifts, labels) -> list:
    """
    Handles final round of impurity removal
    arguments:
    - NMRData object
    - molecule(rdkit.Mol): rdkit Mol object to track the atom numbering
    - shifts(list): list of predicted shifts
    - labels(list) list of proton labels
    returns:
    list of experimetal peak values, same order as in shifts
    """
    (
        exp_peaks,
        spectral_xdata_ppm,
        total_spectral_ydata,
        integrals,
        peak_regions,
        centres,
        cummulative_vectors,
        integral_sum,
        picked_peaks,
        total_params,
        sim_regions,
    ) = integrate_and_remove_impurities(
        mol=molecule,
        spectral_xdata_ppm=nmr_data["xdata"],
        total_spectral_ydata=nmr_data["ydata"],
        peak_regions=nmr_data["peakregions"],
        grouped_peaks=nmr_data["grouped_peaks"],
        picked_peaks=nmr_data["picked_peaks"],
        total_params=nmr_data["params"],
        sim_regions=nmr_data["sim_regions"],
    )
    assigned_shifts, assigned_peaks, assigned_labels, scaled_shifts = (
        iterative_assignment(
            mol=molecule,
            exp_peaks=exp_peaks,
            calculated_shifts=shifts,
            H_labels=labels,
            rounded_integrals=integrals,
        )
    )
    H_exp = [None] * len(shifts)
    for label, peak in zip(assigned_labels, assigned_peaks):

        w = labels.tolist().index(label)

        H_exp[w] = peak
    return np.array(H_exp)
