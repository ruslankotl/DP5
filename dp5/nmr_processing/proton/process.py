import logging
import time

import numpy as np

from dp5.nmr_processing.proton import find_integrals, gradient_peak_picking,\
      spectral_processing, solvent_removal, multiproc_BIC_minimisation,\
          simulate_regions, weighted_region_centres


logger = logging.getLogger(__name__)

def proton_processing(nmr_file, solvent, datatype):
    """
    Handles the initial processing, peak picking, solvent removal and integration.
    Should not handle impurities until assignment
    arguments:
        - nmr_file: file containing NMR data
        - solvent: required for solvent removal
        - datatype: type of FID data
        
    returns:
        - spectral_xdata_ppm
        - total_spectral_ydata
        - peak_regions
        - solvent_region_ind
        - picked_peaks
        - total_params
        - sim_regions
    """
    total_spectral_ydata, spectral_xdata_ppm, corr_distance, uc, noise_std, peak_regions = spectral_processing(nmr_file,
                                                                                                               datatype)

    gradient_peaks, gradient_regions, gradient_groups, std = gradient_peak_picking(total_spectral_ydata, corr_distance,
                                                                                   uc, noise_std, peak_regions)

    start = time.time()

    picked_peaks, grouped_peaks, peak_regions, sim_y, total_params = multiproc_BIC_minimisation(gradient_regions,
                                                                                                gradient_groups,
                                                                                                total_spectral_ydata,
                                                                                                corr_distance,
                                                                                                uc, noise_std)

    end = time.time()

    logger.info(f"minimisation time = {(end - start) / 60:.2f} mins")

    peak_regions, picked_peaks, grouped_peaks, spectral_xdata_ppm, solvent_region_ind = solvent_removal(
        solvent, total_spectral_ydata, spectral_xdata_ppm, picked_peaks, peak_regions, grouped_peaks,
        total_params,
        uc)

    sim_regions, full_sim_data = simulate_regions(total_params, peak_regions, grouped_peaks, total_spectral_ydata,
                                                  spectral_xdata_ppm)

    

    return spectral_xdata_ppm, total_spectral_ydata, peak_regions, solvent_region_ind ,picked_peaks, total_params, sim_regions

def integrate_and_remove_impurities(mol,spectral_xdata_ppm, total_spectral_ydata,peak_regions,picked_peaks, solvent_region_ind,total_params, sim_regions):
    # warning: not done yet!!!
    peak_regions, grouped_peaks, sim_regions, integral_sum, cummulative_vectors, integrals, number_of_protons_structure, optimum_proton_number, total_integral = find_integrals(

        mol, peak_regions, grouped_peaks, sim_regions, picked_peaks, total_params,
        total_spectral_ydata, solvent_region_ind)

    # find region centres

    centres = weighted_region_centres(peak_regions, total_spectral_ydata)

    ################

    exp_peaks = []

    integrals = np.array([int(i) for i in integrals])

    for ind, peak in enumerate(centres):
        exp_peaks += [peak] * integrals[ind]

    integrals = integrals[integrals > 0.5]

    exp_peaks = spectral_xdata_ppm[exp_peaks]

    exp_peaks = np.array([ round(i,4) for i in exp_peaks])
    
    return exp_peaks, spectral_xdata_ppm, total_spectral_ydata, integrals, peak_regions, centres, cummulative_vectors, integral_sum, picked_peaks, total_params, sim_regions

