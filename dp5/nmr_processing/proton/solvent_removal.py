import json
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment as optimise
from lmfit import Parameters

from dp5.nmr_processing.proton.simulate import new_first_order_peak


def solvent_removal(solvent, y_data, x_data, picked_peaks, peak_regions, grouped_peaks, total_params, uc):
    picked_peaks = np.array(picked_peaks)

    solvent_file = (Path(__file__).parent / 'solvents.json').resolve()

    with open(solvent_file) as f:
        solvent_dict = json.load(f)

    if solvent in solvent_dict:
        solvent_data = solvent_dict[solvent]
    else:
        solvent_data = []

    picked_peaks_ppm = x_data[picked_peaks]

    # make differences vector for referencing against multiple solvent peaks
    differences = []
    peaks_to_remove = []
    solvent_regions = []

    for peak in solvent_data:
        speak_ppm = peak["exp_ppm"]
        # if only a singlet is expected for this peak find solvent peak based on amplitude and position
        if not peak["Jv"]:
            probs = norm.pdf(abs(picked_peaks_ppm - speak_ppm),
                             loc=0, scale=0.1) * y_data[picked_peaks]

            # find the maximum probability
            w = np.argmax(probs)

            # append this to the list to remove
            peaks_to_remove.append(picked_peaks[w])

            # append this to the list of differences
            differences.append(speak_ppm - picked_peaks_ppm[w])
        else:
            amp_res = []
            dist_res = []
            pos_res = []

            # limit the search to peaks +- 1 ppm either side

            srange = (picked_peaks_ppm > speak_ppm - 1) * \
                (picked_peaks_ppm < speak_ppm + 1)

            for peak in picked_peaks_ppm[srange]:
                fit_s_peaks, amp_vector, fit_s_y = new_first_order_peak(peak, peak["Jv"], np.arange(len(x_data)), 0.1, uc,
                                                                        1)

                diff_matrix = np.zeros((len(fit_s_peaks), len(picked_peaks)))

                for i, f in enumerate(fit_s_peaks):

                    for j, g in enumerate(picked_peaks):
                        diff_matrix[i, j] = abs(f - g)

                # minimise these distances
                vertical_ind, horizontal_ind = optimise(diff_matrix)
                closest_peaks = np.sort(picked_peaks[horizontal_ind])
                closest_amps = []

                for cpeak in closest_peaks:
                    closest_amps.append(total_params['A' + str(cpeak)])

                # find the amplitude residual between the closest peaks and the predicted pattern
                # normalise these amplitudes
                amp_vector = [i / max(amp_vector) for i in amp_vector]
                closest_amps = [i / max(closest_amps) for i in closest_amps]

                # append to the vector
                amp_res.append(
                    sum([abs(amp_vector[i] - closest_amps[i]) for i in range(len(amp_vector))]))
                dist_res.append(np.sum(np.abs(closest_peaks - fit_s_peaks)))
                pos_res.append(
                    norm.pdf(abs(peak - speak_ppm), loc=0, scale=0.5))

            # use the gsd data to find amplitudes of these peaks
            pos_res = [1 - i / max(pos_res) for i in pos_res]
            dist_res = [i / max(dist_res) for i in dist_res]
            amp_res = [i / max(amp_res) for i in amp_res]

            # calculate geometric mean of metrics for each peak
            g_mean = [(dist_res[i] + amp_res[i] + pos_res[i]) /
                      3 for i in range(0, len(amp_res))]

            # compare the residuals and find the minimum
            minres = np.argmin(g_mean)

            # append the closest peaks to the vector
            fit_s_peaks, amp_vector, fit_s_y = new_first_order_peak(picked_peaks_ppm[srange][minres], peak["Jv"],
                                                                    np.arange(len(x_data)), 0.1, uc, 1)
            diff_matrix = np.zeros((len(fit_s_peaks), len(picked_peaks)))

            for i, f in enumerate(fit_s_peaks):
                for j, g in enumerate(picked_peaks):
                    diff_matrix[i, j] = abs(f - g)

            # minimise these distances
            vertical_ind, horizontal_ind = optimise(diff_matrix)
            closest_peaks = np.sort(picked_peaks[horizontal_ind])

            for peak in closest_peaks:
                ind3 = np.abs(picked_peaks - peak).argmin()

                peaks_to_remove.append(picked_peaks[ind3])

                differences.append(picked_peaks_ppm[ind3] - uc.ppm(peak))

    # find the region this peak is in and append it to the list
    for peak in peaks_to_remove:
        for ind2, region in enumerate(peak_regions):
            if (peak > region[0]) & (peak < region[-1]):
                solvent_regions.append(ind2)
                break

    # now remove the selected peaks from the picked peaks list and grouped peaks

    w = np.searchsorted(picked_peaks, peaks_to_remove)
    picked_peaks = np.delete(picked_peaks, w)

    for ind4, peak in enumerate(peaks_to_remove):
        grouped_peaks[solvent_regions[ind4]] = np.delete(grouped_peaks[solvent_regions[ind4]],
                                                         np.where(grouped_peaks[solvent_regions[ind4]] == peak)).tolist()

    # resimulate the solvent regions
    solvent_region_ind = sorted(list(set(solvent_regions)))

    # now need to reference the spectrum

    # differences = list of differences in ppm found_solvent_peaks - expected_solvent_peaks

    s_differences = sum(differences)
    x_data = x_data + s_differences

    return peak_regions, picked_peaks, grouped_peaks, x_data, solvent_region_ind
