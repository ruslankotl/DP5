import itertools

import numpy as np
from lmfit import Parameters

from dp5.nmr_processing.proton.pearson7 import p7sim, p7plot
from dp5.nmr_processing.helper_functions import lorentzian


def simulate_regions(params, peak_regions, grouped_peaks, y_data, xppm):
    sim_regions = np.empty(len(peak_regions), dtype=object)
    sim_y = np.zeros(len(y_data))

    for ind, region in enumerate(peak_regions):
        sim_y[region] = p7plot(params, region, grouped_peaks[ind], ind, xppm)

        y = p7sim(params, region, grouped_peaks[ind], ind)

        sim_regions[ind] = y

    return sim_regions, sim_y


def new_first_order_peak(start_ppm, J_vals, x_data, corr_distance, uc, spin):
    # new first order peak generator using the method presented in Hoye paper

    start = uc(str(start_ppm) + "ppm")

    start_Hz = uc.hz(start)

    J_vals = np.array(J_vals)

    peaks = np.zeros((2 * spin + 1) ** len(J_vals))

    if spin == 0.5:
        l = [1, -1]

    if spin == 1:
        l = [1, 0, -1]

    # signvector generator

    signvectors = itertools.product(l, repeat=len(J_vals))

    for ind, sv in enumerate(signvectors):
        shift = J_vals * sv

        shift = start_Hz + np.sum(shift)
        peaks[ind] = shift

    peaks = np.sort(peaks)

    peak_vector = np.array(sorted(list(set(peaks)), reverse=True))

    amp_vector = np.zeros(len(peak_vector))

    for peak in peaks:
        index = np.where(peak_vector == peak)
        amp_vector[index] += 1

    pv = []

    for index, peak in enumerate(peak_vector):
        pv.append(uc(peak, "Hz"))

    peak_vector = pv

    split_params = Parameters()

    for index, peak in enumerate(peak_vector):
        split_params.add("amp" + str(peak), value=amp_vector[index])
        split_params.add("pos" + str(peak), value=peak)
        split_params.add("width" + str(peak), value=2 * corr_distance)

    y = lorenz_curves(split_params, x_data, peak_vector)
    y = y / np.max(y)

    # where = np.where(y > 0.001)
    # y = y[where]

    return peak_vector, amp_vector, y


def lorenz_curves(params, x, picked_points):
    y = np.zeros(len(x))
    for peak in picked_points:
        y += lorentzian(
            x,
            params["width" + str(peak)],
            params["pos" + str(peak)],
            params["amp" + str(peak)],
        )
    return y
