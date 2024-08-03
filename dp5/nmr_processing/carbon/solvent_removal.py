import itertools
import numpy as np
from lmfit import Parameters

from ..helper_functions import lorentzian


def solvent_removal(simulated_y_data, spectral_xdata_ppm, solvent, uc, picked_peaks):

    if solvent == "chloroform":

        exp_ppm = [77]

        Jv = [[64]]

    elif solvent == "dimethylsulfoxide":

        exp_ppm = [39.51]

        Jv = [[42, 42, 42]]

    elif solvent == "pyridine":

        exp_ppm = [150.35, 135.91, 123.87]

        Jv = [[55, 55], [49, 49], [50, 50]]

    elif solvent == "methanol":

        exp_ppm = [49.15]

        Jv = [[42.8, 42.8, 42.8]]

    elif solvent == "benzene":

        exp_ppm = [128.39]

        Jv = [[24.3]]

    else:

        exp_ppm = []
        Jv = [[]]

    # remove all solvent peaks

    removed = []

    for J, p in zip(Jv, exp_ppm):

        exp = uc(p, "ppm")

        region = np.arange(exp - 1000, exp + 1000)

        peak_region = []

        for peak in picked_peaks:

            if (peak > exp - 1000) & (peak < exp + 1000):
                peak_region.append(peak)

        # simulate solvent curve

        # find peak centre

        if region[0] + region[-1] & 1:
            centre = int((region[0] + region[-1] + 1) / 2)
        else:
            centre = int((region[0] + region[-1]) / 2)

        centre = uc.ppm(centre)

        params, peak_vector, amp_vector, y = first_order_peak(
            centre, J, np.array(region), 1, uc, 1
        )

        # use simulated curve in convolution

        convolved_y = np.convolve(simulated_y_data[region], y, "same")

        mxpoint = np.argmax(convolved_y)

        mxppm = uc.ppm(region[mxpoint])

        # simulate peak in new position

        params, fit_s_peaks, amp_vector, fit_s_y = first_order_peak(
            mxppm, J, np.array(region), 1, uc, 1
        )

        # find average of fitted peaks for referencing:

        av = sum(fit_s_peaks) / len(fit_s_peaks)

        avppm = uc.ppm(av)

        spectral_xdata_ppm -= avppm - p

        to_remove = []

        # find picked peaks closest to the "fitted" solvent multiplet

        for peak in fit_s_peaks:

            i = np.abs(np.array(picked_peaks) - peak).argmin()

            to_remove.append(i)

        removed.extend([picked_peaks[i] for i in to_remove])

        to_remove = sorted(list(set(to_remove)), reverse=True)

        for peak in to_remove:
            picked_peaks.pop(peak)

    removed = np.array(removed)

    return picked_peaks, removed


def first_order_peak(start_ppm, J_vals, x_data, corr_distance, uc, m):

    # new first order peak generator using the method presented in Hoye paper

    start = uc(str(start_ppm) + "ppm")

    start_Hz = uc.hz(start)

    J_vals = np.array(J_vals)

    if len(J_vals) > 0:

        peaks = np.zeros((2 * m + 1) ** len(J_vals))

        if m == 0.5:

            l = [1, -1]

        if m == 1:

            l = [1, 0, -1]

        # signvector generator

        signvectors = itertools.product(l, repeat=len(J_vals))

        shifts = []

        for ind, sv in enumerate(signvectors):
            shift = J_vals * sv
            shift = start_Hz + 0.5 * (np.sum(shift))
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

    else:
        peak_vector = start

    split_params = Parameters()

    for index, peak in enumerate(peak_vector):
        split_params.add("amp" + str(peak), value=amp_vector[index])
        split_params.add("pos" + str(peak), value=peak)
        split_params.add("width" + str(peak), value=2 * corr_distance)

    y = lorenz_curves(split_params, x_data, peak_vector)

    y = y / np.max(y)

    return split_params, peak_vector, amp_vector, y


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
