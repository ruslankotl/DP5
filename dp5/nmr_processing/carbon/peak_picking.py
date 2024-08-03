import copy
import numpy as np
from scipy.stats.distributions import norm
from lmfit import Parameters
from .fid_corrections import minimisation


def edge_removal(total_spectral_ydata):

    if total_spectral_ydata[0] > 0:
        i = 0
        while total_spectral_ydata[i] > 0:
            total_spectral_ydata[i] = 0
            i += 1
    else:
        i = 0
        while total_spectral_ydata[i] < 0:
            total_spectral_ydata[i] = 0
            i += 1

    if total_spectral_ydata[-1] > 0:
        i = 1
        while total_spectral_ydata[-i] > 0:
            total_spectral_ydata[-i] = 0
            i += 1
    else:
        i = 1
        while total_spectral_ydata[-i] < 0:
            total_spectral_ydata[-i] = 0
            i += 1
    return total_spectral_ydata


def iterative_peak_picking(
    total_spectral_ydata,
    threshold,
    corr_distance,
):

    mu, std = norm.fit(total_spectral_ydata[0:1000])

    picked_peaks = []

    # find all maxima

    maxima = []

    for point in range(1, len(total_spectral_ydata) - 1):
        if (total_spectral_ydata[point] > total_spectral_ydata[point + 1]) & (
            total_spectral_ydata[point] > total_spectral_ydata[point - 1]
        ):
            maxima.append(point)

    # start fitting process

    fit_y = np.zeros(len(total_spectral_ydata))

    while len(maxima) > 0:

        params = Parameters()

        # find peak with greatest amplitude:

        ind1 = np.argmax(total_spectral_ydata[maxima])

        peak = maxima[ind1]

        picked_peaks.append(peak)

        fit_y = minimisation(peak, fit_y, total_spectral_ydata, corr_distance)

        new_maxima = []

        for ind2 in maxima:

            if total_spectral_ydata[ind2] > threshold * std + fit_y[ind2]:

                new_maxima.append(ind2)

        maxima = copy.copy(new_maxima)

    picked_peaks = sorted(picked_peaks)

    return picked_peaks, fit_y
