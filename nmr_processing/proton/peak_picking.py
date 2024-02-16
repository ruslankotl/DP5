import copy

import numpy as np

from dp5.nmr_processing.proton.fid_corrections import baseline_find_signal

def gradient_peak_picking(y_data, corr_distance, uc, std, binary_map_regions):
    final_peaks = []

    # estimate std of second derivative data

    ddy = np.diff(y_data, 2)

    ddy = ddy / np.max(ddy)

    # find peaks

    classification, sigma = baseline_find_signal(-1 * ddy, corr_distance, False, 2)

    ddy1 = np.roll(ddy, 1)

    ddyn1 = np.roll(ddy, -1)

    p = np.where((ddy < ddy1) & (ddy < ddyn1))[0]

    peaks = p[classification[p] == 1]

    peaks1 = np.roll(peaks, 1)

    distance = np.min(abs(peaks1 - peaks))

    # must make sure the convolution kernel is odd in length to prevent the movement of the peaks

    peaks = np.sort(peaks)

    peakscopy = copy.copy(peaks)
    ddycopy = copy.copy(ddy[peaks] / np.max(ddy))

    while distance < corr_distance:
        # roll the peaks one forward
        peakscopy1 = np.roll(peakscopy, 1)

        # find distances between peaks
        diff = np.abs(peakscopy - peakscopy1)

        # find where in the array the smallest distance is
        mindist = np.argmin(diff)

        # what is this distance
        distance = diff[mindist]

        # compare the values of the second derivative at the closest two peaks

        compare = np.argmax(ddycopy[[mindist, mindist - 1]])

        peakscopy = np.delete(peakscopy, mindist - compare)
        ddycopy = np.delete(ddycopy, mindist - compare)

    # remove any peaks that fall into the noise

    n = y_data[peakscopy]

    w = n > 5 * std

    peakscopy = peakscopy[w]

    final_peaks = sorted(list(peakscopy))

    # draw new regions symmetrically around the newly found peaks

    dist_hz = uc(0, "Hz") - uc(9, "Hz")

    peak_regions = []

    for peak in final_peaks:
        l = np.arange(peak + 1, min(peak + dist_hz + 1, len(y_data))).tolist()

        m = np.arange(max(peak - dist_hz, 0), peak).tolist()

        region = m + [peak] + l

        peak_regions.append(region)

    final_regions = [peak_regions[0]]
    final_peaks_separated = [[final_peaks[0]]]

    for region in range(1, len(peak_regions)):

        if peak_regions[region][0] <= final_regions[-1][-1]:

            final_regions[-1] += peak_regions[region]

            final_peaks_separated[-1].append(final_peaks[region])

        else:

            final_regions += [peak_regions[region]]

            final_peaks_separated.append([final_peaks[region]])

    final_regions = [np.arange(min(region), max(region) + 1).tolist() for region in final_regions]

    return final_peaks, final_regions, final_peaks_separated, std