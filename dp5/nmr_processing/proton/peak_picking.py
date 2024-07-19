import copy

import numpy as np

from dp5.nmr_processing.proton.fid_corrections import baseline_find_signal


def gradient_peak_picking(y_data, corr_distance, uc, std, binary_map_regions):
    final_peaks = []

    # estimate std of second derivative data

    ddy = np.diff(y_data, 2)

    ddy = ddy / np.max(ddy)

    # find peaks

    classification, sigma = baseline_find_signal(
        -1 * ddy, corr_distance, False, 2)

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

    final_peaks = np.sort(peakscopy)

    # draw new regions symmetrically around the newly found peaks
    # introduces assumption of J not exceeding 18 Hz

    dist_hz = uc(0, "Hz") - uc(9, "Hz")

    peak_regions = np.empty(shape=final_peaks.shape[0], dtype=object)

    for i, peak in enumerate(final_peaks):

        right = min(peak + dist_hz + 1, len(y_data))
        left = max(peak - dist_hz, 0)
        region = np.arange(left, right, 1)

        peak_regions[i] = region

    final_regions = [peak_regions[0]]
    final_peaks_separated = [[final_peaks[0]]]
    # logic: if regions overlap, stitch together
    for region in range(1, len(peak_regions)):

        if peak_regions[region][0] <= final_regions[-1][-1]:

            final_regions[-1] = np.union1d(final_regions[-1],
                                           peak_regions[region])

            final_peaks_separated[-1].append(final_peaks[region])

        else:

            final_regions += [np.array(peak_regions[region])]

            final_peaks_separated.append([final_peaks[region]])
    # clean up overlapping regions
    # consistent numpyfication for subsequent operations
    final_regions = np.asarray(final_regions, dtype=object)
    final_peaks_separated = [np.array(i) for i in final_peaks_separated]
    final_peaks_separated = np.asarray(final_peaks_separated, dtype=object)

    return final_peaks, final_regions, final_peaks_separated, std
