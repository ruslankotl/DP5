"""Includes baseline correction and phasing"""

import copy

import numpy as np
import nmrglue as ng
from scipy.stats import norm
from scipy.ndimage.filters import gaussian_filter1d as g1d, convolve1d as c1d
from lmfit import Minimizer, Parameters, report_fit
import statsmodels.api as sm

from ..helper_functions import normalise_intensities, lorentzian


def spectral_processing(total_spectral_ydata, uc):
    total_spectral_ydata = initial_processing(total_spectral_ydata)
    corr_distance = estimate_autocorrelation(total_spectral_ydata)
    convolved_y = gaussian_convolution(corr_distance, total_spectral_ydata)
    binary_map_regions = []

    threshold_vector = [3, 2.9, 2.8, 2.7, 2.6, 2.5, 2.2, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5]
    run = 0
    while len(binary_map_regions) < 2:
        threshold = threshold_vector[run]
        run += 1
        picked_points = iterative_point_picking(convolved_y, threshold)
        binary_map_regions, binary_map_list = binary_map(picked_points, uc, convolved_y)

    globalangles, phased_peak_regions, convolved_y_phased = estimate_phase_angles(
        convolved_y, binary_map_regions, corr_distance
    )

    real_convolved_y_phased = list(np.real(convolved_y_phased))
    picked_points_region = iterative_point_picking_region(
        binary_map_regions, real_convolved_y_phased, threshold
    )
    picked_peaks_region = peak_picking_region(
        real_convolved_y_phased, picked_points_region
    )

    p0, p1 = linear_regression(
        picked_peaks_region, globalangles, real_convolved_y_phased, binary_map_regions
    )
    total_spectral_ydata, spectral_ydata = final_phasing(convolved_y, p0, p1)
    total_spectral_ydata = total_spectral_ydata / np.max(total_spectral_ydata)

    return total_spectral_ydata, spectral_ydata, threshold, corr_distance


def initial_processing(total_spectral_ydata):
    real_part = ng.proc_bl.baseline_corrector(np.real(total_spectral_ydata), wd=2)
    im_part = ng.proc_bl.baseline_corrector(np.imag(total_spectral_ydata), wd=2)
    total_spectral_ydata = real_part + 1j * im_part
    return normalise_intensities(total_spectral_ydata)


def estimate_autocorrelation(total_spectral_ydata):

    real_part = np.real(total_spectral_ydata)
    real_part_copy = np.real(total_spectral_ydata)

    gzero = real_part * real_part
    gzero = np.sum(gzero)

    gdx = gzero

    counter = 0

    while gdx > 0.6 * gzero:
        real_part_copy = np.roll(real_part_copy, counter)
        gdx = real_part * real_part_copy
        counter += 1
        gdx = np.sum(gdx)

    corr_distance = counter

    return corr_distance


def gaussian_convolution(corr_distance, total_spectral_ydata):

    real_part = np.real(total_spectral_ydata)
    im_part = np.imag(total_spectral_ydata)

    real_convolved_y = g1d(real_part, corr_distance)
    im_convolved_y = g1d(im_part, corr_distance)

    convolved_y = np.array(real_convolved_y) + 1j * np.array(im_convolved_y)

    convolved_y = convolved_y / np.max(convolved_y)

    return convolved_y


def iterative_point_picking(convolved_y, threshold):

    real_convolved_y = np.real(convolved_y)
    copy_convolved_y = np.array(real_convolved_y)
    picked_points = []
    pickednumber = 1
    while pickednumber > 0:
        mu, std = norm.fit(copy_convolved_y)

        index = np.where(copy_convolved_y - mu > threshold * std)
        pickednumber = len(index[0])
        picked_points.extend(np.ndarray.tolist(index[0]))
        copy_convolved_y = np.delete(copy_convolved_y, index, axis=0)

    copy_convolved_y = np.array(real_convolved_y)
    pickednumber = 1
    while pickednumber > 0:
        mu, std = norm.fit(copy_convolved_y)
        index = np.where(copy_convolved_y - mu < -threshold * std)
        pickednumber = len(index[0])
        picked_points.extend(np.ndarray.tolist(index[0]))
        copy_convolved_y = np.delete(copy_convolved_y, index, axis=0)

    picked_points = sorted(picked_points)

    return picked_points


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


def binary_map(picked_points, uc, convolved_y):

    picked_points = np.array(picked_points)

    # find where peak blocks are

    binary_map_regions = [[picked_points[0]]]

    for x in range(0, len(picked_points) - 1):
        if picked_points[x + 1] != picked_points[x] + 1:
            binary_map_regions[-1].append(picked_points[x])
            binary_map_regions.append([picked_points[x + 1]])
    binary_map_regions[-1].append(picked_points[-1])

    # extend blocks by 50 Hz

    for block in binary_map_regions:
        start = uc.hz(block[0])
        end = start + 50
        end_point = uc(end, "Hz")
        block[0] = end_point

        start = uc.hz(block[1])
        end = start - 50
        end_point = uc(end, "Hz")
        block[1] = end_point

    # draw binary map

    binary_map_list = np.zeros(len(convolved_y))
    for block in binary_map_regions:
        binary_map_list[block[0] : block[1] : 1] = 1

    # stitch blocks together
    blocks = np.where(binary_map_list == 1)

    blocks = blocks[0] - 1
    binary_map_regions = [[blocks[0]]]

    for element in range(0, len(blocks) - 1):
        if blocks[element + 1] != blocks[element] + 1:
            binary_map_regions[-1].append(blocks[element])
            binary_map_regions.append([blocks[element + 1]])
    binary_map_regions[-1].append(blocks[-1])

    return binary_map_regions, binary_map_list


def minimisation(next_peak, fit_y, total_spectral_ydata, corr_distance):

    region = np.arange(
        max(0, next_peak - 100), min(next_peak + 100, len(total_spectral_ydata))
    )

    params = Parameters()

    params.add(
        "amp" + str(next_peak), value=total_spectral_ydata[next_peak], vary=False, min=0
    )
    params.add(
        "width" + str(next_peak),
        value=4 * corr_distance,
        vary=True,
        min=1 * corr_distance,
        max=8 * corr_distance,
    )
    params.add("pos" + str(next_peak), value=next_peak, vary=False)

    # print('minimising')

    out = Minimizer(
        residual,
        params,
        fcn_args=(fit_y[region], next_peak, region, total_spectral_ydata[region]),
    )

    results = out.minimize()

    # append the results params to the total params

    fit_yc = (
        lorentzian(
            np.arange(len(total_spectral_ydata)),
            results.params["width" + str(next_peak)],
            results.params["pos" + str(next_peak)],
            results.params["amp" + str(next_peak)],
        )
        + fit_y
    )

    return fit_yc


def residual(params, fit_y, next_peak, x, y_data):

    y = (
        lorentzian(
            x,
            params["width" + str(next_peak)],
            params["pos" + str(next_peak)],
            params["amp" + str(next_peak)],
        )
        + fit_y
    )

    difference = abs(y - y_data)

    return difference


def iterative_point_picking_region(
    binary_map_regions, real_convolved_y_phased, threshold
):

    copy_convolved_y = np.array(real_convolved_y_phased)
    picked_points = []
    pickednumber = 1

    while pickednumber > 0:
        mu, std = norm.fit(copy_convolved_y)
        index = np.where(
            (copy_convolved_y - mu > threshold * std)
            | (copy_convolved_y - mu < threshold * std)
        )
        picked_points.extend(np.ndarray.tolist(index[0]))
        pickednumber = len(index[0])
        copy_convolved_y = np.delete(copy_convolved_y, index, axis=0)

    picked_points = sorted(picked_points)
    picked_points_region = []

    for region in binary_map_regions:
        picked_points_region.append([])
        for point in picked_points:
            if point > region[0] and point < region[1]:
                picked_points_region[-1].append(point)

    return picked_points_region


def estimate_phase_angles(convolved_y, binary_map_regions, corr_distance):

    convolved_y_phased = np.array(convolved_y)

    def inte(binary_map_regions, peak_regions, corr_distance):
        ## for each region determine the baseline
        integrals = [0] * len(binary_map_regions)

        # first find average of surrounding points to ends of binary map regions to draw base line
        baselines_end = [[0, 0] for i in range(len(binary_map_regions))]
        baselines = [[] for i in range(0, len(binary_map_regions))]

        # find baseline endpoints
        for region in range(0, len(binary_map_regions)):
            for point in range(0, corr_distance - 1):
                baselines_end[region][0] += peak_regions[region][point]

            baselines_end[region][0] = baselines_end[region][0] / corr_distance
            for point in range(0, corr_distance):
                baselines_end[region][1] += peak_regions[region][-point]

            baselines_end[region][1] = baselines_end[region][1] / corr_distance

            ## draw baselines
            baselines[region] = np.linspace(
                baselines_end[region][0],
                baselines_end[region][1],
                len(peak_regions[region]) - 2 * corr_distance,
            )

            ## integrate each region below the baseline
            for point in range(0, len(baselines[region])):
                if (
                    peak_regions[region][point + corr_distance]
                    < baselines[region][point]
                ):
                    integrals[region] += abs(
                        peak_regions[region][point + corr_distance]
                        - baselines[region][point]
                    )

        return integrals

    coarse_angle = np.linspace(-np.pi / 2, np.pi / 2, 1000)
    integral_vector = [0] * 1000
    counter = 0
    # integration
    for angle in coarse_angle:
        copy_total_spectral_ydata = convolved_y * np.exp(-angle * 1j)
        peak_regions = [0] * len(binary_map_regions)
        for region in range(0, len(binary_map_regions)):
            peak_regions[region] = copy_total_spectral_ydata[
                binary_map_regions[region][0] : binary_map_regions[region][1] : 1
            ]

        integral_vector[counter] = inte(binary_map_regions, peak_regions, corr_distance)
        counter = counter + 1

    # find maximum integral for each region and store angles
    integral_vector = np.array(integral_vector)
    maxvector = np.amin(integral_vector, 0)

    counter = 0
    angle1 = [0] * len(binary_map_regions)

    for element in list(maxvector):
        maxangle = np.where(integral_vector == element)
        angle1[counter] = coarse_angle[maxangle[0][0]]
        counter = counter + 1

    # phase each region of the spectrum indepedently

    for region in range(0, len(peak_regions)):
        convolved_y_phased[
            binary_map_regions[region][0] : binary_map_regions[region][1] : 1
        ] = convolved_y_phased[
            binary_map_regions[region][0] : binary_map_regions[region][1] : 1
        ] * np.exp(
            -angle1[region] * 1j
        )

    globalangles = [angle1[i] for i in range(0, len(binary_map_regions))]

    ## phase each peak region separately

    phased_peak_regions = []
    copy_total_spectral_ydata = convolved_y
    peak_regions = [0] * len(binary_map_regions)
    for region in range(0, len(binary_map_regions)):
        peak_regions[region] = copy_total_spectral_ydata[
            binary_map_regions[region][0] : binary_map_regions[region][1] : 1
        ]
    counter = 0

    for region in peak_regions:
        phased_peak_regions.append(region * np.exp(-globalangles[counter] * 1j))
        counter += 1

    return globalangles, phased_peak_regions, convolved_y_phased


def peak_picking_region(real_convolved_y_phased, picked_points_region):
    picked_peaks_region = []

    for region in range(0, len(picked_points_region)):
        picked_peaks_region.append([])

        for index in picked_points_region[region]:
            peak = real_convolved_y_phased[index]
            if (
                peak > real_convolved_y_phased[index + 1]
                and peak > real_convolved_y_phased[index - 1]
            ):
                picked_peaks_region[-1].append(index)
            elif (
                peak < real_convolved_y_phased[index + 1]
                and peak < real_convolved_y_phased[index - 1]
            ):
                picked_peaks_region[-1].append(index)
    return picked_peaks_region


def linear_regression(
    picked_peaks_region, globalangles, real_convolved_y_phased, binary_map_regions
):

    # region weighting vector

    region_weighting_matrix = [1] * len(picked_peaks_region)

    for index, region in enumerate(picked_peaks_region):
        region_weighting_matrix[index] = max(
            [abs(real_convolved_y_phased[peak]) for peak in region]
        )
    max_weight = max(region_weighting_matrix)

    region_weighting_matrix = [i / max_weight for i in region_weighting_matrix]

    # define centres of regions
    region_centres = []

    for region in binary_map_regions:
        region_centres.append(
            (1 - (region[0] + region[1]) / (2 * len(real_convolved_y_phased)))
        )

    #### regression and outlier analysis

    number_of_outliers = 1
    while number_of_outliers > 0:
        region_centres_regression = sm.add_constant(region_centres)
        wls_model = sm.WLS(
            globalangles, region_centres_regression, weights=region_weighting_matrix
        )
        results = wls_model.fit()
        params = results.params
        predictions = [params[1] * i + params[0] for i in region_centres]
        # remove maximum outlier more than 0.6 rad from estimate
        differences = [
            abs(predictions[angle] - globalangles[angle])
            for angle in range(0, len(globalangles))
        ]

        maxdifference = max(differences)
        if maxdifference > 0.6:
            index = differences.index(maxdifference)
            globalangles.pop(index)
            region_centres.pop(index)
            region_weighting_matrix.pop(index)
        else:
            number_of_outliers = 0

    p0 = params[0]
    p1 = params[1]

    return p0, p1


def final_phasing(total_spectral_ydata, p0, p1):

    # total_spectral_ydata = ng.proc_base.ps(total_spectral_ydata, p0=p0, p1=p1)

    relativeposition = np.linspace(1, 0, len(total_spectral_ydata))

    angle = p0 + p1 * relativeposition

    total_spectral_ydata = total_spectral_ydata * np.exp(-1j * angle)

    total_spectral_ydata = ng.proc_bl.baseline_corrector(total_spectral_ydata, wd=2)

    spectral_ydata = ng.proc_base.di(total_spectral_ydata)  # discard the imaginaries
    spectral_ydata = np.ndarray.tolist(spectral_ydata)
    total_spectral_ydata = np.real(total_spectral_ydata)

    return total_spectral_ydata, spectral_ydata
