import logging

import numpy as np
from lmfit import Minimizer, Parameters
from scipy.interpolate import InterpolatedUnivariateSpline

from dp5.nmr_processing.helper_functions import normalise_intensities


logger = logging.getLogger(__name__)


def spectral_processing(total_spectral_ydata):
    """Arguments:
    - ydata: complex array with frequency-domain FID data
    Returns:
    - tydata (np.array): phased and baseline-corrected frequency data
    – corr_distance (int): autocorrelation distance
    – sigma (float): standard deviation of noise
    – peak_regions(np.array[np.array]): regoins containing peak signals
    """
    logger.info("Processing Proton Spectrum")

    corr_distance = estimate_autocorrelation(total_spectral_ydata)
    # baseline and phasing
    tydata = ACMEWLRhybrid(total_spectral_ydata, corr_distance)

    # find final noise distribution
    classification, sigma = baseline_find_signal(
        tydata, corr_distance, True, 1)

    # fall back phasing if fit doesnt converge
    # calculate negative area
    # draw regions

    c1 = np.roll(classification, 1)
    diff = classification - c1
    s_start = np.where(diff == 1)[0]
    s_end = np.where(diff == -1)[0] - 1

    # region start and end indices are known
    # indices inbetween are filled in
    peak_regions = np.empty(shape=s_start.shape[0], dtype=object)
    peak_regions[:] = [np.arange(s, e) for s, e in zip(s_start, s_end)]

    tydata = normalise_intensities(tydata)

    return tydata, corr_distance, sigma, peak_regions


def estimate_autocorrelation(total_spectral_ydata):
    # note this region may have a baseline distortion

    y = np.real(total_spectral_ydata[0:10000])

    params = Parameters()

    # define a basleine polynomial

    order = 6

    for p in range(order + 1):
        params.add("p" + str(p), value=0)

    def poly(params, order, y):

        bl = np.zeros(len(y))
        x = np.arange(len(y))

        for p in range(order + 1):
            bl += params["p" + str(p)] * x ** (p)

        return bl

    def res(params, order, y):

        bl = poly(params, order, y)

        r = abs(y - bl)

        return r

    out = Minimizer(res, params, fcn_args=(order, y))

    results = out.minimize()

    bl = poly(results.params, order, y)

    y = y - bl

    t0 = np.sum(y * y)

    c = 1

    tc = 1

    t = []

    while tc > 0.36:
        tc = np.sum(np.roll(y, c) * y) / t0

        t.append(tc)

        c += 1

    return c


def ACMEWLRhybrid(y, corr_distance):
    """Hybrid ACME-Weighted Linear Regression phase correction algorithm"""

    def residual_function(params, im, real):

        # phase the region

        data = ps(params, im, real, 0)

        # make new baseline for this region

        r = np.linspace(data[0], data[-1], len(real))

        # find negative area

        data -= r

        ds1 = np.abs((data[1:] - data[:-1]))

        p1 = ds1 / np.sum(ds1)

        # Calculation of entropy
        p1[p1 == 0] = 1

        h1 = -p1 * np.log(p1)
        h1s = np.sum(h1)

        # Calculation of penalty
        pfun = 0.0

        as_ = data - np.abs(data)

        sumas = np.sum(as_)

        if sumas < 0:
            pfun = (as_[1:] / 2) ** 2

        p = np.sum(pfun)

        return h1s + 1000 * p

    # find regions

    classification, sigma = baseline_find_signal(y, corr_distance, True, 1)

    c1 = np.roll(classification, 1)

    diff = classification - c1

    s_start = np.where(diff == 1)[0]

    s_end = np.where(diff == -1)[0] - 1

    peak_regions = []

    for r in range(len(s_start)):
        peak_regions.append(np.arange(s_start[r], s_end[r]))

    # for region in peak_regions:
    #    plt.plot(region,y[region],color = 'C1')

    # phase each region independently

    phase_angles = []

    weights = []

    centres = []

    for region in peak_regions:
        params = Parameters()

        params.add("p0", value=0, min=-np.pi, max=np.pi)

        out = Minimizer(
            residual_function, params, fcn_args=(
                np.imag(y[region]), np.real(y[region]))
        )

        results = out.minimize("brute")

        p = results.params

        phase_angles.append(p["p0"] * 1)

        # find weight

        data = ps(p, np.imag(y[region]), np.real(y[region]), 0)

        # make new baseline for this region

        r = np.linspace(data[0], data[-1], len(data))

        # find negative area

        res = data - r

        weights.append(abs(np.sum(res[res > 0] / np.sum(y[y > 0]))))

        centres.append(np.median(region) / len(y))

    sw = sum(weights)

    weights = [w / sw for w in weights]

    # do weighted linear regression on the regions

    # do outlier analysis

    switch = 0

    centres = np.array(centres)

    weights = np.array(weights)

    sweights = np.argsort(weights)[::-1]

    phase_angles = np.array(phase_angles)

    ind1 = 0

    while switch == 0:

        intercept, gradient = np.polynomial.polynomial.polyfit(
            centres, phase_angles, deg=1, w=weights
        )

        predicted_angles = gradient * centres + intercept

        weighted_res = np.abs(predicted_angles - phase_angles) * weights

        # find where largest weighted residual is

        max_res = sweights[ind1]

        s = 0

        if phase_angles[max_res] > 0:

            s = -1

            phase_angles[max_res] -= 2 * np.pi

        else:

            s = +1

            phase_angles[max_res] += 2 * np.pi

        intercept1, gradient1 = np.polynomial.polynomial.polyfit(
            centres, phase_angles, deg=1, w=weights
        )

        new_predicted_angles = gradient1 * centres + intercept1

        new_weighted_res = np.abs(
            new_predicted_angles - phase_angles) * weights

        if np.sum(new_weighted_res) > np.sum(weighted_res):

            switch = 1

            phase_angles[max_res] += -2 * np.pi * s

        ind1 += 1

    # phase the data

    p_final = Parameters()

    p_final.add("p0", value=intercept)
    p_final.add("p1", value=gradient)

    y = ps(p_final, np.imag(y), np.real(y), 1)

    classification, sigma = baseline_find_signal(y, corr_distance, True, 1)
    r = gen_baseline(np.real(y), classification, corr_distance)
    y -= r

    return np.real(y)


def gen_baseline(y_data, sn_vector, corr_distance):
    points = np.arange(len(y_data))

    spl = InterpolatedUnivariateSpline(
        points[sn_vector == 0], y_data[sn_vector == 0], k=1
    )

    r = spl(points)

    # is corr distance odd or even

    if corr_distance % 2 == 0:
        kernel = np.ones((corr_distance + 1) * 10) / ((corr_distance + 1) * 10)
    else:
        kernel = np.ones((corr_distance) * 10) / ((corr_distance) * 10)

    r = np.convolve(r, kernel, mode="same")

    return r


def ps(param, im, real, phase_order):
    x = np.linspace(0, 1, len(real))

    angle = np.zeros(len(x))

    for p in range(phase_order + 1):
        angle += param["p" + str(p)] * x ** (p)

    # phase the data

    R = real * np.cos(angle) - im * np.sin(angle)

    return R


def baseline_find_signal(y_data, cdist, dev, t):
    wd = int(cdist) * 10

    sd_all = _get_sd(y_data, wd)

    snvectort = np.zeros(len(y_data))

    sv = []

    for i in range(0, 4 * cdist):
        x = np.arange(i + wd, len(y_data) - wd, 4 * cdist)

        sample = y_data[x]

        sd_set = _get_sd(sample, wd)

        s = _find_noise_sd(sd_set, 0.999)

        sv.append(s)

    sigma = np.mean(sv)

    b = np.linspace(-0.001, 0.001, 1000)

    if dev == True:
        w = np.where(sd_all > t * sigma)[0]

    else:
        w = np.where(y_data > t * sigma)[0]

    snvectort[w] = 1

    sn_vector = np.zeros(len(y_data))

    w = cdist

    for i in np.arange(len(sn_vector)):
        if snvectort[i] == 1:
            sn_vector[np.maximum(0, i - w): np.minimum(i + w, len(sn_vector))] = 1

    return sn_vector, sigma


def _rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _get_sd(data, k):
    return np.std(_rolling_window(data, k), -1)


def _find_noise_sd(sd_set, ratio):
    """Calculate the median m1 from SDset. exclude the elements greater
    than 2m1from SDset and recalculate the median m2. Repeat until
    m2/m1 converge(sd_set)"""
    m1 = np.median(sd_set)
    S = sd_set <= 2.0 * m1
    tmp = S * sd_set
    sd_set = tmp[tmp != 0]
    m2 = np.median(sd_set)
    while m2 / m1 < ratio:
        m1 = np.median(sd_set)
        S = sd_set <= 2.0 * m1
        tmp = S * sd_set
        sd_set = tmp[tmp != 0]

    return m2
