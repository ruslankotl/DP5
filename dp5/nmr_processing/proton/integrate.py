import logging

import numpy as np
from scipy.stats import norm, gmean

from dp5.nmr_processing.helper_functions import proton_count, methyl_protons, labile_protons

logger = logging.getLogger(__name__)


def find_integrals(
    mol,
    peak_regions,
    grouped_peaks,
    sim_regions,
    picked_peaks,
    params,
    y_data,
):
    """
    - mol: rdkit Mol object
    - peak_regions: the regions where the peaks are found
    - grouped_peaks: peak grouping for subsequent integration
    - sim_regions: regions to simulate
    - picked_peaks: identified peaks
    - params: params
    - y_data: normalised intensity
    """

    # count number of protons in the file
    structure_protons = proton_count(mol)

    # find number of methyl groups
    m = methyl_protons(mol)
    number_of_methyl_groups = len(m)

    # count the number of labile protons in the structure
    l_protons = labile_protons(mol)

    logger.info(f"number of protons = {structure_protons}")
    logger.info(f"number of labile protons = {l_protons}")

    count = 0

    # allow guesses of number of protons in the spectrum between: structure_protons - l_protons, 2 * structure_protons

    number_vector = np.arange(
        structure_protons - l_protons, 2 * structure_protons)

    scores = np.zeros(len(number_vector))

    integrals = integrate_sim_regions(
        sim_regions, grouped_peaks, peak_regions, y_data, params
    )

    for proton_guess in number_vector:

        norm_integrals = normalise_integration(integrals, proton_guess)

        # remove impurities

        # find number of impurities removed

        # find which should be classed as impurities

        r = sum_round(norm_integrals)

        r = np.array(r)

        impurities = len(r[r < 1])

        norm_integrals = norm_integrals[r > 0.5]

        scores[count] = integral_score(
            norm_integrals, structure_protons, proton_guess, l_protons, impurities
        )

        number_of_methyl_groups_integral = np.sum((r - (r % 3)) // 3)

        if number_of_methyl_groups_integral < number_of_methyl_groups:
            scores[count] = 0

        count += 1

    wh = np.argmax(scores)

    best_fit = number_vector[wh]

    logger.info("the best fit number of protons is " + str(best_fit))

    # normalise using this number

    integrals = normalise_integration(integrals, best_fit)

    (
        grouped_peaks,
        integrals,
        peak_regions,
        picked_peaks_,
        impurities,
        sim_regions,
        rounded_integrals,
    ) = remove_impurities(
        integrals, peak_regions, grouped_peaks, picked_peaks, sim_regions
    )

    integral_sum, cummulative_vectors = integral_add(sim_regions, best_fit)

    total_r = 0

    total = 0

    for i in range(0, len(integrals)):
        total += integrals[i]
        total_r += rounded_integrals[i]

    return (
        peak_regions,
        grouped_peaks,
        sim_regions,
        integral_sum,
        cummulative_vectors,
        rounded_integrals,
        structure_protons,
        best_fit,
        total_r,
    )


def integral_score(integrals, structure_protons, proton_guess, l_protons, impurities):

    r_int = sum_round(integrals)

    r_int = np.array(r_int)

    sum_r = np.sum(r_int)

    new_integrals = []

    for ind, integral in enumerate(integrals):
        new_integrals += [integral] * int(r_int[ind])

    new_integrals = np.array(new_integrals)

    differences = new_integrals % 0.5

    std = structure_protons / 8

    diff = abs(proton_guess - structure_protons)

    probs = 4 * (1 - norm.cdf(differences, loc=0, scale=1 / 16))

    mean = gmean(probs) * (1 - norm.cdf(diff, loc=0, scale=std)) * \
        (1 / 2**impurities)

    # only allow intergrals that are between the expected number and that number - the number of labile protons

    if sum_r < structure_protons - l_protons:
        mean = 0

    return mean


def integral_add(sim_regions, proton_guess):

    integrals = [np.sum(sim) for sim in sim_regions]

    i_sum = np.sum(integrals)

    Inorm = proton_guess / i_sum

    integrals_copy = list(integrals)

    integrals_copy = [0] + integrals_copy

    integral_sum = np.cumsum(np.asarray(integrals_copy))
    integral_sum = integral_sum / np.sum(integrals_copy)

    cummulative_vectors = []

    t_sum = 0
    for region in sim_regions:
        t_sum += np.sum(region)
        cummulative_vectors.append(np.cumsum(region))

    for i, region in enumerate(cummulative_vectors):
        cummulative_vectors[i] = region / t_sum

    return integral_sum, cummulative_vectors


def normalise_integration(integrals, initial_proton_guess):
    i_sum = np.sum(integrals)
    integrals = integrals / i_sum
    norm_integrals = integrals * initial_proton_guess

    return norm_integrals


def integrate_sim_regions(
    sim_regions, grouped_peaks, peak_regions, y_data, params
):
    sim_integrals = []

    for r, group in enumerate(grouped_peaks):

        region_integral = 0

        for peak in group:
            region_integral += (
                params["A" + str(peak)]
                * 0.25
                * np.pi
                * params["std" + str(peak)]
                * ((3**0.5 - 2) * params["vregion" + str(r)] + 2)
            )

        sim_integrals.append(region_integral)

    sim_integrals = np.array(sim_integrals)
    y_integral = np.sum(y_data)
    sim_integral = np.sum(sim_integrals)
    k = sim_integral / y_integral
    integrals = []

    for region in peak_regions:
        integrals.append(np.sum(y_data[region]))

    k = np.array(sim_integrals) / np.array(integrals)

    simr_regions = []

    for region in sim_regions:
        simr_regions.append(np.sum(region))

    integrals = k * integrals

    return integrals


def remove_impurities(
    integrals, peak_regions, grouped_peaks, picked_peaks, sim_regions
):
    '''
    - integrals
    - peak_regions
    - grouped_peaks
    - picked_peaks
    - sim_regions
    currently suffers from ragged arrays and numpy not nadling them: must be fixed upstream
    '''
    # find rounded values

    r = sum_round(integrals)

    r = np.array(r)

    to_remove = np.where(r < 0.5)[0]

    number_of_impurities = len(to_remove)

    peaks_to_remove = []
    for group in to_remove:
        peaks_to_remove.extend(grouped_peaks[group])

    whdel = np.isin(picked_peaks, peaks_to_remove)

    picked_peaks = np.delete(picked_peaks, whdel)

    integrals = np.delete(integrals, to_remove)

    peak_regions = np.delete(peak_regions, to_remove)

    grouped_peaks = np.delete(grouped_peaks, to_remove)

    sim_regions = np.delete(sim_regions, to_remove)

    rounded_integrals = r[r > 0.5]

    return (
        grouped_peaks,
        integrals,
        peak_regions,
        picked_peaks,
        number_of_impurities,
        sim_regions,
        rounded_integrals,
    )


def sum_round(a):
    """
    Round each element in the input list to the nearest integer and adjust some
    values to minimize the error in the sum of the rounded values. Used for proton integration

    Parameters:
    - a (list of float): Input list of numeric values.

    Returns:
    - list of int: A list of integers where each element is the result of rounding
      the corresponding element in the input list to the nearest integer, with some
      adjustments to minimize the error in the sum.
    """

    # Round each element in the input list 'a' to the nearest integer
    rounded = [round(i, 0) for i in a]

    # Calculate the sum of the original list 'a' and the sum of the rounded values
    error = sum(a) - sum(rounded)

    # Calculate the integer part of the 'error' divided by 1
    n = int(round(error / 1))

    # Create a copy of the rounded list
    new = rounded[:]

    # Sort the list of tuples containing the difference between elements in 'a' and 'rounded',
    # along with the index of the element, in descending order.
    # Select the top 'abs(n)' elements from the sorted list.
    for _, i in sorted(((a[i] - rounded[i], i) for i in range(len(a))), reverse=n > 0)[
        : abs(n)
    ]:
        # Adjust the corresponding element in the 'new' list by adding 'n / abs(n)'
        new[i] += n / abs(n)

    # Return the modified list 'new'
    return new


def weighted_region_centres(peak_regions, total_spectral_ydata):
    centres = []

    for region in peak_regions:
        w = total_spectral_ydata[region] ** 2

        wx = region * w

        xbar = np.sum(wx) / np.sum(total_spectral_ydata[region] ** 2)

        centres.append(int(xbar))

    return centres
