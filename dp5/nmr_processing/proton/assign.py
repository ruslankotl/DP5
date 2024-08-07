import copy

import numpy as np
from scipy.optimize import linear_sum_assignment as optimise
from scipy.stats import linregress, norm

from dp5.nmr_processing.helper_functions import methyl_protons


def iterative_assignment(mol, exp_peaks, calculated_shifts, H_labels, rounded_integrals):
    """
    Arguments:
    mol: rdkit Mol object. Required to store the molecular conectivity
    exp_peaks: experimental signals as processed by NMR-AI
    calculated_shifts: simulated shifts using DFT or NN
    H_labels: atomic labels for H atoms. Must follow SD File numbering
    rounded_integrals: integrals as processed by NMR-AI

    returns:
    - assigned shifts: unused
    - assigned peaks
    - assigned labels
    - sclaed_shifts: unused
    """

    calculated_shifts = np.array(calculated_shifts)

    H_labels = np.array(H_labels)

    lnum = 0

    new_assigned_shifts = []
    old_assigned_shifts = [1]

    while old_assigned_shifts != new_assigned_shifts:

        if lnum == 0:
            # is that scaling necessary? Looks like DFT calcs
            scaled_shifts = external_scale_proton_shifts(calculated_shifts)

            scaled_mu = 0

            scaled_std = 1

        else:
            old_assigned_shifts = copy.copy(new_assigned_shifts)
            old_assigned_peaks = copy.copy(new_assigned_peaks)

            scaled_shifts, slope, intercept = internal_scale_proton_shifts(
                old_assigned_shifts, old_assigned_peaks, calculated_shifts)

            scaled_std = 1

        # assign methyl groups first

        # find methyl groups

        m_protons = methyl_protons(mol)

        m_shifts = np.array([])

        # find the average shifts of these groups

        for m_group in m_protons:

            s = 0

            for proton in m_group:

                w = np.where(H_labels == proton)

                s += scaled_shifts[w]/3

            m_shifts = np.hstack((m_shifts, s))

        # find peaks these can be assigned too

        methyl_peaks = []

        rounded_integrals = np.array(rounded_integrals)

        w = (rounded_integrals - (rounded_integrals % 3)) // 3

        for ind, peak in enumerate(sorted(list(set(exp_peaks)))[::-1]):
            methyl_peaks += [peak] * w[ind]

        # create difference matrix

        diff_matrix = np.zeros((len(m_shifts), len(methyl_peaks)))

        for ind1, i in enumerate(m_shifts):
            for ind2, j in enumerate(methyl_peaks):
                diff_matrix[ind1, ind2] = j-i

        prob_matrix = proton_probabilities(diff_matrix, scaled_mu, scaled_std)

        prob_matrix = prob_matrix**2

        prob_matrix = 1 - prob_matrix

        vertical_ind, horizontal_ind = optimise(prob_matrix)

        # unpack this assignment

        opt_labelsm = []

        opt_shiftsm = []

        opt_peaksm = []

        for j in vertical_ind:

            opt_labelsm.extend(m_protons[j])

        for i in horizontal_ind:

            opt_peaksm += 3*[methyl_peaks[i]]

        for label in opt_labelsm:

            w = np.where(H_labels == label)

            opt_shiftsm.append(calculated_shifts[w][0])

        # remove shifts/peaks/labels for the list to assign

        calculated_shiftsp = copy.copy(calculated_shifts)

        exp_peaksp = copy.copy(exp_peaks)

        scaled_shiftsp = copy.copy(scaled_shifts)

        H_labelsp = copy.copy(H_labels)

        # peaks

        for p in opt_peaksm:

            w = np.where(exp_peaksp == p)[0][0]

            exp_peaksp = np.delete(exp_peaksp, w)

        # shifts

        for s in opt_shiftsm:

            w = np.where(calculated_shiftsp == s)[0][0]

            calculated_shiftsp = np.delete(calculated_shiftsp, w)
            scaled_shiftsp = np.delete(scaled_shiftsp, w)

        # labels

        for l in opt_labelsm:

            w = np.where(H_labelsp == l)[0][0]

            H_labelsp = np.delete(H_labelsp, w)

        # assigned everything else

        diff_matrix = np.zeros((len(calculated_shiftsp), len(exp_peaksp)))

        for ind1, i in enumerate(scaled_shiftsp):
            for ind2, j in enumerate(exp_peaksp):
                diff_matrix[ind1, ind2] = j-i

        prob_matrix = proton_probabilities(diff_matrix, scaled_mu, scaled_std)

        b = abs(diff_matrix) >= 1

        # find any rows that are all zeros

        b = np.where(np.sum(prob_matrix, 1) == 0)

        prob_matrix[b] = - np.inf

        prob_matrix = np.delete(prob_matrix, b, 0)

        unassignable_shifts = calculated_shiftsp[b]

        ccalculated_shiftsp = np.delete(calculated_shiftsp, b)

        ##############################

        prob_matrix = prob_matrix**2

        prob_matrix = 1 - prob_matrix

        vertical_ind, horizontal_ind = optimise(prob_matrix)

        opt_peaksp = exp_peaksp[horizontal_ind]

        opt_shiftsp = ccalculated_shiftsp[vertical_ind]

        opt_labelsp = H_labelsp[vertical_ind]

        opt_shifts, opt_peaks, opt_labels = removecrossassignments(
            opt_peaksp, opt_shiftsp, opt_labelsp)

        # combine these assignments

        opt_peaks = np.hstack((opt_peaksm, opt_peaksp))

        opt_shifts = np.hstack((opt_shiftsm, opt_shiftsp))

        opt_labels = np.hstack((opt_labelsm, opt_labelsp))

        # check for any shifts that have not been assigned

        copyshifts = list(copy.copy(calculated_shifts))
        copylabels = list(copy.copy(H_labels))

        for shift, label in zip(opt_shifts, opt_labels):

            copyshifts.remove(shift)
            copylabels.remove(label)

        # assign these to the closest peaks - regardless of integrals

        for shift, label in zip(copyshifts, copylabels):

            mindiff = np.array(exp_peaks - shift).argmin()

            opt_peaks = np.append(opt_peaks, exp_peaks[mindiff])

            opt_labels = np.append(opt_labels, label)

            opt_shifts = np.append(opt_shifts, shift)

        # sort output wrt original H labels

        indv = []

        for label in opt_labels:

            wh = np.where(H_labels == label)

            indv.append(wh[0][0])

        ind = np.argsort(opt_shifts)[::-1]

        assigned_shifts = opt_shifts[indv]

        assigned_peaks = opt_peaks[indv]

        assigned_labels = opt_labels[indv]

        ind = np.argsort(assigned_shifts)

        assigned_shifts = assigned_shifts[ind].tolist()
        assigned_peaks = assigned_peaks[ind].tolist()
        assigned_labels = assigned_labels[ind].tolist()

        lnum += 1

        new_assigned_shifts = copy.copy(assigned_shifts)
        new_assigned_peaks = copy.copy(assigned_peaks)

    return assigned_shifts, assigned_peaks, assigned_labels, scaled_shifts


def proton_probabilities(diff_matrix, scaled_mu, scaled_std):

    prob_matrix = norm.pdf(diff_matrix, scaled_mu, scaled_std) / \
        norm.pdf(scaled_mu, scaled_mu, scaled_std)

    return prob_matrix


def removecrossassignments(exp, calc, labels):

    # sort these in decending order

    s = np.argsort(calc)[::-1]

    calc = calc[s]

    exp = exp[s]

    labels = labels[s]

    # generate difference matrix
    switch = 0

    expcopy = np.array(exp)

    while switch == 0:

        swapm = np.zeros([len(calc), len(calc)])

        for i, Hi in enumerate(expcopy):
            for j, Hj in enumerate(expcopy):

                if i > j:

                    swapm[i, j] = 0
                else:
                    swapm[i, j] = round(Hi - Hj, 1)

        w = np.argwhere(swapm < 0)

        if len(w > 0):
            expcopy[w[0]] = expcopy[w[0][::-1]]

        else:
            switch = 1

    return calc, expcopy, labels


def external_scale_proton_shifts(calculated_shifts):
    """
    - Calculated_shifts(numpy array)
    may not be true, depends on particular basis set"""
    scaled = 0.9770793502768845 * calculated_shifts - 0.019505417520415236

    return scaled


def internal_scale_proton_shifts(assigned_shifts, assigned_peaks, calculated_shifts):

    slope, intercept, r_value, p_value, std_err = linregress(
        assigned_shifts, assigned_peaks)

    scaled_shifts = calculated_shifts * slope + intercept

    return scaled_shifts, slope, intercept
