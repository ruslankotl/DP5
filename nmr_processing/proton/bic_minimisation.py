import multiprocessing as mp
import copy

import numpy as np
from lmfit import Minimizer, Parameters

from dp5.nmr_processing.proton.pearson7 import p7sim, p7residual

def multiproc_BIC_minimisation(peak_regions, grouped_peaks, total_spectral_ydata, corr_distance, uc, std):
    maxproc = mp.cpu_count()

    pool = mp.Pool(maxproc)

    new_grouped_peaks = [[] for i in range(len(peak_regions))]
    new_grouped_params = [[] for i in range(len(peak_regions))]
    new_sim_y = [[] for i in range(len(peak_regions))]

    def BIC_minimisation_region_full(ind1, uc, peak_regions, grouped_peaks, total_spectral_ydata, corr_distance, std):

        ################################################################################################################
        # initialise process
        ################################################################################################################

        # print("minimising region " + str(ind1) + " of " + str(len(peak_regions)))

        BIC_param = 15

        region = np.array(peak_regions[ind1])

        region_y = total_spectral_ydata[region]

        fit_y = np.zeros(len(region_y))

        copy_peaks = np.array(grouped_peaks[ind1])

        params = Parameters()

        fitted_peaks = []

        ttotal = 0

        ################################################################################################################
        # build initial model
        ################################################################################################################

        # params.add('vregion' + str(ind1), value=2.5, max=5, min=1)

        params.add('vregion' + str(ind1), value=0.5, max=1, min=0)

        distance = uc(0, "hz") - uc(5, "hz")

        std_upper = uc(0, "hz") - uc(1, "hz")
        av_std = uc(0, "hz") - uc(0.2, "hz")
        std_lower = uc(0, "hz") - uc(0.1, "hz")

        # build model

        while (len(copy_peaks) > 0):
            # pick peak that is furthest from fitted data:

            diff_array = region_y - fit_y

            ind2 = np.argmax(diff_array[copy_peaks - region[0]])

            maxpeak = copy_peaks[ind2]

            copy_peaks = np.delete(copy_peaks, ind2)

            # only allow params < distance away vary at a time

            # add new params

            fitted_peaks.append(maxpeak)

            fitted_peaks = sorted(fitted_peaks)

            params.add('A' + str(maxpeak), value=total_spectral_ydata[maxpeak], min=0, max=1, vary=True)

            # params.add('std' + str(maxpeak), value=av_std, vary=True, min = std_lower,
            #               max = std_upper)

            params.add('std' + str(maxpeak), value=av_std, vary=True)

            params.add('mu' + str(maxpeak), value=maxpeak, vary=True
                       , min=maxpeak - 4 * corr_distance, max=maxpeak + 4 * corr_distance)

            # adjust amplitudes and widths of the current model

        initial_y = p7sim(params, region, fitted_peaks, ind1)

        inty = np.sum(region_y[region_y > 0])

        intmodel = np.sum(initial_y)

        # check the region can be optimised this way

        # find peak with max amplitude

        maxamp = 0

        for peak in fitted_peaks:
            amp = params['A' + str(peak)]
            if amp > maxamp:
                maxamp = copy.copy(amp)

        maxintegral = maxamp * len(region)

        if maxintegral > inty:

            # set initial conditions

            while (intmodel / inty < 0.99) or (intmodel / inty > 1.01):

                for f in fitted_peaks:
                    params['std' + str(f)].set(value=params['std' + str(f)] * inty / intmodel)

                initial_y = p7sim(params, region, fitted_peaks, ind1)

                for f in fitted_peaks:
                    params['A' + str(f)].set(
                        value=params['A' + str(f)] * region_y[int(params['mu' + str(f)]) - region[0]] / (
                            initial_y[f - region[0]]))

                initial_y = p7sim(params, region, fitted_peaks, ind1)

                intmodel = np.sum(initial_y)

        # print('built model region ' + str(ind1))

        ################################################################################################################
        # now relax all params
        ################################################################################################################

        # allow all params to vary

        params['vregion' + str(ind1)].set(vary=True)

        for peak in fitted_peaks:
            params['A' + str(peak)].set(vary=False, min=max(0, params['A' + str(peak)] - 0.01),
                                        max=min(params['A' + str(peak)] + 0.01, 1))
            params['mu' + str(peak)].set(vary=False)
            params['std' + str(peak)].set(vary=False, min=min(std_lower, params['std' + str(peak)] - av_std),
                                          max=max(params['std' + str(peak)] + av_std, std_upper))

        out = Minimizer(p7residual, params,
                        fcn_args=(region, fitted_peaks, region_y, ind1, False))

        results = out.minimize()

        params = results.params

        # print('relaxed params region ' + str(ind1))

        ################################################################################################################
        # now remove peaks in turn
        ################################################################################################################

        trial_y = p7sim(params, region, fitted_peaks, ind1)

        trial_peaks = np.array(fitted_peaks)

        amps = []

        for peak in trial_peaks:
            amps.append(params['A' + str(peak)])

        r = trial_y - region_y

        chi2 = r ** 2

        N = len(chi2)

        BIC = N * np.log(np.sum(chi2) / N) + np.log(N) * (3 * len(fitted_peaks) + 2)

        while (len(trial_peaks) > 0):

            new_params = copy.copy(params)

            # find peak with smallest amp

            minpeak = trial_peaks[np.argmin(amps)]

            # remove this peak from the set left to try

            trial_peaks = np.delete(trial_peaks, np.argmin(amps))
            amps = np.delete(amps, np.argmin(amps))

            # remove this peak from the trial peaks list and the trial params

            new_params.__delitem__('A' + str(minpeak))
            new_params.__delitem__('mu' + str(minpeak))
            new_params.__delitem__('std' + str(minpeak))

            new_fitted_peaks = np.delete(fitted_peaks, np.where(fitted_peaks == minpeak))

            # simulate data with one fewer peak

            new_trial_y = p7sim(new_params, region, new_fitted_peaks, ind1)

            r = new_trial_y - region_y

            chi2 = np.sum(r ** 2)

            N = len(new_trial_y)

            new_BIC = N * np.log(chi2 / N) + np.log(N) * (3 * len(new_fitted_peaks) + 2)

            # if the fit is significantly better remove this peak

            if new_BIC < BIC - BIC_param:
                fitted_peaks = copy.copy(new_fitted_peaks)

                params = copy.copy(new_params)

                BIC = copy.copy(new_BIC)

        fitted_peaks = sorted(fitted_peaks)

        fit_y = p7sim(params, region, fitted_peaks, ind1)

        ################################################################################################################

        return fitted_peaks, params, fit_y

    # order regions by size to efficiently fill cores

    region_lengths = np.array([len(g) for g in grouped_peaks])

    sorted_regions = np.argsort(region_lengths)[::-1]

    res = [[] for i in peak_regions]

    # write output files

    for ind1 in range(len(peak_regions)):
        res[ind1] = pool.apply_async(BIC_minimisation_region_full,
                                     [ind1, uc, peak_regions, grouped_peaks, total_spectral_ydata, corr_distance,
                                      std])

    for ind1 in sorted_regions:
        new_grouped_peaks[ind1], new_grouped_params[ind1], new_sim_y[ind1] = res[ind1].get()

    #### unpack the parameters and split groups

    final_grouped_peaks = []
    final_peak_regions = []
    final_sim_y = []
    final_peaks = []

    total_params = Parameters()

    new_peaks = []

    dist_hz = uc(0, "Hz") - uc(20, "Hz")

    newgroupind = 0
    oldgroupind = 0

    for group in new_grouped_peaks:

        group = sorted(group)

        if len(group) > 0:
            final_grouped_peaks.append([])
            total_params.add('vregion' + str(newgroupind),
                             value=new_grouped_params[oldgroupind]['vregion' + str(oldgroupind)])

            final_peaks.extend(group)

        for ind2, peak in enumerate(group):

            # check if the group should be split

            if ind2 > 0:

                # if there is a gap of more than 20Hz split the group

                if peak > group[ind2 - 1] + dist_hz:
                    # track the numer of splits included to ensure v parameter is added to the correct group each time

                    newgroupind += 1

                    # if a split occures add a new v parameter for the new group

                    total_params.add('vregion' + str(newgroupind),
                                     value=new_grouped_params[oldgroupind]['vregion' + str(oldgroupind)])

                    # allow peaks to be added to the new group
                    final_grouped_peaks.append([])

            # finally append the peak
            final_grouped_peaks[-1].append(peak)
            total_params.add('A' + str(peak), value=new_grouped_params[oldgroupind]['A' + str(peak)])
            total_params.add('std' + str(peak), value=new_grouped_params[oldgroupind]['std' + str(peak)])
            total_params.add('mu' + str(peak), value=new_grouped_params[oldgroupind]['mu' + str(peak)], vary=False)

        if len(group) > 0:
            newgroupind += 1

        oldgroupind += 1

    # draw regions between midpoints of groups

    for ind4, group in enumerate(final_grouped_peaks):

        if ind4 == 0:
            lower_point = 0
            higher_point = int((group[-1] + final_grouped_peaks[ind4 + 1][0]) / 2)

        elif ind4 == len(final_grouped_peaks) - 1:
            lower_point = int((group[0] + final_grouped_peaks[ind4 - 1][-1]) / 2)
            higher_point = len(total_spectral_ydata)

        else:
            lower_point = int((group[0] + final_grouped_peaks[ind4 - 1][-1]) / 2)

            higher_point = int((group[-1] + final_grouped_peaks[ind4 + 1][0]) / 2)

        final_peak_regions.append(np.arange(lower_point, higher_point))

    # now simulate new regions and store region data

    for ind3, region in enumerate(final_peak_regions):
        fit_y = p7sim(total_params, region, final_grouped_peaks[ind3], ind3)

        final_sim_y.append(fit_y)

    return final_peaks, final_grouped_peaks, final_peak_regions, final_sim_y, total_params


