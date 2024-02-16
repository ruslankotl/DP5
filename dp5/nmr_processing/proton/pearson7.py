import numpy as np

from dp5.nmr_processing.helper_functions import generalised_lorentzian as p7


def p7residual(params, x, picked_points, y_data, region, differential):
    """
    - params: lmfit Parameters object
    """
    y = np.zeros(len(x))

    for peak in picked_points:
        y += p7(x, params['mu' + str(peak)], params['std' + str(peak)], params['vregion' + str(region)],
                params['A' + str(peak)])

    if differential == True:

        res = abs(y - y_data)

        av = np.average(res)

        av2 = np.average(res ** 2)

        difference = (res ** 2 - av2) - (res - av) ** 2

    else:
        difference = (y - y_data) ** 2

    return difference


def p7residualsolvent(params, x, picked_points, y_data, region, differential):
    y = np.zeros(len(x))

    for peak in picked_points:
        y += p7(x, params['mu' + str(peak)], params['std' + str(peak)], params['vregion' + str(region)],
                params['A' + str(peak)] * params['global_amp'])

    if differential == True:

        dy = np.gradient(y)
        ddy = np.gradient(y)

        dy_ydata = np.gradient(y_data)
        ddy_ydata = np.gradient(dy_ydata)

        difference = (y - y_data) ** 2 + (dy_ydata - dy) ** 2 + (ddy_ydata - ddy) ** 2

    else:
        difference = abs(y - y_data)

    return difference


def p7simsolvent(params, x, picked_points, region):
    y = np.zeros(len(x))

    for peak in picked_points:
        y += p7(x, params['mu' + str(peak)], params['std' + str(peak)], params['vregion' + str(region)],
                params['A' + str(peak)] * params['global_amp'])

    return y


def p7sim(params, x, picked_points, region):
    y = np.zeros(len(x))

    for peak in picked_points:
        y += p7(x, params['mu' + str(peak)], params['std' + str(peak)], params['vregion' + str(region)],
                params['A' + str(peak)])

    return y


def p7plot(params, region, group, ind, xppm):
    region_j = np.zeros(len(region))

    for peak in group:
        j = p7(region, params['mu' + str(peak)], params['std' + str(peak)], params['vregion' + str(ind)],
               params['A' + str(peak)])

        region_j += j

    return region_j