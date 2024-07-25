import numpy as np
from scipy.stats import linregress


def _scale_nmr(calc_shifts, exp_shifts):
    """
    Linear correction to calculated shifts
    Arguments:
    - calc_shifts(list): calculated shifts
    - exp_shifts(list): experimental shifts as assigned

    Returns:
    - scaled calculated shifts for error analysis
    """

    if len(calc_shifts) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(
            exp_shifts, calc_shifts
        )
        if not isinstance(calc_shifts, np.ndarray) or isinstance(
            exp_shifts, np.ndarray
        ):
            calc_shifts = np.array(calc_shifts)
            exp_shifts = np.array(exp_shifts)
        scaled_shifts = (calc_shifts - intercept) / slope
    else:
        scaled_shifts = calc_shifts

    return scaled_shifts


def scale_nmr(calc_shifts, exp_shifts):
    if not isinstance(calc_shifts, np.ndarray) or not isinstance(
        exp_shifts, np.ndarray
    ):
        calc_shifts = np.array(calc_shifts)
        exp_shifts = np.array(exp_shifts)

    if len(calc_shifts.shape) > 1:
        scale_axis = lambda conf_shifts: _scale_nmr(conf_shifts, exp_shifts)
        scaled_shifts = np.apply_along_axis(
            scale_axis, 1, np.array(calc_shifts, dtype=np.float32)
        )
    else:
        scaled_shifts = _scale_nmr(calc_shifts, exp_shifts)

    return scaled_shifts
