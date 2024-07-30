import pickle
from pathlib import Path

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
        intercept = np.nan_to_num(intercept, nan=0)
        slope = np.nan_to_num(slope, nan=1)
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


class AnalysisData:
    """Container class for DP4 or DP5 analysis."""

    # TODO: incorporate json
    def __init__(self, mols, path):
        self.mols = [mol.input_file for mol in mols]
        self.path = path

    @property
    def exists(self):
        return Path(self.path).exists()

    @property
    def values_dict(self):
        keys_to_exclude = ("path", "exists")
        return {k: v for k, v in self.__dict__.items() if k not in keys_to_exclude}

    def load(self):
        with open(self.path, "rb") as f:
            data = pickle.load(f)
            for key, value in data.items():
                setattr(self, key, value)

    def __iter__(self):
        return (
            dict(zip(self.values_dict.keys(), values))
            for values in zip(*self.values_dict.values())
        )

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.values_dict, f)

    @property
    def by_mol(self):

        return [
            dict(zip(self.values_dict.keys(), values))
            for values in zip(*self.values_dict.values())
        ]

    def append(self, mol_dict):
        """Adds data from a molecule.

        Searches for relevant attributes, appends the data to the class"""
        for key, value in mol_dict.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = [value]
            else:
                self.__dict__[key].append(value)

    def from_mol_dicts(self, dicts):
        """Reads molecular dictionaries containing properties, appends them as class attributes"""
        transposed_dict = dict()
        for dp4d in dicts:
            for key, value in dp4d.items():
                if key in transposed_dict.keys():
                    transposed_dict[key].append(value)
                else:
                    transposed_dict[key] = [value]
        self.__dict__.update(transposed_dict)
