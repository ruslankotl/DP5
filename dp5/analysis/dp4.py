import logging
from math import prod
from pathlib import Path
import pickle
import json

import numpy as np
from scipy import stats

from .utils import scale_nmr, AnalysisData

logger = logging.getLogger(__name__)


class DP4:
    def __init__(self, output_folder, dp4_config):
        """Load statistical model first
        Arguments:
        - output_folder(Path): path to the output folder
        - dp4_config(dict): must contain keys `stats_model` and `param_file`
        Explanation:
        - `dp4_config['stats_model']` specifies the model to use
        - `dp4_config['param_file']` specifies the Gaussian parameters
        will save data to a file
        """

        save_dir = Path(output_folder) / "dp4"
        save_dir.mkdir(exist_ok=True)
        self.save_dir = save_dir

        # define default statictical parameters
        meanC = 0.0
        meanH = 0.0
        stdevC = 2.269372270818724
        stdevH = 0.18731058105269952

        stats_model = dp4_config["stats_model"]
        param_file = dp4_config["param_file"]

        if stats_model in ("g", "m"):
            logger.info("Statistical model parameter file: %s" % param_file)
            if param_file in ("none", ""):
                logger.info("Using default statistical model")

            else:
                logger.info("Reading DP4 parameters from %s" % str(param_file))
                meanC, stdevC, meanH, stdevH = self.read_parameters(param_file)
        else:
            logger.error("Statistical model not recognised: %s" % stats_model)
            logger.info("Using default statistical model")
        self.H_probability = DP4ProbabilityCalculator(mean=meanH, stdev=stdevH)
        self.C_probability = DP4ProbabilityCalculator(mean=meanC, stdev=stdevC)

    def read_parameters(self, file):
        """Read distribution parameters from the external file
        lines of the file: comment, carbon mean errors, carbon standard deviations,
        proton mean errors, proton standard deviations
        Arguments:
        - file(str or Path): filename
        """
        with open(file, "r") as f:
            inp = f.readlines()

        Cmeans = [float(x) for x in inp[1].split(",")]
        Cstdevs = [float(x) for x in inp[2].split(",")]
        Hmeans = [float(x) for x in inp[3].split(",")]
        Hstdevs = [float(x) for x in inp[4].split(",")]
        # necessary to prevent failure at evaluation stage
        if len(Cmeans) != len(Cstdevs) or len(Hmeans) != len(Hstdevs):
            logger.critical(
                "Number of parameters does not match. Please check the file."
            )
            raise ValueError(
                "The DP4 parameters file could not be read correctly. Please check the file"
            )

        return Cmeans, Cstdevs, Hmeans, Hstdevs

    def __call__(self, mols):
        """Compute DP4 probability for the molecules
        return dictionary containing the data?
        """
        if len(mols) < 2:
            logger.warn("DP4 score requires multiple candidate structures.")
        logger.info("Starting DP4 analysis")
        dp4_dicts = AnalysisData(self.save_dir / "data_dic.p")
        C_data = []
        H_data = []
        keys = ["{0}shifts", "{0}exp", "{0}labels", "{0}errors", "{0}probs"]
        H_keys = [i.format("H") for i in keys]
        C_keys = [i.format("C") for i in keys]
        for mol in mols:
            mol_dict = dict()

            *H_metadata, H_dp4 = self.dp4_proton(mol.H_shifts, mol.H_exp, mol.H_labels)
            H_dict = {k: v for k, v in zip(H_keys, H_metadata)}
            mol_dict.update(H_dict)
            H_data.append(H_dp4)

            *C_metadata, C_dp4 = self.dp4_carbon(mol.C_shifts, mol.C_exp, mol.C_labels)
            C_dict = {k: v for k, v in zip(C_keys, C_metadata)}
            mol_dict.update(C_dict)
            C_data.append(C_dp4)

            dp4_dicts.append(mol_dict)

        C_data = np.array(C_data)
        H_data = np.array(H_data)
        total_data = C_data * H_data

        dp4_dicts.CDP4probs = C_data / C_data.sum()
        dp4_dicts.HDP4probs = H_data / H_data.sum()
        dp4_dicts.DP4Probs = total_data / total_data.sum()

        logger.info("Saving raw DP4 data")
        dp4_dicts.save()
        return dp4_dicts.by_mol

    def dp4_proton(self, calculated, experimental, labels):
        """Generates unscaled DP4 score for protons in the molecule.
        Arguments:
        - calculated: scaled calculated NMR shifts
        - experimental: experimental NMR shifts
        - labels: atom labels, match the SD File numbering
        returns:
        - scaled calculated NMR shifts used in the analysis
        - experimental NMR shifts used in the analysis
        - atom labels used in the analysis
        - atomic errors
        - probabilities
        - DP4 scores
        """
        return self._dp4(calculated, experimental, labels, self.H_probability)

    def dp4_carbon(self, calculated, experimental, labels):
        """Generates unscaled DP4 score for carbons in the molecule.
        Arguments:
        - calculated: scaled calculated NMR shifts
        - experimental: experimental NMR shifts
        - labels: atom labels, match the SD File numbering
        returns:
        - scaled calculated NMR shifts used in the analysis
        - experimental NMR shifts used in the analysis
        - atom labels used in the analysis
        - atomic errors
        - probabilities
        - DP4 scores
        """
        return self._dp4(calculated, experimental, labels, self.C_probability)

    def _dp4(self, calculated: list, experimental: list, labels: list, probability):
        """Generates unscaled DP4 score for nuclei in the molecule.
        Arguments:
        - calculated: scaled calculated NMR shifts
        - experimental: experimental NMR shifts
        - labels: atom labels, match the SD File numbering
        - probability(function): returns the number
        returns:
        - scaled calculated NMR shifts used in the analysis
        - experimental NMR shifts used in the analysis
        - atom labels used in the analysis
        - atomic errors
        - probabilities
        - DP4 scores
        """
        # remove calculated peaks that do not match the signal
        has_exp = experimental != None

        new_calcs = calculated[has_exp]
        new_exps = experimental[has_exp]
        new_labs = labels[has_exp]

        new_calcs = scale_nmr(new_calcs, new_exps)
        errors = new_calcs - new_exps
        probs = probability(errors)
        # take the product of probabilities
        dp4_score = prod(probs, start=1)
        return new_calcs, new_exps, new_labs, errors, probs, dp4_score


class DP4ProbabilityCalculator:
    def __init__(self, mean, stdev):
        if isinstance(mean, list) and isinstance(stdev, list):
            # the check is redundant
            # see assert statement in DP4.read_parameters
            if len(mean) != len(stdev):
                raise ValueError(
                    "Dimensions of mean and standard deviation do not match"
                )
            if len(mean) > 1:
                self.probability = lambda error: self.multiple_gaussian(
                    error, mean, stdev
                )
            else:
                self.probability = lambda error: self.single_gaussian(
                    error, mean[0], stdev[0]
                )
        elif isinstance(mean, float) and isinstance(stdev, float):
            self.probability = lambda error: self.single_gaussian(error, mean, stdev)
        else:
            raise TypeError("Types of mean and standard deviation do not match!")

    def __call__(self, error):
        # numpy is weird about arrays out of single floats
        dp4_score = np.apply_along_axis(
            self.probability, 0, np.array(error, dtype=np.float32)
        )
        return dp4_score

    @staticmethod
    def single_gaussian(error, mean, stdev):
        z = abs((error - mean) / stdev)
        cdp4 = 2 * stats.norm.cdf(-z)

        return cdp4

    @staticmethod
    def multiple_gaussian(error, means, stdevs):
        res = 0
        for mean, stdev in zip(means, stdevs):
            res += stats.norm(mean, stdev).pdf(error)

        return res / len(means)
