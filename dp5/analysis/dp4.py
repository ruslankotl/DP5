"""Carries out DP4 analysis of structural proposals."""

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
    "Handles DP4 analysis"

    def __init__(self, output_folder, dp4_config):
        """Initialise DP4 Analysis object.

        Specify the save path, load relevant statistical models, set up DP4 calculators for each nucleus

        Arguments:
         output_folder(Path): path to the output folder
         dp4_config(dict): must contain keys ``stats_model`` and ``param_file``
        Explanation:
         ``dp4_config['stats_model']`` specifies the model to use.
         ``dp4_config['param_file']`` specifies the Gaussian parameters.
         Data will be saved in the output folder.
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
            logger.warning("DP4 score requires multiple candidate structures.")
        logger.info("Starting DP4 analysis")
        dp4_dicts = DP4Data(mols, self.save_dir / "data_dic.p")
        C_data = []
        H_data = []
        keys = [
            "{0}labels",
            "{0}shifts",
            "{0}scaled",
            "{0}exp",
            "{0}errors",
            "{0}probs",
        ]
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
        dp4_dicts.DP4probs = total_data / total_data.sum()

        logger.info("Saving raw DP4 data")
        dp4_dicts.save()
        return dp4_dicts.output

    def dp4_proton(self, calculated, experimental, labels):
        """Generates unscaled DP4 score for protons in the molecule.
        Arguments:
         calculated: scaled calculated NMR shifts
         experimental: experimental NMR shifts
         labels: atom labels, match the SD File numbering
        returns:
         scaled calculated NMR shifts used in the analysis
         experimental NMR shifts used in the analysis
         atom labels used in the analysis
         atomic errors
         probabilities
         DP4 scores
        """
        return self._dp4(calculated, experimental, labels, self.H_probability)

    def dp4_carbon(self, calculated, experimental, labels):
        """Generates unscaled DP4 score for carbons in the molecule.
        Arguments:
         calculated: scaled calculated NMR shifts
         experimental: experimental NMR shifts
         labels: atom labels, match the SD File numbering
        returns:
         scaled calculated NMR shifts used in the analysis
         experimental NMR shifts used in the analysis
         atom labels used in the analysis
         atomic errors
         probabilities
         DP4 scores
        """
        return self._dp4(calculated, experimental, labels, self.C_probability)

    def _dp4(self, calculated: list, experimental: list, labels: list, probability):
        """Generates unscaled DP4 score for nuclei in the molecule.
        Arguments:
         calculated: scaled calculated NMR shifts
         experimental: experimental NMR shifts
         labels: atom labels, match the SD File numbering
         probability(function): returns the number
        returns:
         scaled calculated NMR shifts used in the analysis
         experimental NMR shifts used in the analysis
         atom labels used in the analysis
         atomic errors
         probabilities
         DP4 scores
        """
        # remove calculated peaks that do not match the signal
        has_exp = np.isfinite(experimental)

        new_calcs = calculated[has_exp]
        new_exps = experimental[has_exp]
        new_labs = labels[has_exp]

        new_scaled = scale_nmr(new_calcs, new_exps)
        errors = new_scaled - new_exps
        probs = probability(errors)
        # take the product of probabilities
        dp4_score = prod(probs, start=1)
        return new_labs, new_calcs, new_scaled, new_exps, errors, probs, dp4_score


class DP4ProbabilityCalculator:
    """Estimates DP4 probability for a given nucleus."""

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


class DP4Data(AnalysisData):
    """Collates DP4 Analysis Data. Saves it as a pickle file and returns text summary for printing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def output(self):
        output_dict = dict()
        output_dict["C_output"] = []
        output_dict["H_output"] = []
        output_dict["CDP4_output"] = []
        output_dict["HDP4_output"] = []
        output_dict["DP4_output"] = []

        for mol, clab, cshift, cscal, cexp, cerr in zip(
            self.mols, self.Clabels, self.Cshifts, self.Cscaled, self.Cexp, self.Cerrors
        ):
            output = f"\nAssigned C NMR shift for {mol}:"
            output += self.print_assignment(clab, cshift, cscal, cexp, cerr)
            output_dict["C_output"].append(output)

        for mol, hlab, hshift, hscal, hexp, herr in zip(
            self.mols, self.Hlabels, self.Hshifts, self.Hscaled, self.Hexp, self.Herrors
        ):
            output = f"\nAssigned H NMR shift for {mol}:"
            output += self.print_assignment(hlab, hshift, hscal, hexp, herr)
            output_dict["H_output"].append(output)
        for mol, hdp4, cdp4, dp4 in zip(
            self.mols, self.HDP4probs, self.CDP4probs, self.DP4probs
        ):
            output_dict["HDP4_output"].append(
                f"Proton DP4 probability for {mol}: {hdp4}"
            )
            output_dict["CDP4_output"].append(
                f"Carbon DP4 probability for {mol}: {cdp4}"
            )
            output_dict["DP4_output"].append(f"DP4 probability for {mol}: {dp4}")

        t_dic = [
            dict(zip(output_dict.keys(), values))
            for values in zip(*output_dict.values())
        ]
        dp4_output = "\n\n".join([mol["C_output"] + mol["H_output"] for mol in t_dic])
        dp4_output += "\n\n"
        dp4_output += "\n".join([mol["HDP4_output"] for mol in t_dic])
        dp4_output += "\n\n"
        dp4_output += "\n".join([mol["CDP4_output"] for mol in t_dic])
        dp4_output += "\n\n"
        dp4_output += "\n".join([mol["DP4_output"] for mol in t_dic])

        return dp4_output

    @staticmethod
    def print_assignment(labels, calculated, scaled, exp, error):
        """Prints table for molecule"""

        s = np.argsort(calculated)
        svalues = calculated[s]
        slabels = labels[s]
        sscaled = scaled[s]
        sexp = exp[s]
        serror = error[s]

        output = f"\nlabel, calc, corrected, exp, error"

        for lab, calc, scal, ex, er in zip(slabels, svalues, sscaled, sexp, serror):
            output += f"\n{lab:6s} {calc:6.2f} {scal:6.2f} {ex:6.2f} {er:6.2f}"
        return output
