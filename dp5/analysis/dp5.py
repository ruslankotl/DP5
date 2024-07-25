from pathlib import Path
import logging

import pathos.multiprocessing as mp

from dp5.neural_net.CNN_model import *
from dp5.analysis.utils import scale_nmr

logger = logging.getLogger(__name__)


class DP5:
    def __init__(self, output_folder: Path, use_dft_shifts: bool):
        logger.info("Setting up DP5 method")
        self.output_folder = output_folder
        self.dft_shifts = use_dft_shifts

        if use_dft_shifts:
            # must load model for error prediction
            self.C_DP5 = DP5ProbabilityCalculator(
                atom_type="C",
                model_file="NMRdb-CASCADEset_Error_mean_model_atom_features256.hdf5",
                batch_size=16,
                transform_file="pca_10_ERRORrep_Error_decomp.p",
                kde_file="pca_10_kde_ERRORrep_Error_kernel.p",
                kde_prob_function=error_kde_probfunction,
            )
        else:
            # must load model for shift preiction
            self.C_DP5 = DP5ProbabilityCalculator(
                atom_type="C",
                model_file="NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5",
                batch_size=16,
                transform_file="pca_10_EXP_decomp.p",
                kde_file="pca_10_kde_EXP_kernel.p",
                kde_prob_function=exp_kde_probfunction,
            )

        if not self.output_folder.exists():
            self.output_folder.mkdir()

        if not (self.output_folder / "dp5").exists():
            (self.output_folder / "dp5").mkdir()

        # if exists

    def save(self, file):
        for attr in self.__dict__:
            pass

    def __call__(self, mols):
        # have to generate representations for accepted things
        dp5_labels = []
        self.C_DP5(mols)


def remove_unassigned_data(calculated, experimental, labels):
    """check nmr_ai.py for experimental labelling"""
    new_calcs, new_exps, new_labs = [], [], []
    for calc, exp, lab in zip(calculated, experimental, labels):
        if exp is not None:
            new_calcs.append(calc)
            new_exps.append(exp)
            new_labs.append(lab)
    return new_calcs, new_exps, new_labs


class DP5ProbabilityCalculator:
    def __init__(
        self,
        atom_type,
        model_file,
        batch_size,
        transform_file,
        kde_file,
        kde_prob_function,
    ):
        self.atom_type = atom_type
        self.model = build_model(model_file=model_file)
        self.batch_size = batch_size
        with open(Path(__file__).parent / transform_file, "rb") as tf:
            self.transform = pickle.load(tf)
        with open(Path(__file__).parent / kde_file, "rb") as kf:
            self.kde = pickle.load(kf)
        self.kde_prob_function = kde_prob_function

    def __call__(self, mols):
        # must generate representations
        all_labels = []
        all_shifts = []
        rep_df = []
        for mol_id, mol in enumerate(mols):
            calculated, experimental, labels, indices = self.get_shifts_and_labels(mol)
            # drop unassigned !
            has_exp = experimental != None
            new_calcs = calculated[:, has_exp]
            new_exps = experimental[has_exp]
            new_labs = labels[has_exp]
            new_inds = indices[has_exp]
            new_scaled = scale_nmr(new_calcs, new_exps)

            # there are two modes, error, and experiment!
            all_labels.append(new_labs)
            for conf_id, (conf, conf_shifts) in enumerate(
                zip(mol.rdkit_mols, new_calcs)
            ):
                rep_df.append((mol_id, conf_id, conf, np.array(new_inds)))
                all_shifts.append(conf_shifts)
        rep_df = pd.DataFrame(
            rep_df, columns=["mol_id", "conf_id", "Mol", "atom_index"]
        )
        # now return condensed representations! These are now grouped by conformer
        rep_df["representations"] = extract_representations(
            self.model, rep_df, self.batch_size
        )
        rep_df["representations"] = rep_df["representations"].apply(
            self.transform.transform
        )
        # to generate the objects for kde, must explode the rep_df, concatenate with ??, run_kde
        pass

    def get_shifts_and_labels(self, mol):
        """
        Arguments:
        - self.atom_type
        - mol: Molecule object
        Returns:
        - calculated conformer shifts
        - assigned experimental shifts
        - 0-based indices of relevat atoms
        """
        at = self.atom_type
        conformer_shifts = getattr(mol, "conformer_%s_pred" % at)
        assigned_shifts = getattr(mol, "%s_exp" % at)
        atom_labels = getattr(mol, "%s_labels" % at)
        atom_indices = np.array([int(label[len(at) :]) - 1 for label in atom_labels])

        return conformer_shifts, assigned_shifts, atom_labels, atom_indices


class ErrorDP5ProbabilityCalculator(DP5ProbabilityCalculator):
    def __init__(self, atom_type, model, batch_size, transform_file, kde_file):
        super().__init__(atom_type, model, batch_size, transform_file, kde_file)


class ExpDP5ProbabilityCalculator(DP5ProbabilityCalculator):
    def __init__(self, atom_type, model, batch_size, transform_file, kde_file):
        super().__init__(atom_type, model, batch_size, transform_file, kde_file)


def kde_probabilities(kernel_path):
    with open(kernel_path, "rb") as kp:
        kernel = pickle.load(kp)


def error_kde_probfunction(kernel, conf_shifts, conf_reps, exp_data):
    # loop through atoms in the test molecule - generate kde for all of them.

    # check if this has been calculated

    n_points = 250

    x = np.linspace(-20, 20, n_points)

    ones_ = np.ones(n_points)

    n_comp = 10

    p_s = []

    scaled_shifts = scale_nmr(conf_shifts, exp_data)

    scaled_errors = [shift - exp for shift, exp in zip(scaled_shifts, exp_data)]

    for rep, value in zip(conf_reps, scaled_errors):

        # do kde hybrid part here to yield the atomic probability p

        point = np.vstack((rep.reshape(n_comp, 1) * ones_, x))

        pdf = kernel.pdf(point)

        integral_ = np.sum(pdf)

        if integral_ != 0:
            max_x = x[np.argmax(pdf)]

            low_point = max(-20, max_x - abs(max_x - value))

            high_point = min(20, max_x + abs(max_x - value))

            low_bound = np.argmin(np.abs(x - low_point))

            high_bound = np.argmin(np.abs(x - high_point))

            bound_integral = np.sum(
                pdf[min(low_bound, high_bound) : max(low_bound, high_bound)]
            )

            p_s.append(bound_integral / integral_)

    return p_s


def exp_kde_probfunction(kernel, conf_shifts, conf_reps, exp_data):
    # loop through atoms in the test molecule - generate kde for all of them.

    # check if this has been calculated

    n_points = 250

    x = np.linspace(0, 250, n_points)

    ones_ = np.ones(n_points)

    n_comp = 10

    p_s = []

    for rep, value in zip(conf_reps, exp_data):

        # do kde hybrid part here to yield the atomic probability p

        point = np.vstack((rep.reshape(n_comp, 1) * ones_, x))

        pdf = kernel.pdf(point)

        integral_ = np.sum(pdf)

        if integral_ != 0:

            max_x = x[np.argmax(pdf)]

            low_point = max(0, max_x - abs(max_x - value))

            high_point = min(250, max_x + abs(max_x - value))

            low_bound = np.argmin(np.abs(x - low_point))

            high_bound = np.argmin(np.abs(x - high_point))

            bound_integral = np.sum(
                pdf[min(low_bound, high_bound) : max(low_bound, high_bound)]
            )

            p_s.append(bound_integral / integral_)

        else:

            p_s.append(1)

    return p_s
