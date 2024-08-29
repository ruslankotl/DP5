"""Runs DP5 analysis for organic molecules."""

from pathlib import Path
import logging
import dill as pickle
from abc import abstractmethod

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats.kde import gaussian_kde
from sklearn.neighbors import KernelDensity

from dp5.neural_net.CNN_model import *
from dp5.analysis.utils import scale_nmr, AnalysisData

logger = logging.getLogger(__name__)


class DP5:
    """Performs DP5 analysis"""

    def __init__(self, output_folder: Path, use_dft_shifts: bool):
        """Initialise the settings.

        Arguments:
          output_folder(Path): path for saved DP5 data
          use_dft_shifts(bool): if set, analyses errors of DFT calculations, compares shifts againt their environments otherwise.

        """
        logger.info("Setting up DP5 method")
        self.output_folder = output_folder
        self.dft_shifts = use_dft_shifts

        if use_dft_shifts:
            # must load model for error prediction
            self.C_DP5 = ErrorDP5ProbabilityCalculator(
                atom_type="C",
                model_file="NMRdb-CASCADEset_Error_mean_model_atom_features256.hdf5",
                batch_size=16,
                transform_file="pca_10_ERRORrep_Error_decomp.p",
                kde_file="pca_10_kde_ERRORrep_Error_kernel.p",
                dp5_correct_scaling="Error_correct_kde.p",
                dp5_incorrect_scaling="Error_incorrect_kde.p",
            )
        else:
            # must load model for shift preiction
            self.C_DP5 = ExpDP5ProbabilityCalculator(
                atom_type="C",
                model_file="NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5",
                batch_size=16,
                transform_file="pca_10_EXP_decomp.p",
                kde_file="pca_10_EXP_sklearn_kde.pkl",
                dp5_correct_scaling=None,
                dp5_incorrect_scaling=None,
            )

        if not self.output_folder.exists():
            self.output_folder.mkdir()

        if not (self.output_folder / "dp5").exists():
            (self.output_folder / "dp5").mkdir()

    def __call__(self, mols):
        """Runs DP5 calculations.

        Arguments:
          mols: Molecule objects
        """
        data_dic_path = self.output_folder / "dp5" / "data_dic.p"
        dp5_data = DP5Data(mols, data_dic_path)
        if dp5_data.exists:
            logger.info("Found existing DP5 probability file")
            dp5_data.load()
        else:
            logger.info("Calculating DP5 probabilites...")
            (
                dp5_data.Clabels,
                dp5_data.Cshifts,
                dp5_data.Cexp,
                dp5_data.Cerrors,
                dp5_data.Cconf_atom_probs,
                dp5_data.CDP5_atom_probs,
                dp5_data.CDP5_mol_probs,
            ) = self.C_DP5(mols)
            dp5_data.save()
        return dp5_data.output


class DP5ProbabilityCalculator:
    def __init__(
        self,
        atom_type,
        model_file,
        batch_size,
        transform_file,
        kde_file,
        dp5_correct_scaling=None,
        dp5_incorrect_scaling=None,
    ):
        """Initialises DP5 Probability calculator for one atom.

        Arguments:
          atom_type (str): atom symbol. Can be 'C' or 'H'
          model_file (str): path for representation generating model to load.
          batch_size (int): batch size for the model.
          transform_file (str): Path to the Scikit-learn PCA file relative to ``dp5/analysis`` folder. Reduces dimensionality of the representation.
          kde_file (str): Path to :py:obj:`scipy.stats.gaussian_kde` or :py:obj:`sklearn.neighbors.KernelDensity` object. Estimates DP5 probabilities
          dp5_correct_scaling (str). Path to :py:obj:`scipy.stats.gaussian_kde` or :py:obj:`sklearn.neighbors.KernelDensity` object. Estimates :math:`P(correct|structure)` for rescaling. Default is None (no scaling)
          dp5_incorrect_scaling (str). Path to :py:obj:`scipy.stats.gaussian_kde` or :py:obj:`sklearn.neighbors.KernelDensity` object. Estimates :math:`P(incorrect|structure)` for rescaling. Default is None (no scaling)

        """
        self.atom_type = atom_type
        self.model = build_model(model_file=model_file)
        self.batch_size = batch_size
        self.transform = _load_pickle(transform_file)
        self.kde = KernelDensityEstimator(kde_file)
        if dp5_correct_scaling is not None:
            self.dp5_correct_kde = KernelDensityEstimator(dp5_correct_scaling)
        if dp5_incorrect_scaling is not None:
            self.dp5_incorrect_kde = KernelDensityEstimator(dp5_incorrect_scaling)

    @abstractmethod
    def rescale_probabilities(self, mol_probs, errors, error_threshold=2):
        """
        Scales and aggregated atomic probabilities.

        Computes geometric means of atomic probabilities to generate final molecular probabilities.
        """
        total_probs = np.array([np.exp(np.log(arr).mean()) for arr in mol_probs])
        return mol_probs, total_probs

    @abstractmethod
    def kde_probfunction(self, df):
        return NotImplementedError("KDE sampling function not implemented")

    @staticmethod
    def boltzmann_weight(df, col):
        return df.groupby("mol_id")[["conf_population", col]].apply(
            lambda x: (x[col] * x["conf_population"]).sum()
        )

    def __call__(self, mols):
        """Carries out DP5 analysis.

        Arguments:
          mols(list of :py:class:`~dp5.run.data_structures.Molecule`): :py:class:`dp5.run.data_structures.Molecule` objects used in the calculation. Must contain shifts and labels for the provided atom.

        Returns:
          A tuple containing lists of labels of atoms used in the analysis,
          their calculated shifts, their experimental shifts,
          scaled errors, DP5 probabilities for each atom in each conformer,
          Boltzmann-weighted atom DP5 probabilites, and total molecular DP5 probabilities.
        """
        all_labels = []
        rep_df = []
        for mol_id, mol in enumerate(mols):
            calculated, experimental, labels, indices = self.get_shifts_and_labels(mol)
            # drop unassigned !
            has_exp = np.isfinite(experimental)
            new_calcs = calculated[:, has_exp]
            new_exps = experimental[has_exp]
            new_labs = labels[has_exp]
            new_inds = indices[has_exp]

            # generate scaled errors
            scaled = scale_nmr(new_calcs, new_exps)
            corrected_errors = scaled - new_exps[np.newaxis, :]

            all_labels.append(new_labs)

            rep_df.append(
                (
                    mol_id,
                    range(mol.conformers.shape[0]),
                    mol.rdkit_mols,
                    new_inds,
                    mol.populations,
                    new_calcs,
                    new_exps,
                    corrected_errors,
                )
            )

        rep_df = pd.DataFrame(
            rep_df,
            columns=[
                "mol_id",
                "conf_id",
                "Mol",
                "atom_index",
                "conf_population",
                "conf_shifts",
                "exp_shifts",
                "errors",
            ],
        )
        # each row of dataframe represents a geometry
        rep_df = rep_df.explode(
            ["conf_id", "Mol", "conf_shifts", "conf_population", "errors"],
            ignore_index=True,
        )
        logger.info("Extracting atomic representations")
        # now return condensed representations! These are now grouped by conformer
        rep_df["representations"] = extract_representations(
            self.model, rep_df, self.batch_size
        )
        logger.debug("Transforming representations")
        rep_df["representations"] = rep_df["representations"].apply(
            self.transform.transform
        )
        logger.info("Estimating atomic probabilities")
        rep_df["atom_probs"] = self.kde_probfunction(rep_df)
        atom_probs = [np.stack(df) for i, df in rep_df.groupby("mol_id")["atom_probs"]]

        weighted_probs = self.boltzmann_weight(rep_df, "atom_probs")
        weighted_probs = 1 - weighted_probs

        weighted_errors = self.boltzmann_weight(rep_df, "errors")
        cmae = weighted_errors.apply(lambda x: np.mean(np.abs(x)))

        # rescale and aggregate probabilities
        weighted_probs, total_probs = self.rescale_probabilities(weighted_probs, cmae)

        calc_shifts_analysed = self.boltzmann_weight(rep_df, "conf_shifts")
        exp_shifts_analysed = rep_df.groupby("mol_id")["exp_shifts"].first()

        # eventually return atomic probs, weighted atomic probs, DP5 scores
        logger.info("Atomic probabilities estimated")
        return (
            all_labels,
            calc_shifts_analysed,
            exp_shifts_analysed,
            weighted_errors,
            atom_probs,
            weighted_probs,
            total_probs,
        )

    def get_shifts_and_labels(self, mol):
        """
        Returns calculated and experimental shifts for nuclei in the molecule.

        Arguments:
          self.atom_type(str): nuclei being analysed
          mols(:py:class:`~dp5.run.data_structures.Molecule`): :py:class:`dp5.run.data_structures.Molecule`. Must contain shifts and labels for the provided atom.

        Returns:
          calculated conformer shifts
          assigned experimental shifts
          0-based indices of relevant atoms
        """
        at = self.atom_type
        conformer_shifts = getattr(mol, "conformer_%s_pred" % at)
        assigned_shifts = getattr(mol, "%s_exp" % at)
        atom_labels = getattr(mol, "%s_labels" % at)
        atom_indices = np.array([int(label[len(at) :]) - 1 for label in atom_labels])

        return conformer_shifts, assigned_shifts, atom_labels, atom_indices


class ErrorDP5ProbabilityCalculator(DP5ProbabilityCalculator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def kde_probfunction(self, df):
        # loop through atoms in the test molecule - generate kde for all of them.
        # implement joblib parallel search
        # check if this has been calculated

        min_value = -20
        max_value = 20
        n_points = 250

        x = np.linspace(min_value, max_value, n_points)

        probs = []
        with Parallel(prefer="threads", n_jobs=-1) as pool:
            for i, (rep, errors) in tqdm(
                df[["representations", "errors"]].iterrows(),
                total=len(df),
                desc="Computing error KDEs",
                leave=True,
            ):
                num_atoms, num_components = rep.shape
                rep_b = np.broadcast_to(
                    rep[:, :, np.newaxis], shape=(num_atoms, num_components, n_points)
                )
                x_b = np.broadcast_to(
                    x[np.newaxis, np.newaxis, :], shape=(num_atoms, 1, n_points)
                )
                point = np.concatenate((rep_b, x_b), axis=1)

                results = pool(delayed(self.kde)(atom) for atom in point[:])

                conf_probs = []
                for pdf, error in zip(results, errors):
                    integral = 0
                    if pdf.sum() != 0:
                        max_x = x[np.argmax(pdf)]

                        low_point = max(min_value, max_x - abs(max_x - error))
                        high_point = min(max_value, max_x + abs(max_x - error))

                        low_bound = np.argmin(np.abs(x - low_point))
                        high_bound = np.argmin(np.abs(x - high_point))

                        bound_integral = np.sum(
                            pdf[min(low_bound, high_bound) : max(low_bound, high_bound)]
                        )
                        integral = bound_integral / pdf.sum()
                    conf_probs.append(integral)
                probs.append(np.array(conf_probs))

        return probs

    def rescale_probabilities(self, mol_probs, errors, error_threshold=2):
        _, total_probs = super().rescale_probabilities(mol_probs, errors)
        scaled_probs = []
        scaled_total = []
        for prob, error, total in zip(mol_probs, errors, total_probs):
            if error < error_threshold:
                vector = np.concatenate((prob, np.atleast_1d(total)))
                correct = self.dp5_correct_kde(vector)
                incorrect = self.dp5_incorrect_kde(vector)
                scaled = correct / (correct + incorrect)
                scaled_probs.append(scaled[:-1])
                scaled_total.append(scaled[-1])
            else:
                scaled_probs.append(prob)
                scaled_total.append(total)
        return scaled_probs, scaled_total


class ExpDP5ProbabilityCalculator(DP5ProbabilityCalculator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def kde_probfunction(self, df):
        """Since the result is compared to the experimental shifts, weights the representations and runs KDE on those."""
        # loop through atoms in the test molecule - generate kde for all of them.
        total_reps = self.boltzmann_weight(df, "representations")
        exp_data = df.groupby("mol_id")["exp_shifts"].first()
        mol_df = pd.DataFrame({"representations": total_reps, "exp_shifts": exp_data})

        min_value = 0
        max_value = 250
        n_points = 250

        x = np.linspace(min_value, max_value, n_points)

        mol_probs = []
        with Parallel(prefer="threads", n_jobs=-1) as pool:
            for i, (rep, exp) in tqdm(
                mol_df[["representations", "exp_shifts"]].iterrows(),
                total=len(mol_df),
                desc="Computing experimental KDEs",
                leave=True,
            ):
                num_atoms, num_components = rep.shape
                rep_b = np.broadcast_to(
                    rep[:, :, np.newaxis], shape=(num_atoms, num_components, n_points)
                )
                x_b = np.broadcast_to(
                    x[np.newaxis, np.newaxis, :], shape=(num_atoms, 1, n_points)
                )
                point = np.concatenate((rep_b, x_b), axis=1)

                results = pool(delayed(self.kde)(atom) for atom in point[:])

                conf_probs = []
                for pdf, value in zip(results, exp):
                    integral = 0
                    if pdf.sum() != 0:
                        max_x = x[np.argmax(pdf)]

                        low_point = max(min_value, max_x - abs(max_x - value))
                        high_point = min(max_value, max_x + abs(max_x - value))

                        low_bound = np.argmin(np.abs(x - low_point))
                        high_bound = np.argmin(np.abs(x - high_point))

                        bound_integral = np.sum(
                            pdf[min(low_bound, high_bound) : max(low_bound, high_bound)]
                        )
                        integral = bound_integral / pdf.sum()
                    conf_probs.append(integral)
                mol_probs.append(np.array(conf_probs))
        consistency_hack = {i: probs for i, probs in enumerate(mol_probs)}
        consistent_probs = df["mol_id"].map(consistency_hack)
        return consistent_probs

    def rescale_probabilities(self, *args, **kwargs):
        return super().rescale_probabilities(*args, **kwargs)


class KernelDensityEstimator:
    def __init__(self, path_to_pickle):
        self.estimator = _load_pickle(path_to_pickle)
        if type(self.estimator) is gaussian_kde:
            self.evaluate = self._scipy_estimator
        elif type(self.estimator) is KernelDensity:
            self.evaluate = self._sklearn_estimator

    def __call__(self, data):
        return self.evaluate(data)

    def _scipy_estimator(self, data):
        return self.estimator(data)

    def _sklearn_estimator(self, data):
        return self.estimator.score_samples(data.T)


class DP5Data(AnalysisData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def output(self):
        """Uncomment when H-DP5 is implemented"""
        output_dict = dict()
        output_dict["C_output"] = []
        # output_dict["H_output"] = []
        output_dict["CDP5_output"] = []
        # output_dict["HDP5_output"] = []
        # output_dict["DP5_output"] = []
        for mol, clab, cshift, cexp, cerr, cpr in zip(
            self.mols,
            self.Clabels,
            self.Cshifts,
            self.Cexp,
            self.Cerrors,
            self.CDP5_atom_probs,
        ):
            output = f"\nAssigned C NMR shift for {mol}:"
            output += self.print_assignment(clab, cshift, cexp, cerr, cpr)
            output_dict["C_output"].append(output)

        # for mol, hlab, hshift, hscal, hexp, herr in zip(
        #    self.mols, self.Hlabels, self.Hshifts, self.Hscaled, self.Hexp, self.Herrors
        # ):
        #    output = f"\nAssigned H NMR shift for {mol}:"
        #    output += self.print_assignment(hlab, hshift, hscal, hexp, herr)
        #    output_dict["H_output"].append(output)

        for mol, cdp5 in zip(self.mols, self.CDP5_mol_probs):
            output_dict["CDP5_output"].append(
                f"Carbon DP5 probability for {mol}: {cdp5}"
            )
        t_dic = [
            dict(zip(output_dict.keys(), values))
            for values in zip(*output_dict.values())
        ]
        dp5_output = "\n\n".join([mol["C_output"] for mol in t_dic])
        dp5_output += "\n\n"
        dp5_output += "\n".join([mol["CDP5_output"] for mol in t_dic])
        return dp5_output

    @staticmethod
    def print_assignment(labels, calculated, exp, error, probs):
        """Prints table for molecule"""

        s = np.argsort(calculated)
        svalues = calculated[s]
        slabels = labels[s]
        sexp = exp[s]
        serror = error[s]
        sprob = probs[s]

        output = f"\nlabel, calc, exp, error, prob"

        for lab, calc, ex, er, p in zip(slabels, svalues, sexp, serror, sprob):
            output += f"\n{lab:6s} {calc:6.2f} {ex:6.2f} {er:6.2f} {p:6.2f}"
        return output


def _load_pickle(path: str):
    """
    Loads a pickled object from relative or absolute path.
    Searches within this folder first, then within current folder, then by absolute path.

    Arguments
      path(str): path to the pickled file

    Returns
      Loaded object
    """
    _abs_path = Path(path).resolve()
    _default_path = Path(__file__).parent / path

    if _default_path.exists():
        _path = _default_path
    elif _abs_path.exists():
        _path = _abs_path
    else:
        raise FileNotFoundError("No files found at %s" % (path))
    with open(_path, "rb") as f:
        return pickle.load(f)
