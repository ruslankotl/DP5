from pathlib import Path
import logging
import warnings

from tqdm import tqdm
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
            self.C_DP5 = ErrorDP5ProbabilityCalculator(
                atom_type="C",
                model_file="NMRdb-CASCADEset_Error_mean_model_atom_features256.hdf5",
                batch_size=16,
                transform_file="pca_10_ERRORrep_Error_decomp.p",
                kde_file="pca_10_kde_ERRORrep_Error_kernel.p",
            )
        else:
            # must load model for shift preiction
            self.C_DP5 = ExpDP5ProbabilityCalculator(
                atom_type="C",
                model_file="NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5",
                batch_size=16,
                transform_file="pca_10_EXP_decomp.p",
                kde_file="pca_10_kde_EXP_kernel.p",
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
        c_dp5 = self.C_DP5(mols)


class DP5ProbabilityCalculator:
    def __init__(
        self,
        atom_type,
        model_file,
        batch_size,
        transform_file,
        kde_file,
    ):
        self.atom_type = atom_type
        self.model = build_model(model_file=model_file)
        self.batch_size = batch_size
        with open(Path(__file__).parent / transform_file, "rb") as tf:
            self.transform = pickle.load(tf)
        with open(Path(__file__).parent / kde_file, "rb") as kf:
            self.kde = pickle.load(kf)

    def kde_iterator(self, df):
        yield df[["representations", "conf_population", "conf_shifts", "exp_shifts"]]

    def __call__(self, mols):
        # must generate representations
        all_labels = []
        rep_df = []
        for mol_id, mol in enumerate(mols):
            calculated, experimental, labels, indices = self.get_shifts_and_labels(mol)
            # drop unassigned !
            has_exp = experimental != None
            new_calcs = calculated[:, has_exp]
            new_exps = experimental[has_exp]
            new_labs = labels[has_exp]
            new_inds = indices[has_exp]

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
            ],
        )
        rep_df = rep_df.explode(
            ["conf_id", "Mol", "conf_shifts", "conf_population"], ignore_index=True
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
        # to generate the objects for kde, must explode the rep_df, concatenate with ??, run_kde
        rep_df["atom_probs"] = self.kde_probfunction(rep_df)

        rep_df["weighted_atom_probs"] = rep_df["atom_probs"] * rep_df["conf_population"]
        weighted_probs = rep_df.groupby("mol_id")["weighted_atom_probs"].sum()
        weighted_probs = 1 - weighted_probs
        total_probs = np.array([np.exp(np.log(arr).mean()) for arr in weighted_probs])
        # rescale if necessary
        # eventually return atomic probs, weighted atomic probs, DP5 scores
        logger.info("Atomic probabilities estimated")
        return rep_df["atom_probs"], weighted_probs, total_probs

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
    def __init__(self, atom_type, model_file, batch_size, transform_file, kde_file):
        super().__init__(atom_type, model_file, batch_size, transform_file, kde_file)

    def kde_probfunction(self, df):
        # loop through atoms in the test molecule - generate kde for all of them.
        # implement joblib parallel search
        # check if this has been calculated

        df["scaled_shifts"] = df[["conf_shifts", "exp_shifts"]].apply(
            lambda row: scale_nmr(*row), axis=1
        )
        df["errors"] = df["scaled_shifts"] - df["exp_shifts"]

        min_value = -20
        max_value = 20
        n_points = 250

        x = np.linspace(min_value, max_value, n_points)

        probs = []
        with mp.ProcessingPool(nodes=mp.cpu_count()) as pool:
            for i, (rep, errors) in tqdm(
                df[["representations", "errors"]].iterrows(),
                total=len(df),
                desc="Computing error KDEs",
                leave=False,
            ):
                num_atoms, num_components = rep.shape
                rep_b = np.broadcast_to(
                    rep[:, :, np.newaxis], shape=(num_atoms, num_components, n_points)
                )
                x_b = np.broadcast_to(
                    x[np.newaxis, np.newaxis, :], shape=(num_atoms, 1, n_points)
                )
                point = np.concatenate((rep_b, x_b), axis=1)

                results = pool.map(self.kde, point[:])

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


class ExpDP5ProbabilityCalculator(DP5ProbabilityCalculator):
    def __init__(self, atom_type, model_file, batch_size, transform_file, kde_file):
        super().__init__(atom_type, model_file, batch_size, transform_file, kde_file)

    def kde_probfunction(self, df):
        """Deadass assumes no conformer search whatsoever, unsuitable for anything worthwhile"""
        # loop through atoms in the test molecule - generate kde for all of them.

        # check if this has been calculated
        conf_shifts = df["conf_shifts"]
        conf_reps = df["representations"]
        exp_data = df["exp_shifts"]

        df["weighted_reps"] = df["representations"] * df["conf_population"]

        total_reps = df.groupby("mol_id")["weighted_reps"].sum()
        exp_data = df.groupby("mol_id")["exp_shifts"].first()
        mol_df = pd.DataFrame({"representations": total_reps, "exp_shifts": exp_data})

        min_value = 0
        max_value = 250
        n_points = 250

        x = np.linspace(min_value, max_value, n_points)

        mol_probs = []
        with mp.ProcessingPool(nodes=mp.cpu_count()) as pool:
            for i, (rep, exp) in tqdm(
                mol_df[["representations", "exp_shifts"]].iterrows(),
                total=len(mol_df),
                desc="Computing experimental KDEs",
                leave=False,
            ):
                num_atoms, num_components = rep.shape
                rep_b = np.broadcast_to(
                    rep[:, :, np.newaxis], shape=(num_atoms, num_components, n_points)
                )
                x_b = np.broadcast_to(
                    x[np.newaxis, np.newaxis, :], shape=(num_atoms, 1, n_points)
                )
                point = np.concatenate((rep_b, x_b), axis=1)

                results = pool.map(self.kde, point[:])

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
