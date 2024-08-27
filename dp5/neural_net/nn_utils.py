import logging

from .CNN_model import get_shifts_and_labels


logger = logging.getLogger(__name__)


def get_nn_shifts(mols, batch_size=16):
    """
    Predicts shifts from rdkit Mol objects.
    Arguments:
    - list of lists of RDKit mol objects
    Returns:
    - list of lists of 13C chemical shifts for each atom in a molecule
    - list of lists of C atomic labels
    - list of lists of 1H chemical shifts for each atom in a molecule
    - list of lists of H atomic labels
    """

    C_shifts, C_labels = predict_C_shifts(mols, batch_size)

    H_shifts, H_labels = predict_H_shifts(mols, batch_size)

    # will add H_shifts and H_labels later!

    return C_shifts, C_labels, H_shifts, H_labels


def predict_C_shifts(mols, batch_size):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return get_shifts_and_labels(
        mols,
        atomic_symbol="C",
        model_path="NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5",
        batch_size=batch_size,
    )


def predict_H_shifts(mols, batch_size):
    return [[[]]] * len(mols), [[]] * len(mols)
