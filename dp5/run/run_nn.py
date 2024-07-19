import logging


import numpy as np
import pandas as pd


from dp5.neural_net import predict_shifts, load_NMR_prediction_model


logger = logging.getLogger(__name__)


def get_nn_shifts(mols):
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

    C_shifts, C_labels = predict_cascade_shifts(mols)

    H_shifts, H_labels = [[[]]]*len(mols), [[]]*len(mols)

    # will add H_shifts and H_labels later!

    return C_shifts, C_labels, H_shifts, H_labels


def predict_cascade_shifts(mols, filepath="NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5", atomic_symbol='C'):
    """Predicts shifts

    Arguments:
    - list of lists of rdkit Mol objects
    - filepath: path to the prediction model

    Returns:
    - all_shifts: list of lists of shifts
    - all_labels: list of lists of labels
    """
    model = load_NMR_prediction_model(filepath)
    logger.info('Loaded NMR prediction model')

    all_shifts = []
    all_labels = []

    for mol in mols:
        iso_df = []
        shifts = []

        inds = [at.GetIdx() for at in mol[0].GetAtoms()
                if at.GetSymbol() == atomic_symbol]
        mol_labels = [f'{atomic_symbol}{i+1}' for i in inds]
        all_labels.append(mol_labels)

        for i, conf in enumerate(mol):
            iso_df.append((i, conf, np.array(inds)))

        iso_df = pd.DataFrame(iso_df, columns=['conf_id', 'Mol', 'atom_index'])
        shifts = predict_shifts(model, iso_df)
        all_shifts.append([i.tolist() for i in shifts])

    logger.info("Successfully predicted NMR shifts")
    return all_shifts, all_labels
