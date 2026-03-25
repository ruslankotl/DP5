import logging
import sys
from importlib import import_module
from pathlib import Path
import dataclasses


logger = logging.getLogger(__name__)


def dft_calculations(mols, workflow, config):
    """
    Run DFT stages requested by the workflow and update molecule objects.

    This function dynamically imports the configured backend module,
    instantiates its ``DFTMethod`` class, and then executes optimisation,
    single-point energy, and/or NMR stages depending on workflow flags.

    :param mols: Molecule objects handled by the molecules container.
    :type mols: list
    :param workflow: Workflow control flags (for example ``dft_opt``,
        ``dft_energies``, ``dft_nmr``).
    :type workflow: dict
    :param config: DFT configuration subsection.
    :type config: dict
    :returns: Updated molecule list.
    :rtype: list
    """

    try:
        Method = import_module(f'.{config["method"]}', "dp5.dft")
        logger.debug(f"Loaded {config['method']} as DFT method")
    except ModuleNotFoundError:
        logger.critical(f"No DFT module {config['method']} found. Terminating.")
        sys.exit(1)

    try:
        method = Method.DFTMethod(config)
        logger.debug(f"Initialised {config['method']}")
    except AttributeError as e:
        logger.critical(
            f"{config['method']} does not implement DFTMethod. Terminating."
        )
        sys.exit(1)

    if workflow["dft_opt"]:
        logger.info("Getting optimised geometries...")

        atoms, geometries, energies = method.opt(mols)

        for mol, m_atoms, m_geoms, m_energies in zip(mols, atoms, geometries, energies):
            mol.atoms = m_atoms
            mol.conformers = m_geoms
            if not workflow["dft_energies"]:
                mol.energies = m_energies
        # now to update the geometries
        # should define an object which only copies relevant attributes

    if workflow["dft_energies"]:
        logger.info("Getting DFT energies...")
        atoms, geometries, energies = method.energy(mols)

        for mol, m_atoms, m_geoms, m_energies in zip(mols, atoms, geometries, energies):
            mol.atoms = m_atoms
            mol.conformers = m_geoms
            mol.energies = m_energies

    if workflow["dft_nmr"]:
        logger.info("Getting DFT NMR data...")
        atoms, geometries, energies, shieldings, shielding_labels = method.nmr(mols)
        for mol, m_atoms, m_geoms, m_energies, m_shieldings, m_labels in zip(
            mols, atoms, geometries, energies, shieldings, shielding_labels
        ):

            mol.conformer_C_pred, mol.C_labels, mol.conformer_H_pred, mol.H_labels = (
                _shielding_to_shift(
                    m_atoms, m_shieldings, config["c13_tms"], config["h1_tms"]
                )
            )
            mol.shielding_labels = m_labels
            # handle the shifts here

            if not (workflow["dft_energies"] or workflow["dft_opt"]):
                mol.energies = m_energies

    return mols


def _shielding_to_shift(atoms, shieldings, c13_ref, h1_ref):
    """Convert shielding tensors to predicted chemical shifts.

    The conversion uses the internal linear referencing expression employed by
    DP5 for both ``13C`` and ``1H`` nuclei.

    :param atoms: Atom labels for the molecule.
    :type atoms: list[str]
    :param shieldings: Shielding values grouped by conformer.
    :type shieldings: list[list[float]]
    :param c13_ref: ``13C`` reference shielding value.
    :type c13_ref: float
    :param h1_ref: ``1H`` reference shielding value.
    :type h1_ref: float
    :returns: ``(C_pred, C_labels, H_pred, H_labels)``.
    :rtype: tuple[list, list, list, list]
    """
    C_pred = []
    C_labels = []
    H_pred = []
    H_labels = []
    shieldings_by_atom = zip(*shieldings)
    for i, (atom, shielding) in enumerate(zip(atoms, shieldings_by_atom), start=1):
        if atom == "C":
            shift = [
                (c13_ref - c_sh) / (1 - (c13_ref * (10**-6))) for c_sh in shielding
            ]
            C_pred.append(shift)
            C_labels.append("C%i" % i)
        if atom == "H":
            shift = [(h1_ref - h_sh) / (1 - (h1_ref * (10**-6))) for h_sh in shielding]
            H_pred.append(shift)
            H_labels.append("H%i" % i)
    C_pred = [list(conf) for conf in zip(*C_pred)]
    H_pred = [list(conf) for conf in zip(*H_pred)]

    return C_pred, C_labels, H_pred, H_labels
