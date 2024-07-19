"""
The main script. Replaces PyDP4.py, orchestrates the high-level workflow.

Molecules is the container class to contain all the data. MM and DFT methods no longer modify it directly to improve maintainability.
"""

import logging

from .data_structures import Molecules
from .mol_file_preparation import prepare_inputs
from dp5.nmr_processing import NMRData


logger = logging.getLogger(__name__)


def runner(config):
    logger.info("Starting DP4 workflow")

    config["structure"] = prepare_inputs(
        config["structure"],
        config["input_type"],
        config["stereocentres"],
        config["workflow"],
    )

    logger.info(f"Final structure input files:{config['structure']}")

    logger.info(f"NMR input paths:{config['nmr_file']}")

    data = Molecules(config)

    if config["workflow"]["conf_search"] and not (
        config["workflow"]["restart_dft"] or config["workflow"]["calculations_complete"]
    ):

        logger.info("Conformational search requested")
        data.get_conformers()

    else:
        logger.info("No conformational search requested")

    if (
        config["workflow"]["dft_energies"]
        or config["workflow"]["dft_nmr"]
        or config["workflow"]["dft_opt"]
    ):

        logger.info("DFT calculations requested")
        data.get_dft_data()

    else:
        logger.info("No DFT calculations requested")

    if not config["workflow"]["dft_nmr"]:
        logger.info("Generating chemical shifts using a neural network")
        data.get_nn_nmr_shifts()

    # heading into legacy code area
    nmr_data = NMRData(
        config["nmr_file"],
        config["solvent"],
        config["output_folder"],
    )
    # process data first!!!!
    data.assign_nmr_spectra(nmr_data)

    # now that we have assigned it, time for DP4
    if config["workflow"]["dp5"]:
        data.dp5_analysis()
    if config["workflow"]["dp4"]:
        data.dp4_analysis()
