"""
The main script. Replaces PyDP4.py, orchestrates the high-level workflow.

Molecules is the container class to contain all the data. MM and DFT methods no longer modify it directly.
"""

import logging

from dp5.run.data_structures import Molecules
from dp5.run.load_nmr_files import find_nmr_files
from dp5.run.mol_file_preparation import prepare_inputs
from dp5.nmr_processing import NMRData


logger = logging.getLogger(__name__)


def runner(config):
    logger.info("Starting DP4 workflow")

    config['structure'] = prepare_inputs(config['structure'],
                                                                       config['input_type'],
                                                                       config['stereocentres'],
                                                                       config['workflow'])
    
    logger.info(f"Final structure input files:{config['structure']}")

    config['nmr_file'] = find_nmr_files(config['nmr_file'])
    logger.info(f"Final NMR input files:{config['nmr_file']}")

    data = Molecules(config)

    if config['workflow']['conf_search'] and not (
        config['workflow']['restart_dft'] or
        config['workflow']['calculations_complete']):

        logger.info("Conformational search requested")
        data.get_conformers()
        
    else:
        logger.info("No conformational search requested")

    if (config['workflow']['dft_energies'] or
        config['workflow']['dft_nmr'] or
        config['workflow']['dft_opt']):

        logger.info("DFT calculations requested")
        data.get_dft_data()

    else:
        logger.info("No DFT calculations requested")

    if not config['workflow']['dft_nmr']:
        logger.info("Generating chemical shifts using a neural network")
        data.get_nn_nmr_shifts()

    # heading into legacy code area
    nmr_data = NMRData(config['structure'], config['nmr_file'], config['solvent'], config['output_folder'])
    # process data first!!!!
    data.assign_nmr_spectra(nmr_data)
