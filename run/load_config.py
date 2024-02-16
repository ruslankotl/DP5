"""
The goal is to parse configuration. 

First import config file to supersede default, 
and override I/O from command line if required.

Takes one file with line-delineated Smiles or InChIs, or several SDFiles.
Returns the final config for the run
"""
from pathlib import Path
import argparse
import os
import logging

import tomli

from dp5.run import runner, setup_logger

LOGLEVEL_CHOICES = tuple(level.lower() for level in logging._nameToLevel.keys())

DEFAULT_BASE_CONFIG_PATH = (Path(__file__).parent.parent / 'config/default_config.toml').resolve()

def main(): 

    parser = argparse.ArgumentParser(description="Load config and start the workflow manager.")

    parser.add_argument('-s', '--structure_files', nargs='*', default=[], type=str, help=
    "One or more SDF file for the structures to be verified by DP4. At least one\
    is required, if automatic diastereomer generation is used.")

    parser.add_argument('-n',"--nmr_file", help="Experimental NMR description, assigned\
    with the atom numbers from the structure file")

    parser.add_argument("-c", "--config", help="Load a config file", type=str, default=DEFAULT_BASE_CONFIG_PATH)

    parser.add_argument('-o', '--output', help="Output directory for calculations, default is current working directory.", default='')

    parser.add_argument('-i', '--input_type', help='Input file format. Default is sdf.',choices=['sdf', 'smiles', 'smarts', 'inchi'], default='sdf')

    parser.add_argument('-w', '--workflow', help="Defines which steps to include in the workflow, " 
                                                 "can contain g for generate diastereomers, m for molecular mechanics conformational search, " 
                                                 "o for DFT optimization, e for DFT single-point energies, n for DFT NMR calculation, " 
                                                 "a for computational and experimental NMR data extraction, " 
                                                 "s for computational and experimental NMR data extraction and stats analysis, "
                                                 "w for DP5 probability calculation.",
                        required=False)
    
    parser.add_argument('--stereocentres', nargs='*', default=[], type=int, help=
    "Atom indices matching input SD File for stereocentres to mutate, keep the rest intact.")

    parser.add_argument('-l', '--log_filename', help="Path to log file", default='')

    parser.add_argument('--log_level', choices=LOGLEVEL_CHOICES)

    args = parser.parse_args()

    # load custom configuration
    config_path = (Path.cwd() / args.config).resolve()
    with open(config_path,'rb') as f:
        config = tomli.load(f)

    # commandline overrides all config files
    if args.log_level is not None:
        config['log_level'] = args.log_level
    logger = setup_logger(
        name=__package__, level=config['log_level'].upper(), filename=args.log_filename
    ,propagate=True)

    logger.info("Preparing configuration")
    # Override workflow flag
    if args.workflow is not None:
        config['workflow']['cleanup'] = ('c' in args.workflow)
        config['workflow']['generate'] = ('g' in args.workflow)
        config['workflow']['conf_search'] = ('m' in args.workflow)
        config['workflow']['dft_nmr'] = ('n' in args.workflow)
        config['workflow']['dft_energies'] = ('e' in args.workflow)
        config['workflow']['dft_opt'] = ('o' in args.workflow)
        config['workflow']['dp4'] = ('s' in args.workflow)
        config['workflow']['dp5'] = ('w' in args.workflow)
        config['workflow']['assign_only'] = ('a' in args.workflow)

    logger.info(f"Structure cleanup required: {config['workflow']['cleanup']}")
    logger.info(f"Diastereomer generation required: {config['workflow']['generate']}")
    logger.info(f"Conformational search required: {config['workflow']['conf_search']}")
    logger.info(f"DFT geometry optimisation required: {config['workflow']['dft_opt']}")
    logger.info(f"DFT energy calculation required: {config['workflow']['dft_energies']}")
    logger.info(f"DFT shielding calculation required: {config['workflow']['dft_nmr']}")
    logger.info(f"DP4 analysis required: {config['workflow']['dp4']}")
    logger.info(f"DP5 analysis required: {config['workflow']['dp5']}")

    # reads command line argument if supplied, else reads config
    if args.structure_files:
        config['structure'] = args.structure_files
        config['input_type'] = args.input_type
        config['stereocentres'] = args.stereocentres
        logger.debug(f"Read structures {', '.join(config['structure'])} from command line")
    elif config['structure']:
        logger.debug(f"Read structures {', '.join(config['structure'])} from config file")
    else:
        logger.critical("No structures specified")
        raise ValueError("No structures specified")
    
    logger.info(f"Structure input files: {', '.join(config['structure'])}")
    
    if args.nmr_file:
        logger.debug(f"Read NMR File {args.nmr_file} from command line")
        config['nmr_file'] = args.nmr_file
    elif config['nmr_file']:
        logger.debug(f"Read NMR File {config['nmr_file']} from config file")
    else:
        logger.critical("No NMR data specified")
        raise ValueError("No NMR data specified")

    # set up TMS constants
    with open((Path(__file__).parent.parent/'dft'/'TMSdata').resolve()) as file:
        _params_found = False
        _solvent = config['dft']['solvent'] if config['dft']['solvent'] else 'none'
        for line in file:
            line = line.strip()
            if line:
                functional, basis_set, solvent, tms_c, tms_h = line.split()
                if (config['dft']['n_functional'] == functional and
                    config['dft']['n_basis_set'] == basis_set and
                    _solvent == solvent):
                    _params_found = True

                    config['dft']['c13_tms'] = float(tms_c)
                    config['dft']['h1_tms'] = float(tms_h)

                    break

    if not _params_found:
        logger.warning('No reference shielding found for the conditions, using default values!')
        functional, basis_set, solvent = ('b3lyp', '6-31G**', 'none')

    logger.info("Read shielding parameters for: ")
    logger.info("NMR DFT functional: %s, basis set: %s, solvent: %s", 
                functional, basis_set, solvent)

    logger.info(f"13C reference shielding: {config['dft']['c13_tms']:.1f} ppm")
    logger.info(f"1H reference shielding: {config['dft']['h1_tms']:.2f} ppm")

    if args.output:
        config['output_folder'] = args.output
    config['output_folder'] = (Path.cwd() / config['output_folder']).resolve()

    config['dft']['solvent'] = config['solvent']
                            

    runner(config)

    logger.info("Program terminated normally")


if __name__=="__main__":
    main()