import logging
from typing import List
import sys
from importlib import import_module
from pathlib import Path
import dataclasses

from dp5.conformer_search import five_conf, conf_prune

logger = logging.getLogger(__name__)


def conf_search(mols, config) -> List:
    """
    arguments:
    - mols: list of Molecule objects, handled by the Molecules container
    - config: conformational search config
    """

    try:
        Method = import_module(f'.{config["method"]}', "dp5.conformer_search")
        logger.debug(f"Loaded {config['method']} as conformational search method")
    except ModuleNotFoundError:
        logger.critical(f"No conformer search module {config['method']} found")
        sys.exit(1)
        raise ValueError(f"No conformer search module {config['method']} was found")

    try:
        method = Method.ConfSearchMethod(config)
        logger.debug(f"Initialised {config['method']}")
    except AttributeError:
        logger.critical(
            f"{config['method']} does not implement ConfSearchMethod. Terminating."
        )
        sys.exit(1)
        raise ValueError(f"{config['method']} does not implement ConfSearchMethod.")
    # input names have no extension!
    input_names = [mol.base_name for mol in mols]
    final_names = input_names.copy()

    flipped_inputs = []
    if config["manual_five_membered_rings"]:
        for file in input_names:
            if not Path(f"{file}rot.sdf").exists():
                # returns empty string if unnecessary
                rot = five_conf.main(f"{file}.sdf", config["five_membered_ring_atoms"])
                if not rot:
                    continue
            flipped_inputs.append(f"{file}rot")
        final_names.extend(flipped_inputs)

    logger.info(f"Using {config['method']} for conformational search")
    cs_data = method(final_names)
    logger.info("Conformational search complete")

    merged_data = []
    logger.info(f"Reading output files: {input_names}")
    for mol, data in zip(input_names, cs_data):
        coords = data.conformers
        energies = data.energies
        logger.debug(f"{len(coords)} conformers read for {mol}")

        if f"{mol}rot" in flipped_inputs:
            # merges the rotated inputs
            logger.info(f"found {mol}rot, appending to {mol}")
            rot_idx = final_names.index(f"{mol}rot")
            rot_data = cs_data[rot_idx]
            logger.debug(
                f"{len(rot_data.conformers)} rotated conformers read for {mol}rot"
            )
            coords.extend(rot_data.conformers)
            coords.extend(rot_data.energies)

        all_conformers = [(e, c) for e, c in zip(energies, coords)]
        all_conformers.sort(key=lambda x: x[0])

        emin = all_conformers[0][0]

        all_conformers = [
            (energy, coordinates)
            for energy, coordinates in all_conformers
            if energy < emin + config["energy_cutoff"]
        ]
        logger.debug(
            f"{len(all_conformers)} total conformers read within {config['energy_cutoff']:.1f} kJ/mol for {mol}"
        )

        if len(all_conformers) > config["conf_per_structure"] and config["conf_prune"]:
            logger.info(f"Conformer pruning required for {mol}")
            unique_conf_ids, cutoff = conf_prune(
                conformers=coords,
                cutoff=config["rmsd_cutoff"],
                conf_limit=config["conf_per_structure"],
            )

            all_conformers = [
                conformer
                for conformer, unique in zip(all_conformers, unique_conf_ids)
                if unique
            ]
            logger.info(f"Retained {len(all_conformers)} conformers for {mol}")
            logger.info(f"Adjusted RMSD threshold: {cutoff:.2f} \u00c5")

        energies, coords = zip(*all_conformers)

        data = dataclasses.replace(data, conformers=coords, energies=energies)

        merged_data.append(data)

    return merged_data
