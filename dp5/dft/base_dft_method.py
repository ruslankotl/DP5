"""
wrapper for DFT methods.

Should implement setup, running, and reading the calculations
"""

import time
import subprocess
import sys
import shutil
from abc import abstractmethod, ABC
from math import isnan
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class BaseDFTMethod(ABC):

    hartree_to_kJ = 2625.499629554010
    gasConstant = 8.3145
    temperature = 298.15

    def __init__(self, settings):
        self.settings = settings
        if not isnan(self.settings["charge"]):
            self.charge = self.settings["charge"]
        else:
            self.charge = None
        self.force_converge = self.settings["optimisation_converged"]
        self.dft_complete = self.settings["dft_complete"]
        self.memory = self.settings["memory"]
        self.num_processors = self.settings["num_processors"]

        self.tag = ""
        self.input_format = ".in"
        self.output_format = ".out"
        self.executable = "echo"

    @abstractmethod
    def __repr__(self) -> str:
        return "DFTMethod"

    def _get_files(self, mols, calc_type):
        jobdir = Path.cwd()

        data = jobdir / calc_type
        data.mkdir(exist_ok=True)

        files_to_run = []
        input_files = {mol.base_name: [] for mol in mols}
        output_files = {mol.base_name: [] for mol in mols}

        for mol in mols:
            if self.charge:
                charge = self.charge
            else:
                charge = mol.charge

            for i, conf in enumerate(mol.conformers, start=1):
                filename = data / f"{mol}{self.tag}inp{i:03}"
                input_name = filename.with_suffix(self.input_format)
                output_name = filename.with_suffix(self.output_format)
                geom = conf

                if output_name.exists():
                    logger.debug(f"Reading DFT data from file {output_name}")
                    atoms, coords, *_, completed, converged = self.read_file(
                        output_name
                    )
                    if completed:
                        if calc_type == "opt":
                            if converged or self.force_converge:
                                output_files[mol.base_name].append(filename)
                                continue
                            else:
                                if coords:
                                    logger.info(
                                        f"Reusing partially optimised coordinates for {filename}"
                                    )
                                    geom = coords
                                output_name.unlink()
                        else:
                            output_files[mol.base_name].append(filename)
                            continue
                    else:
                        output_name.unlink()
                logger.debug(f"Creating new DFT file {input_name}")
                self.write_file(filename, geom, mol.atoms, charge, calc_type)
                input_files[mol.base_name].append(filename)
                files_to_run.append(filename)

        if files_to_run:
            logger.debug(f"Starting DFT calculations for {calc_type}")
            completed = self._run_calcs(files_to_run)

            for mol in mols:
                for file in completed:
                    if file in input_files[mol.base_name]:
                        output_files[mol.base_name].append(file)

        output_files = [output_files[key] for key in sorted(output_files.keys())]

        return output_files

    def _get_prerun_files(self, mols, calc_type: str):
        """
        Gathers and returns output files for calculations.
        Assumes all your files have terminated normally.

        Arguments:
        - mols: a list of Molecule objects
        - calc_type: a type of calculation.
        Returns:
         - list of lists of output files
        """
        jobdir = Path.cwd()
        data = jobdir / calc_type

        prerun_files = []

        if data.is_dir():
            for mol in mols:
                outputs = []
                pattern = f"{mol}{self.tag}inp*{self.input_format}"
                logger.debug(f"Searching for {pattern}")
                inputs = sorted([file for file in data.glob(pattern)])
                for input_file in inputs:
                    output_file = input_file.with_suffix(self.output_format)
                    if not output_file.exists():
                        logger.warning(f"Cannot find output file for {input_file}")
                        continue
                    if self.is_completed(output_file):
                        outputs.append(output_file)
                    else:
                        logger.warning(f"Calculations not complete for {output_file}")
                prerun_files.append(outputs)
        else:
            logger.error(f"No folder found at {data}!")

        return prerun_files

    def get_files(self, mols, calc_type):
        if self.dft_complete:
            logger.debug("Loading pre-run files")
            files = self._get_prerun_files(mols, calc_type)
        else:
            logger.debug("No pre-run files found, starting new calculations")
            files = self._get_files(mols, calc_type)
        return files

    def opt(self, mols):
        files = self.get_files(mols, "opt")

        atoms = []
        conformers = []
        energies = []

        for mol_data in files:
            mol_conformers = []
            mol_energies = []
            for file in mol_data:
                filename = file.with_suffix(self.output_format)
                logger.debug(f"Reading DFT output file: {filename}")
                mol_atoms, coords, energy, *_, opt_converged = self.read_file(filename)

                if opt_converged or self.force_converge:
                    mol_conformers.append(coords)
                    mol_energies.append(energy)
                else:
                    logger.critical(
                        f"Optimisation for {file} has not converged! Terminating..."
                    )
                    sys.exit(1)

            atoms.append(mol_atoms)
            conformers.append(mol_conformers)
            energies.append(mol_energies)

        return atoms, conformers, energies

    def energy(self, mols):
        files = self.get_files(mols, "e")

        atoms = []
        conformers = []
        energies = []

        for mol_data in files:
            mol_conformers = []
            mol_energies = []
            for file in mol_data:
                filename = file.with_suffix(self.output_format)
                logger.debug(f"Reading DFT output file: {filename}")
                mol_atoms, coords, energy, *_ = self.read_file(filename)
                mol_conformers.append(coords)
                mol_energies.append(energy)

            atoms.append(mol_atoms)
            conformers.append(mol_conformers)
            energies.append(mol_energies)

        return atoms, conformers, energies

    def nmr(self, mols):
        files = self.get_files(mols, "nmr")

        atoms = []
        conformers = []
        energies = []
        shieldings = []
        shielding_labels = []

        for mol_data in files:
            mol_conformers = []
            mol_energies = []
            mol_shieldings = []
            mol_shielding_labels = []
            for file in mol_data:
                filename = file.with_suffix(self.output_format)
                logger.debug(f"Reading DFT output file: {filename}")
                mol_atoms, coords, energy, shielding, shielding_label, *flags = self.read_file(filename)
                mol_conformers.append(coords)
                mol_energies.append(energy)
                mol_shieldings.append(shielding)
                mol_shielding_labels.append(shielding_label)

            atoms.append(mol_atoms)
            conformers.append(mol_conformers)
            energies.append(mol_energies)
            shieldings.append(mol_shieldings)
            shielding_labels.append(mol_shielding_labels)

        return atoms, conformers, energies, shieldings, shielding_labels

    @abstractmethod
    def write_file(self, filename, coordinates, atoms, charge, calc_type):
        raise NotImplementedError("not implemented")

    def _run_calcs(self, jobs: list):
        """
        - jobs: list of gaussian input files for which calculations are required
        """
        num_complete = 0
        jobs_complete = []

        if shutil.which(self.executable) is None:
            logger.critical(
                f"{self} executable not found at {self.executable}. Terminating..."
            )
            sys.exit(1)

        for file in jobs:
            logger.info(f"Starting calculations for {file}")
            cmd = self.prepare_command(file)
            outp = subprocess.check_output(cmd, shell=True, timeout=86400)

            num_complete += 1
            if self.is_completed(file.with_suffix(self.output_format)):
                jobs_complete.append(file)
                logger.info(f"{self} job {num_complete} of {len(jobs)} completed")
            else:
                logger.info(f"Job {file} terminated with an error. Continuing")

        if num_complete > 0:
            logger.info(f"{num_complete} jobs completed successfully")

        return jobs_complete

    @abstractmethod
    def prepare_command(self, file):
        return (
            f"{self.executable} {file}{self.input_format} > {file}{self.output_format}"
        )

    def is_converged(self, file):
        *_, converged = self.read_file(file)
        return converged

    def is_completed(self, file):
        *_, completed, converged = self.read_file(file)
        return completed

    @abstractmethod
    def read_file(self, file):
        """Reads output file. Inspired by cclib parser

        Returns:
            - atoms
            - coordinates
            - energies
            - shieldings
            - shielding labels
            - if calculation terminated normally
            - if optimisation has converged

        """
        atoms = []
        coordinates = []
        energy = None
        shieldings = []
        shielding_labels = []
        # check for normal termination
        completed = False
        # check for optimisation convergence
        converged = False
        with open(file, "r") as f:
            while not completed:
                try:
                    line = next(f)  # get new line using next
                    if "first line of geometry block":
                        atoms.append("atom")
                        coordinates.append("coordinates")

                    if "shielding":
                        shieldings.append("shielding")
                        shielding_labels.append("shielding label")

                    if "optimisation converged":
                        converged = True

                    if "energy":
                        energy = 42

                    if "normal termination":
                        completed = True
                    # add your geometry section, compare with gaussian.py!
                except StopIteration:
                    logger.error(
                        f"Calculations in {file} have not terminated normally."
                    )
                    break
            # write your functions
        return (
            atoms,
            coordinates,
            energy,
            shieldings,
            shielding_labels,
            completed,
            converged,
        )

    def atom_num_to_symbol(self, anum: int) -> str:

        Lookup = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
        ]

        if anum > 0 and anum < len(Lookup):
            return Lookup[anum - 1]
        else:
            print(f"No such element with atomic number {anum}")
            return 0
