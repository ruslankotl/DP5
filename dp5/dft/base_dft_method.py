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
    """Abstract base class for all DFT backends used by DP5.

    Concrete backends (for example Gaussian, ORCA, and NWChem) inherit from
    this class and implement file generation, command preparation, and output
    parsing for a specific quantum chemistry engine.

    The class also provides the orchestration logic that is backend-agnostic:
    selecting files to run, reusing completed calculations, launching jobs, and
    collecting parsed optimisation/energy/NMR data.
    """

    hartree_to_kJ = 2625.499629554010
    gasConstant = 8.3145
    temperature = 298.15

    def __init__(self, settings):
        """Initialise a DFT backend from the DFT config section.

        :param settings: DFT configuration dictionary loaded from the DP5
            input configuration.
        :type settings: dict
        """
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
        """Create missing inputs, run required jobs, and collect outputs.

        For each conformer in each molecule, this method determines whether an
        output already exists and can be reused, whether an incomplete output
        must be discarded, or whether a new input file must be written and run.

        :param mols: Molecules with conformer geometries.
        :type mols: list
        :param calc_type: Calculation type label (``"opt"``, ``"e"``,
            ``"nmr"``).
        :type calc_type: str
        :returns: Output file stems grouped per molecule.
        :rtype: list[list[pathlib.Path]]
        """
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
        """Return completed calculation files for the requested calculation.

        If ``self.dft_complete`` is set, pre-existing files are loaded from
        disk. Otherwise new calculations are prepared and executed.

        :param mols: Molecules with conformer geometries.
        :type mols: list
        :param calc_type: Calculation type label.
        :type calc_type: str
        :returns: Output file stems grouped per molecule.
        :rtype: list[list[pathlib.Path]]
        """
        if self.dft_complete:
            logger.debug("Loading pre-run files")
            files = self._get_prerun_files(mols, calc_type)
        else:
            logger.debug("No pre-run files found, starting new calculations")
            files = self._get_files(mols, calc_type)
        return files

    def opt(self, mols):
        """Run/read geometry optimisation results.

        :param mols: Molecules to optimise.
        :type mols: list
        :returns: Tuple of ``(atoms, conformers, energies)`` grouped by
            molecule, where conformers and energies correspond to converged
            optimisation outputs.
        :rtype: tuple[list, list, list]
        """
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
        """Run/read single-point energy calculations.

        :param mols: Molecules to evaluate.
        :type mols: list
        :returns: Tuple of ``(atoms, conformers, energies)`` grouped by
            molecule.
        :rtype: tuple[list, list, list]
        """
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
        """Run/read NMR shielding calculations.

        :param mols: Molecules to evaluate.
        :type mols: list
        :returns: Tuple of ``(atoms, conformers, energies, shieldings,
            shielding_labels)`` grouped by molecule.
        :rtype: tuple[list, list, list, list, list]
        """
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
        """Write an engine-specific input file for one conformer.

        :param filename: File stem without extension.
        :type filename: pathlib.Path
        :param coordinates: Cartesian coordinates for one conformer.
        :type coordinates: list
        :param atoms: Element symbols matching ``coordinates``.
        :type atoms: list[str]
        :param charge: Molecular charge to use for the job.
        :type charge: int | float
        :param calc_type: Calculation type label.
        :type calc_type: str
        """
        raise NotImplementedError("not implemented")

    def _run_calcs(self, jobs: list):
        """
        Execute backend commands for a list of prepared input file stems.

        :param jobs: List of file stems for which calculations are required.
        :type jobs: list[pathlib.Path]
        :returns: Subset of ``jobs`` with outputs marked as completed.
        :rtype: list[pathlib.Path]
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
        """Build a shell command to run one calculation.

        Concrete subclasses can override this to handle engine-specific command
        syntax.

        :param file: File stem without extension.
        :type file: pathlib.Path
        :returns: Shell command string.
        :rtype: str
        """
        return (
            f"{self.executable} {file}{self.input_format} > {file}{self.output_format}"
        )

    def is_converged(self, file):
        """Check whether an optimisation is marked as converged.

        :param file: Output file path.
        :type file: pathlib.Path
        :returns: ``True`` if optimisation converged.
        :rtype: bool
        """
        *_, converged = self.read_file(file)
        return converged

    def is_completed(self, file):
        """Check whether a calculation terminated normally.

        :param file: Output file path.
        :type file: pathlib.Path
        :returns: ``True`` if calculation completion flag is found.
        :rtype: bool
        """
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
        """Convert atomic number to element symbol.

        :param anum: Atomic number (1-indexed).
        :type anum: int
        :returns: Element symbol.
        :rtype: str
        """

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
