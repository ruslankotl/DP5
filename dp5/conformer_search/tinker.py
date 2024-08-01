# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:52:21 2015

@author: ke291

Contains all of the Tinker specific code for input generation, calculation
execution and output interpretation. Called by PyDP4.py.
"""

import os
import shlex
import shutil
import subprocess
import logging

from rdkit import Chem
from rdkit.Chem.rdMolHash import MolHash, HashFunction


from dp5.conformer_search.base_cs_method import BaseConfSearch, ConfData
import dp5.conformer_search.sdftinkerxyzpy as sdftinkerxyzpy


logger = logging.getLogger(__name__)
__all__ = ["ConfSearchMethod"]


class ConfSearchMethod(BaseConfSearch):

    def __init__(self, settings):
        self.settings = settings
        if not self.settings["executable"]["tinker"]:
            logger.warning(
                "No Tinker executable specified. Please check your config file."
            )
        self.executable = self.settings["executable"]["tinker"]

        self.tinker_exe = os.path.join(self.executable, "bin", "scan")
        self.ff_params = os.path.join(self.executable, "params", "mmff.prm")

    def __repr__(self):
        return "Tinker"

    def prepare_input(self, inputs):
        for input in inputs:
            convinp = sdftinkerxyzpy.main(input)
            outp = subprocess.check_output(convinp + input + ".sdf", shell=True)
        return inputs

    def _run(self):

        completed = 0
        total = len(self.inputs)
        outputs = []

        if shutil.which(self.tinker_exe) is None:
            logger.critical(f"Could not find Tinker executable at {self.tinker_exe}")
            raise RuntimeError(f"Tinker executable not found at {self.tinker_exe}")

        if not os.path.exists(self.ff_params):
            logger.critical(f"Could not find MMFF parameters at {self.ff_params}")
            raise RuntimeError(f"MMFF parameters not found at {self.ff_params}")

        for input in self.inputs:
            if os.path.exists(f"{input}.tout") and os.path.exists(f"{input}.arc"):
                logger.info(f"Found output files for {input}")

            else:
                logger.info(f"Running conformational search for {input}")
                with open(f"{input}.tout", "w") as tout:
                    subprocess.run(
                        f"{self.tinker_exe} {input}.xyz {self.ff_params} 0 10 20 0.00001",
                        text=True,
                        shell=True,
                        check=True,
                        stdout=tout,
                    )

            outputs.append(input)
            completed = completed + 1
            logger.info(f"{completed} out of {total} conformer searches complete")

        return outputs

    def parse_output(self):
        final_list = []

        for file in self.inputs:
            final_list.append(self._parse_output(file))

        return final_list

    def _parse_output(self, file):

        all_energies, charge = self._get_energies_charge(file)
        atoms, all_conformers = self._read_arc(file)

        conformers = []
        energies = []

        for energy, conformer in zip(all_energies, all_conformers):
            if energy < min(energies) + self.settings["energy_cutoff"]:
                energies.append(energy)
                conformers.append(conformer)

        conf_data = ConfData(atoms, conformers, charge, energies)
        return conf_data

    def _get_energies_charge(self, file):

        with open(f"{file}.tout", "r") as f:
            inp = f.readlines()

        if len(inp) < 13:
            logger.critical(f"{file}.tout is incomplete")
            raise ValueError(f"{file}.tout is incomplete")

        energies = []

        # Get the conformer energies from the file
        energies = []
        for line in inp[13:]:
            data = line[:-1].split("  ")
            data = [_f for _f in data if _f]
            if len(data) >= 3:
                if "Map" in data[0] and "Minimum" in data[1]:
                    energies.append(float(data[-1]))

        mol = Chem.MolFromMolFile(f"{file}.sdf", removeHs=False)
        charge = int(MolHash(mol, HashFunction.NetCharge))

        return energies, charge

    def _read_arc(self, file):

        def GetAtomSymbol(AtomNum):
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

            # RK: original implementation is maintained
            if AtomNum > 0 and AtomNum < len(Lookup):
                return Lookup[AtomNum - 1]
            else:
                logger.error(
                    f"Element with atomic number {AtomNum} not supported, will return 0"
                )
                return 0

        atypes, anums = self._extract_atom_types()

        with open(f"{file}.arc", "r") as conffile:
            confdata = conffile.readlines()

        # output data: conformers - list of x,y,z lists, atoms - list of atoms
        conformers = []
        atoms = []
        atypes = [x[:3] for x in atypes]

        for line in confdata:
            data = [_f for _f in line.split("  ") if _f]
            if len(data) < 3:
                conformers.append([])
            else:
                if len(conformers) == 1:
                    anum = anums[atypes.index(data[1][:3])]
                    atoms.append(GetAtomSymbol(anum))
                conformers[-1].append([x for x in data[2:5]])

        return atoms, conformers

    def _extract_atom_types(self):
        atomtypes = []
        atomnums = []

        with open(self.ff_params, "w") as f:
            for line in f:
                if line.split(" ")[0] == "atom":
                    data = shlex.split(line, posix=False)
                    atomtypes.append(data[3])
                    atomnums.append(int(data[-3]))
        return atomtypes, atomnums
