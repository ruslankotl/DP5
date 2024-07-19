import logging
import os
import shutil
import sys
import time
import pathlib
import subprocess
import re

from dp5.dft import BaseDFTMethod

logger = logging.getLogger(__name__)


class DFTMethod(BaseDFTMethod):
    def __init__(self, settings):
        super().__init__(settings)

        self.tag = 'nw'
        self.input_format = '.nw'
        self.output_format = '.nwo'

        self.executable = settings['executable']['nwchem']

        if shutil.which(self.executable) is None:
            logger.warning(f'NWChem executable not found at {self.executable}')
            self.executable = ''

    functional_dict = {'b3lyp': 'b3lyp',
                       'm062x': 'm06-2x', 'mpw1pw91': 'mpw91 perdew91'}

    def __repr__(self) -> str:
        return 'NWChem'

    def write_file(self, filename, coordinates, atoms, charge, calc_type):
        with open(f"{filename}.nw", "w") as f:
            f.write('memory stack 1500 mb heap 1500 mb global 3000 mb\n')
            # scratch folder left for hpc
            f.write(f'echo\n\nstart molecule\n\ntitle "{filename}"\n')
            f.write('echo\n\nstart\n\n')
            if self.charge:
                f.write(f'charge {self.charge}\n\n')
            else:
                f.write(f'charge {charge}\n\n')
            f.write('geometry units angstroms print xyz autosym\n')
            for atom, (x, y, z) in zip(atoms, coordinates):
                f.write(f'  {atom} {x} {y} {z}\n')

            gaus_to_nwchem_basis = {
                '6-31g(d,p)': '6-31g**', '6-311g(d)': '6-311g*'}

            if calc_type == 'nmr':
                basis = self.settings['n_basis_set']
            elif calc_type == 'opt':
                basis = self.settings['o_basis_set']
            elif calc_type == 'e':
                basis = self.settings['e_basis_set']

            basis = gaus_to_nwchem_basis.get(basis.lower(), basis)

            f.write(f'end\n\nbasis\n  * library {basis}\nend\n\n')

            gaus_to_nwchem_solvents = {
                'chloroform': 'chcl3', 'dimethylsulfoxide': 'dmso'}

            solvent = gaus_to_nwchem_solvents.get(
                self.settings['solvent'], self.settings['solvent'])
            if solvent:
                f.write(
                    f'cosmo\n  do_cosmo_smd true\n  solvent {solvent}\nend\n\n')

            if calc_type == 'nmr':
                f.write(self.nmr_options())
            elif calc_type == 'opt':
                f.write(self.opt_options())
            elif calc_type == 'e':
                f.write(self.e_options())

    def nmr_options(self):
        functional = self.settings['n_functional']
        functional = self.functional_dict.get(functional.lower(), functional)

        suffix = (f"dft\n  xc {functional}\n  mult 1\nend\n"
                  f"task dft energy\n\nproperty\n  shielding\nend\ntask dft property")
        return suffix

    def opt_options(self):
        functional = self.settings['o_functional']
        functional = self.functional_dict.get(functional.lower(), functional)

        suffix = (f"dft\n  xc {functional}\n  mult 1\nend\n"
                  f"task dft optimize\n\n")
        return suffix

    def e_options(self):
        functional = self.settings['e_functional']
        functional = self.functional_dict.get(functional.lower(), functional)

        suffix = (f"dft\n  xc {functional}\n  mult 1\nend\n"
                  f"task dft energy\n\nproperty\n  shielding\nend\ntask dft property")
        return suffix

    def prepare_command(self, file):
        return super().prepare_command(file)

    def read_file(self, file) -> tuple[list[str], list[list[list[float]]], list[float], list[float], bool, bool]:

        atoms = []
        coordinates = []
        energy = None
        shieldings = []
        shielding_labels = []
        completed = False
        opt_converged = False

        with open(file, 'r') as f:
            while not completed:
                try:
                    line = next(f)

                    if 'Geometry "geometry"' in line:
                        atoms = []
                        coordinates = []
                        for _ in range(7):
                            line = next(f)
                        while len(line) >= 2:
                            _, atom, _, x, y, z = line.split()
                            atoms.append(atom)
                            coordinates.append([float(i)
                                                for i in (x, y, z)])
                            line = next(f)

                    if 'Chemical Shielding' in line:
                        while not line.strip().startswith('Task'):
                            line = next(f).strip()
                            if line.startswith('Atom:'):
                                _, at_num, at_type = line.split()
                                shielding_labels.append(f'{at_type}{at_num}')
                            if line.startswith('isotropic'):
                                *_, shielding = line.split()
                                shieldings.append(float(shielding))

                    if 'Total DFT energy' in line:
                        start = line.index('Total')
                        energy = float(line[start+19:])

                    if 'Optimization converged' in line:
                        opt_converged = True

                    if "AUTHORS" in line:
                        completed = True

                except StopIteration:
                    logger.error(
                        f"Calculations in {file} have not terminated normally.")

        return atoms, coordinates, energy, shieldings, shielding_labels, completed, opt_converged
