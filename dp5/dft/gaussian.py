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

        self.tag = "g"
        self.input_format = ".com"
        self.output_format = ".out"

        if "GAUS_EXEDIR" in os.environ:
            gausdir = os.environ["GAUSS_EXEDIR"]
            if shutil.which(os.path.join(gausdir, "g09")) is None:
                self.executable = os.path.join(gausdir, "g16")
            else:
                self.executable = os.path.join(gausdir, "g09")
        else:
            self.executable = self.settings["executable"]["gaussian"]

        if shutil.which(self.executable) is None:
            logger.warning(
                f"Gaussian executable not found at {self.executable}. Calculations will not run!"
            )

    def __repr__(self):
        return "Gaussian"

    def write_file(self, filename, coordinates, atoms, charge, type):

        with open(f"{filename}.com", "w") as f:

            if self.settings["num_processors"] > 1:
                f.write(f"%nprocshared={self.settings['num_processors']}\n")

            f.write(f"%mem={self.settings['memory']}MB\n" + f"%chk={filename}.chk\n")

            if type == "nmr":
                f.write(self.nmr_options())
            elif type == "e":
                f.write(self.e_options())
            elif type == "opt":
                f.write(self.opt_options())

            f.write(f"\n{filename}\n\n" + f"{charge} 1\n")

            for atom, (x, y, z) in zip(atoms, coordinates):
                f.write(f"{atom}  {x:.6f}  {y:.6f}  {z:.6f}\n")

            f.write("\n")

            # type specific footer

    def nmr_options(self):

        ultrafine = (
            " int=ultrafine"
            if (self.settings["n_functional"]).lower() == "m062x"
            else " "
        )
        solvent = (
            f" scrf=(solvent={self.settings['solvent']})"
            if self.settings["solvent"]
            else " "
        )

        route = (
            f"# {self.settings['n_functional']}/{self.settings['n_basis_set']}"
            f"{ultrafine} nmr=giao{solvent}\n"
        )

        return route

    def e_options(self):

        ultrafine = (
            " int=ultrafine"
            if (self.settings["e_functional"]).lower() == "m062x"
            else " "
        )
        solvent = (
            f" scrf=(solvent={self.settings['solvent']})"
            if self.settings["solvent"]
            else " "
        )

        route = (
            f"# {self.settings['e_functional']}/{self.settings['e_basis_set']}"
            f"{ultrafine}{solvent}\n"
        )

        return route

    def opt_options(self):

        ultrafine = (
            " int=ultrafine"
            if (self.settings["o_functional"]).lower() == "m062x"
            else " "
        )
        solvent = (
            f" scrf=(solvent={self.settings['solvent']})"
            if self.settings["solvent"]
            else " "
        )
        calc_fc = ",CalcFC" if self.settings["calc_force_constants"] else ""
        step_size = (
            f",MaxStep={100*self.settings['opt_step_size']}"
            if self.settings["opt_step_size"] != 0.3
            else ""
        )

        route = (
            f"# {self.settings['o_functional']}/{self.settings['o_basis_set']}"
            f"{ultrafine} Opt=(maxcycles={self.settings['max_opt_cycles']}{calc_fc}){step_size}{solvent}\n"
        )

        return route

    def _run_calcs(self, jobs: list):
        return super()._run_calcs(jobs)

    def prepare_command(self, file):
        return f"{self.executable} < {file}{self.input_format} > {file}{self.output_format}"

    def read_file(self, file):

        atoms = []
        coordinates = []
        energy = None
        shieldings = []
        shielding_labels = []
        completed = False
        opt_converged = False

        with open(file, "r") as f:
            while not completed:
                try:
                    line = next(f)
                    # reads geometry
                    if (
                        line.strip() == "Standard orientation:"
                        or line.strip() == "Input orientation:"
                    ):
                        atoms = []
                        coordinates = []
                        # skips lines
                        for _ in range(5):
                            line = next(f)

                        while not "----------" in line:
                            atom_idx, atom_number, _, *coords = line.split()
                            atom_label = self.atom_num_to_symbol(int(atom_number))
                            atoms.append(atom_label)
                            coordinates.append([float(x) for x in coords])
                            line = next(f)

                    # reads energy
                    if "SCF Done:" in line:
                        start = line.index(") =")
                        end = line.index("A.U.")
                        energy = float(line[start + 4 : end]) * self.hartree_to_kJ

                    # reads shieldings
                    if line == " SCF GIAO Magnetic shielding tensor (ppm):\n":
                        shieldings = []
                        line = next(f)
                        while "Isotropic" in line:
                            atom_idx, atom_label, _, _, shielding, *_ = line.split()
                            shieldings.append(float(shielding))
                            shielding_labels.append(f"{atom_label}{atom_idx}")

                            for _ in range(5):
                                line = next(f)

                    if "Stationary point found" in line:
                        opt_converged = True

                    if (
                        line == "Error termination request processed by link 9999."
                        or "Normal termination" in line
                    ):
                        completed = True

                except StopIteration:
                    logger.error(
                        f"Calculations in {file} have not terminated normally."
                    )
                    break

        return (
            atoms,
            coordinates,
            energy,
            shieldings,
            shielding_labels,
            completed,
            opt_converged,
        )
