# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:15:47 2015

@author: ke291

Contains all of the MacroModel specific code for input generation, calculation
execution and output interpretation. Called by PyDP4.py.
"""

import os
from pathlib import Path
import shutil
import sys
import subprocess
import shutil
import time
import re
import logging

from dp5.conformer_search.base_cs_method import BaseConfSearch, ConfData

logger = logging.getLogger(__name__)


class ConfSearchMethod(BaseConfSearch):

    def __init__(self, inputs, settings):
        super().__init__(inputs, settings)
        self.energy_cutoff = self.settings["energy_cutoff"]

        if settings["force_field"].lower() == "mmff":
            self.ff_code = 10
        elif settings["force_field"].lower() == "opls":
            self.ff_code = 14
        else:
            logger.critical("Invalid force field provided")
            raise ValueError("Invalid force field provided")

        if settings["executable"]["schrodinger"]:
            self.executable = settings["executable"]["schrodinger"]
        else:
            schrodinger_path = os.getenv("SCHRODINGER")
            if schrodinger_path is None:
                if os.path.exists("/usr/local/shared/schrodinger/current"):
                    self.executable = "/usr/local/shared/schrodinger/current"
                else:
                    self.executable = ""
                    logger.error("Could not find Schrodinger folder")
                    logger.error("Please provide the path in your config file")
            else:
                self.executable = schrodinger_path

    def __repr__(self):
        return "MacroModel"

    def prepare_input(self):
        if os.name == "nt":
            convinp = '"' + self.executable + '/utilities/sdconvert" -isd '
        else:
            convinp = self.executable + "/utilities/sdconvert -isd "

        for input in self.inputs:
            if not os.path.exists(f"{input}.mae"):
                if self.executable:
                    outp = subprocess.check_output(
                        f"{convinp}{input}.sdf -omae {input}.mae", shell=True
                    )
                else:
                    logger.critical("Cannot convert input to .mae format")
                    raise RuntimeError("Schrodinger path not found!")

            with open((Path.cwd() / f"{input}.com").resolve(), "w") as f:
                cmd = self.command.format(
                    input, self.ff_code, self.settings["step_count"]
                )
                f.write(cmd)
                logger.info(f"Prepared Macromodel input for {input}")

        logger.info("Macromodel input prepared successfully")

        return self.inputs

    def _run(self):

        completed = 0
        total = len(self.inputs)
        outputs = []

        installed = True
        path_to_exe = os.path.join(self.executable, "bmin")

        if shutil.which(path_to_exe) is None:
            logger.error(f"Could not find MacroModel executable at {path_to_exe}")
            installed = False

        for input in self.inputs:

            # first, check if we have run this already
            if os.path.exists(f"{input}.log"):
                if self.IsMMCompleted(f"{input}.log"):
                    logger.info(f"Valid {input}.log exists")
                    completed += 1
                    outputs.append(f"{input}.log")
                else:
                    logger.warning(f"{input}.log is incomplete, deleting...")
                    os.remove(f"{input}.log")

            else:
                if not installed:
                    logger.critical("Cannot launch conformational search")
                    raise RuntimeError("Path to Schrodinger not found!")

                logger.info(f"Starting conformational search for {input}")
                outp = subprocess.check_output(f"{path_to_exe} {input}", shell=True)

                time.sleep(60)
                while not self.IsMMCompleted(input + ".log"):
                    time.sleep(30)

                completed = completed + 1
                outputs.append(input)

            logger.info(f"{completed} out of {total} inputs processed")

        return outputs

    def _parse_output(self, file):
        output = f"{file}.log"
        if self.IsMMCompleted(output):
            result = self.read_macromodel(file, self.energy_cutoff)
            conf_data = ConfData(*result)
        else:
            logger.critical(f"{output} is not completed")
            raise FileNotFoundError(f"{output} containes incomplete calculations")
        return conf_data

    def read_macromodel(self, file, cutoff_energy):

        atoms_to_symbols = {
            "1": "C",
            "2": "C",
            "3": "C",
            "4": "C",
            "5": "C",
            "6": "C",
            "7": "C",
            "8": "C",
            "9": "C",
            "10": "C",
            "11": "C",
            "12": "C",
            "13": "C",
            "14": "O",
            "15": "O",
            "16": "O",
            "18": "O",
            "20": "O",
            "21": "O",
            "23": "O",
            "24": "N",
            "25": "N",
            "26": "N",
            "31": "N",
            "32": "N",
            "38": "N",
            "39": "N",
            "40": "N",
            "41": "H",
            "42": "H",
            "43": "H",
            "44": "H",
            "45": "H",
            "48": "H",
            "49": "S",
            "51": "S",
            "52": "S",
            "53": "P",
            "54": "B",
            "55": "B",
            "56": "F",
            "57": "Cl",
            "58": "Br",
            "59": "I",
            "60": "Si",
            "100": "S",
            "101": "S",
            "102": "Cl",
            "103": "B",
            "104": "F",
            "109": "S",
            "110": "S",
            "113": "S",
            "114": "S",
        }

        atoms = []
        coordinates = []
        charge = None
        energies = []

        with open(f"{file}-out.mae") as f:
            while True:
                try:
                    line = next(f)
                    # get energies
                    if line.startswith("p_m_ct") or line.startswith("f_m_ct"):
                        energy_offset = 0
                        while not ("mmod_Potential_Energy" in line):
                            line = next(f)
                            energy_offset += 1
                        while not ":::" in line:
                            line = next(f)

                        for _ in range(energy_offset):
                            line = next(f)
                        energies.append(float(line))

                    if line.startswith(" m_atom["):
                        offset = -1
                        while not ":::" in line:
                            if line.startswith("  i_m_mmod_type"):
                                atype = offset
                            elif line.startswith("  r_m_x_coord"):
                                xpos = offset
                            elif line.startswith("  r_m_y_coord"):
                                ypos = offset
                            elif line.startswith("  r_m_z_coord"):
                                zpos = offset
                            elif line.startswith("  r_m_charge1"):
                                cpos = offset
                            offset += 1
                            line = next(f)

                        line = next(f)
                        _atoms = []
                        coords = []
                        _charge = 0
                        while not ":::" in line:
                            line = line.split()
                            _atoms.append(line[atype])
                            coords.append(
                                [float(i) for i in (line[xpos], line[ypos], line[zpos])]
                            )
                            _charge += float(line[cpos])
                            line = next(f)
                        if not atoms:
                            atoms = [atoms_to_symbols[i] for i in _atoms]
                        if charge is None:
                            charge = int(_charge)
                        coordinates.append(coords)

                except StopIteration:
                    break

        final_coordinates = []
        final_energies = []
        e_min = min(energies)
        for coord, energy in zip(coordinates, energies):
            if energy <= e_min + cutoff_energy:
                final_coordinates.append(coord)
                final_energies.append(energy)

        return atoms, final_coordinates, charge, final_energies

    def IsMMCompleted(self, f):
        with open(f, "r") as Gfile:
            outp = Gfile.readlines()

        if os.name == "nt":
            i = -2
        else:
            i = -3

        if "normal termination" in outp[i]:
            return True
        else:
            return False

    command = """{0}.mae
    {0}-out.mae
    MMOD       0      1      0      0     0.0000     0.0000     0.0000     0.0000
    FFLD  {1:>6}      1      0      0     1.0000     0.0000     0.0000     0.0000
    BDCO       0      0      0      0    41.5692 99999.0000     0.0000     0.0000
    READ       0      0      0      0     0.0000     0.0000     0.0000     0.0000
    CRMS       0      0      0      0     0.0000     0.2500     0.0000     0.0000
    LMCS  {2:>6}      0      0      0     0.0000     0.0000     3.0000     6.0000
    NANT       0      0      0      0     0.0000     0.0000     0.0000     0.0000
    MCNV       0      0      0      0     0.0000     0.0000     0.0000     0.0000
    MCSS       2      0      0      0    21.0000     0.0000     0.0000     0.0000
    MCOP       1      0      0      0     0.5000     0.0000     0.0000     0.0000
    DEMX       0 333333      0      0    21.0000    42.0000     0.0000     0.0000
    MSYM       0      0      0      0     0.0000     0.0000     0.0000     0.0000
    AUOP       0      0      0      0  2500.0000     0.0000     0.0000     0.0000
    AUTO       0      2      1      1     0.0000    -1.0000     0.0000     2.0000
    CONV       2      0      0      0     0.0010     0.0000     0.0000     0.0000
    MINI       1      0 999999      0     0.0000     0.0000     0.0000     0.0000
    """
