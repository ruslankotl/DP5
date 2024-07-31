import logging
import os
import shutil

from dp5.dft import BaseDFTMethod


logger = logging.getLogger(__name__)


class DFTMethod(BaseDFTMethod):
    def __init__(self, settings):
        super().__init__(settings)

        self.tag = "o"
        self.input_format = ".inp"
        self.output_format = ".out"

        self.executable = self.settings["executable"]["orca"]

        if shutil.which(self.executable) is None:
            logger.warning(
                f"Gaussian executable not found at {self.executable}. Calculations will not run!"
            )

        for i in "noe":
            self.settings[f"{i}_basis_set"] = (
                "def2-tzvp"
                if self.settings[f"{i}_basis_set"] == "def2tzvp"
                else self.settings[f"{i}_basis_set"]
            )

    def __repr__(self) -> str:
        return "ORCA"

    def prepare_command(self, file):
        return super().prepare_command(file)

    def write_file(self, filename, coordinates, atoms, charge, calc_type):

        with open(f"{filename}.inp", "w") as f:

            if self.settings["num_processors"] > 1:
                f.write(f"%pal nprocs {self.settings['num_processors']}\n  end\n")

            # add solvents!!!

            if calc_type == "opt":
                f.write(self._opt_settings())
            elif calc_type == "nmr":
                f.write(self._nmr_settings())
            else:
                f.write(self._energy_settings())

            if self.settings["solvent"]:
                f.write(
                    f"%cpcm\n  smd true\n"
                    f'  smdSolvent "{self.settings["solvent"]}"\n'
                    "end\n"
                )

            f.write(f"* xyz {charge} 1\n")
            for atom, (x, y, z) in zip(atoms, coordinates):
                f.write(f"{atom}  {x:.6f}  {y:.6f}  {z:.6f}\n")
            f.write("*\n")

            if calc_type == "nmr":
                f.write(
                    "%eprnmr\n"
                    "  Ori GIAO\n"
                    "  NMRShielding 2\n"
                    "  Nuclei = all {shift}\n"
                    "end\n"
                )

    functional_dict = {
        "b3lyp": "functional b3lyp_g",
        "m062x": "functional m062x",
        "mpw1pw91": "exchange gga_x_mpw91\n  correlation gga_c_pw91",
    }
    basis_dict = {"def2tzvp": "def2-tzvp"}

    def _opt_settings(self):
        route = (
            "%method\n"
            "  Method dft\n"
            "  RunTyp opt\n"
            f"  {self.functional_dict[self.settings['o_functional'].lower()]}\n"
            "end\n"
            "%basis\n"
            f"  basis \"{self.settings['o_basis_set']}\"\n"
            "end\n"
            "%geom\n"
            f"  MaxIter {self.settings['max_opt_cycles']}\n"
            f"  MaxStep {0.01*self.settings['max_opt_cycles']}\n"
            f"  calc_hess {self.settings['calc_force_constants']}\n"
            "end\n"
        )
        return route

    def _nmr_settings(self):
        route = (
            "%method\n"
            "  Method dft\n"
            "  RunTyp energy\n"
            f"  {self.functional_dict[self.settings['n_functional'].lower()]}\n"
            "end\n"
            "%basis\n"
            f"  basis \"{self.settings['n_basis_set']}\"\n"
            "end\n"
        )
        return route

    def _energy_settings(self):
        route = (
            "%method\n"
            "  Method dft\n"
            "  RunTyp energy\n"
            f"  {self.functional_dict[self.settings['e_functional'].lower()]}\n"
            "end\n"
            "%basis\n"
            f"  basis \"{self.settings['e_basis_set']}\"\n"
            "end\n"
        )
        return route

    def read_file(self, file):
        atoms = []
        coordinates = []
        energy = None
        shieldings = []
        _shieldings = []
        shielding_labels = []
        # check for normal termination
        completed = False
        # check for optimisation convergence
        converged = False
        with open(file, "r") as f:
            while not completed:
                try:
                    line = next(f).strip()  # get new line using next
                    if line == "CARTESIAN COORDINATES (ANGSTROEM)":
                        _atoms = []
                        coordinates = []
                        for _ in range(2):
                            line = next(f).strip()
                        while line:
                            atom, *coords = line.split()
                            _atoms.append(atom)
                            coordinates.append([float(i) for i in coords])
                            line = next(f).strip()
                        if not atoms:
                            atoms = _atoms

                    if line == "CHEMICAL SHIELDING SUMMARY (ppm)":
                        for _ in range(6):
                            line = next(f).strip()
                        while line:
                            num, elem, shielding, _ = line.split()
                            _shieldings.append(
                                (
                                    int(num) + 1,
                                    float(shielding),
                                    "%s%i" % (elem, int(num) + 1),
                                )
                            )

                            line = next(f).strip()

                    if line == "***        THE OPTIMIZATION HAS CONVERGED     ***":
                        converged = True

                    if line.startswith("FINAL SINGLE POINT ENERGY"):
                        *_, _energy = line.split()
                        energy = self.hartree_to_kJ * float(_energy)

                    if line == "****ORCA TERMINATED NORMALLY****":
                        completed = True

                except StopIteration:
                    logger.error(
                        f"Calculations in {file} have not terminated normally."
                    )
                    break
            # write your functions
        # ORCA does not sort by atom :(
        if _shieldings:
            _shieldings.sort(key=lambda x: x[0])
            _, shieldings, shielding_labels = [list(x) for x in zip(*_shieldings)]
        return (
            atoms,
            coordinates,
            energy,
            shieldings,
            shielding_labels,
            completed,
            converged,
        )
