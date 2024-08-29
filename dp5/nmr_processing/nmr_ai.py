"""
Functions after rewrite:
NMRData structure
Should have an assignment method
"""

from .helper_functions import *
from .proton.process import proton_processing, proton_assignment
from .proton.plot import plot_proton
from .description_files import (
    process_description,
    pairwise_assignment,
    matching_assignment,
)
from .carbon.process import carbon_processing, carbon_assignment
from .carbon.plot import plot_carbon

import pickle
import logging
from pathlib import Path
from typing import List

gasConstant = 8.3145
temperature = 298.15
hartreeEnergy = 2625.499629554010
logger = logging.getLogger(__name__)


class NMRData:
    """
    must spit out the labels and shifts
    Arguments:
    nmr_source: NMR source files
    solvent: solvent used
    output_folder: path to output folder
    """

    def __init__(
        self, nmr_source: List[str], solvent: str, output_folder: Path = Path.cwd()
    ):
        self.nmr_source = [Path(i) for i in nmr_source]
        self.solvent = solvent
        self.output_folder = output_folder
        self.Atoms = []  # Element labels
        self.Cshifts = []  # Experimental C NMR shifts
        self.Clabels = []  # Experimental C NMR labels, if any
        self.Hshifts = []  # Experimental H NMR shifts
        self.Hlabels = []  # Experimental H NMR labels, if any
        self.Equivalents = (
            []
        )  # Atoms assumed to be NMR equivalent in computational data
        self.Omits = []
        self.protondata = {}
        self.carbondata = {}

        logger.info(f"Reading NMR data from {nmr_source}")

        self.search_files()
        # will guess a file. sets proton_fid and carbon_fid attributes if FID data detected

        if hasattr(self, "proton_fid"):
            self.process_proton()
        if hasattr(self, "carbon_fid"):
            self.process_carbon()

    def search_files(self):
        """Automatically searches the path for NMR data, guesses the nucleus."""
        for item in self.nmr_source:
            if item.is_dir() and (item / "fid").exists():
                logging.info("Bruker FID data found at %s" % (str(item)))
                nucleus, total_spectral_ydata, uc = read_bruker(item)
            elif item.is_file and item.suffix in (".dx", ".jdx"):
                logging.info("JCAMP-DX FID data found at %s" % (str(item)))
                nucleus, total_spectral_ydata, uc = read_jcamp(item)
            else:
                logging.info("NMR Description data found at %s" % (str(item)))
                self.process_description(item)
                return

            if nucleus == "1H":
                logger.info(f"1H NMR FID data found at: {item}")
                self.proton_fid = total_spectral_ydata, uc
            elif nucleus == "13C":
                logger.info(f"13C NMR FID data found at: {item}")
                self.carbon_fid = total_spectral_ydata, uc
        return

    def process_proton(self):
        pdir = self.output_folder / "protondata"
        gdir = self.output_folder / "graphs" / "protondata"
        if pdir.exists():
            with open(pdir, "rb") as f:
                self.protondata = pickle.load(f)
        else:
            ydata, uc = self.proton_fid
            (
                self.protondata["xdata"],
                self.protondata["ydata"],
                self.protondata["peak_regions"],
                self.protondata["grouped_peaks"],
                self.protondata["picked_peaks"],
                self.protondata["params"],
                self.protondata["sim_regions"],
            ) = proton_processing(ydata, uc, self.solvent)
            with open(pdir, "wb") as f:
                pickle.dump(self.protondata, f)

    def process_carbon(self):
        """NMR-AI not yet implemented"""
        cdir = self.output_folder / "carbondata"
        gdir = self.output_folder / "graphs" / "carbondata"
        if cdir.exists():
            with open(cdir, "rb") as f:
                self.carbondata = pickle.load(f)
        else:
            ydata, uc = self.carbon_fid

            (
                self.carbondata["xdata"],
                self.carbondata["ydata"],
                self.carbondata["exppeaks"],
                self.carbondata["simulated_ydata"],
                self.carbondata["removed"],
            ) = carbon_processing(ydata, uc, self.solvent)
            with open(cdir, "wb") as f:
                pickle.dump(self.carbondata, f)

    def process_description(self, file):

        (
            self.C_labels,
            self.C_exp,
            self.H_labels,
            self.H_exp,
            self.equivalents,
            self.omits,
        ) = process_description(file)

    def assign(self, mol):
        """
        mol: Molecule object
        Assigns data to molecule
        returns:
        - assigned experimental carbon shifts
        - scaled calculated carbon shifts
        - assigned experimental proton shifts
        - scaled calculated proton shifts
        """
        C_exp = []
        H_exp = []

        _mol = mol.rdkit_mols[0]

        C_shifts = mol.C_shifts
        C_labels = mol.C_labels
        H_shifts = mol.H_shifts
        H_labels = mol.H_labels

        if self.protondata:
            H_exp = proton_assignment(self.protondata, _mol, H_shifts, H_labels)
            plot_proton(self.protondata, self.output_folder, mol, H_exp)
        elif hasattr(self, "H_exp"):
            H_exp = pairwise_assignment(H_shifts, self.H_exp)

        if self.carbondata:
            C_exp = carbon_assignment(self.carbondata, _mol, C_shifts, C_labels)
            plot_carbon(self.carbondata, self.output_folder, mol, C_exp)
        elif hasattr(self, "C_exp"):
            C_exp = matching_assignment(C_shifts, self.C_exp, threshold=40)

        return C_exp, H_exp

    def __call__(self, mol):
        return self.assign(mol)
