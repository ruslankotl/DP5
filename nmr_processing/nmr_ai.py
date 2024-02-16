"""
Functions after rewrite:
NMRData structure

"""

import re
import os
import math
import copy
import pickle
import shutil
import sys
import logging
from pathlib import Path

gasConstant = 8.3145
temperature = 298.15
hartreeEnergy = 2625.499629554010
logger = logging.getLogger(__name__)

from dp5.run.data_structures import Molecule
from dp5.nmr_processing import proton_processing, carbon_processing

class NMRData:
    """
    must spit out the labels and shifts
    """

    def __init__(self, structures, nmr_source, solvent, output_folder):
        self.structures = structures
        self.nmr_source = nmr_source
        self.solvent = solvent
        self.output_folder = output_folder
        self.data_type = 'desc'
        self.Atoms = []             # Element labels
        self.Cshifts = []           # Experimental C NMR shifts
        self.Clabels = []           # Experimental C NMR labels, if any
        self.Hshifts = []           # Experimental H NMR shifts
        self.Hlabels = []           # Experimental H NMR labels, if any
        self.Equivalents = []       # Atoms assumed to be NMR equivalent in computational data
        self.Omits = []
        self.protondata = {}
        self.carbondata = {}

        logger.info(f'Reading NMR data from {self.nmr_source}')

        if len(self.nmr_source)==0:
            logger.critical('No NMR Data Added, quitting...')
            sys.exit(1)

        else:
            for path in self.nmr_source:
                path = Path()
                if path.exists():
                    if path.is_dir():
                        if path.parts[-1] == 'Proton' or path.parts[-1] == 'proton':
                            proton_processing(self.nmr_source, self.solvent, 'bruker') # process bruker proton
                        elif path.parts[-1] == 'Carbon' or path.parts[-1] == 'carbon':
                            self.carbondata = carbon_processing(self.nmr_source, self.solvent, 'bruker')
                    elif path.parts[-1].lower() == 'proton.dx':
                        proton_processing(self.nmr_source, self.solvent,'jcamp')
                    elif path.parts[-1].lower() == 'carbon.dx':
                        self.carbondata = carbon_processing(self.nmr_source, self.solvent,'jcamp')
                    else:
                        pass
                else:
                    logger.fatal("No file found at %s" % str(path))
                    sys.exit(1)

    def process_proton(self, fid_type):
        pdir = self.output_folder / 'pickles' / 'protondata'
        gdir = self.output_folder / 'graphs' / 'protondata'
        if self.files_present('protondata'):
            pass
        else:
            self.protondata["xdata"], self.protondata["ydata"],\
            self.protondata["peakregions"], self.protondata["solventregions"],\
            self.protondata["picked_peaks"], self.protondata["params"],\
            self.protondata["sim_regions"] = proton_processing(
                self.nmr_source, self.solvent, fid_type)


    def process_carbon(self, fid_type):
        pass



