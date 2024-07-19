from pathlib import Path
import logging

import pathos.multiprocessing as mp

from dp5.neural_net.CNN_model import *

logger = logging.getLogger(__name__)


class DP5:
    def __init__(self, output_folder: Path, use_dft_shifts: bool):
        logger.info("Setting up DP5 method")
        self.output_folder = output_folder
        self.dft_shifts = use_dft_shifts

        if not self.output_folder.exists():
            self.output_folder.mkdir()

        if not (self.output_folder/'dp5').exists():
            (self.output_folder/'dp5').mkdir()

        # if exists

    def save(self, file):
        for attr in self.__dict__:
            pass

    def __call__(self, mols):
        pass
