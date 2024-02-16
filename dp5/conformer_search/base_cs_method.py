"""
Sets up a template for conformer search class. Uses run as main method.
Should return a list of atoms, conformer geometries, charges, and energies.
"""

from abc import abstractmethod, ABC
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


class BaseConfSearch(ABC):

    def __init__(self, inputs, settings):
        self.inputs = inputs
        self.settings = settings

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError("No __repr__ attribute provided.")

    def __call__(self) -> list:
        """
        Handles input preparation, execution, and processing. The only method called externally.
        
        """
        logger.info(f"Using {self} as conformer search method")
        self.inputs = self.prepare_input()
        self.outputs = self._run()
        logger.debug(f"Conformer search output: {self.outputs}")

        return self.parse_output()
    
    @abstractmethod
    def _run(self):
        """
        Executes the conformer search.
        Arguments:
        - self.inputs: specified at initialisation
        Returns:
        - paths to output files
        """
        raise NotImplementedError("run method is not implemented")
    
    @abstractmethod
    def prepare_input(self):
        raise NotImplementedError("prepare method is not implemented")

    @abstractmethod
    def _parse_output(self, file):
        raise NotImplementedError("Method for reading output files is not implemented")

    def parse_output(self):
        final_list = []
        for file in self.inputs:
            conf_data = self._parse_output(file)
            assert isinstance(conf_data, ConfData), "Please pack your conformer search data into ConfData"
            final_list.append(conf_data)

        return final_list
    
@dataclass(frozen=True)    
class ConfData:
    atoms: list[str]
    conformers: list[list[list[float]]]
    charge: int
    energies: list[float]