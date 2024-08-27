import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdForceFieldHelpers

from dp5.conformer_search.run_cs import conf_search
from dp5.dft.run_dft import dft_calculations
from dp5.neural_net.nn_utils import get_nn_shifts
from dp5.analysis.dp5 import DP5
from dp5.analysis.dp4 import DP4


class Molecule:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.base_name = input_file.rsplit(".", maxsplit=1)[0]
        mol = Chem.MolFromMolFile(input_file, removeHs=False)

        self._atoms = [at.GetSymbol() for at in mol.GetAtoms()]
        self._conformers = [mol.GetConformer(0).GetPositions()]
        self._charge = sum([at.GetFormalCharge() for at in mol.GetAtoms()])

        # estimates force field energy
        prop = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, prop)
        self._energies = np.array([float(ff.CalcEnergy()) * 4.184])
        # creates mol object for further manipulation
        self._rdkit_mols = None
        self._populations = None
        self._mol = mol

    def __repr__(self) -> str:
        return self.base_name

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, new_atoms):
        self._atoms = np.array(new_atoms)

    @property
    def conformers(self):
        return self._conformers

    @conformers.setter
    def conformers(self, values):
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if values.shape[2] == 3:
            self._conformers = values
        else:
            raise ValueError("Cannot set coordinates!")
        self._rdkit_mols = None

    @property
    def energies(self):
        return self._energies

    @energies.setter
    def energies(self, values):
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        self._energies = values
        self._populations = None

    @property
    def charge(self):
        return self._charge

    @charge.setter
    def charge(self, value):
        self._charge = int(value)

    @property
    def rdkit_mols(self):
        if self._rdkit_mols is None:
            mols = []
            for conformer in self.conformers:
                molecule = Chem.Mol(self._mol)
                conf = molecule.GetConformer(0)
                for atom, atom_coord in enumerate(conformer):
                    x, y, z = atom_coord
                    conf.SetAtomPosition(atom, Point3D(x, y, z))
                mol = Chem.Mol(molecule, confId=0)
                mols.append(mol)
            self._rdkit_mols = mols
        return self._rdkit_mols

    @property
    def conformer_C_pred(self):
        return self._conformer_C_pred

    @conformer_C_pred.setter
    def conformer_C_pred(self, values):
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        self._conformer_C_pred = values

    @property
    def C_labels(self):
        return self._C_labels

    @C_labels.setter
    def C_labels(self, labels):
        self._C_labels = np.array(labels)

    @property
    def H_labels(self):
        return self._H_labels

    @H_labels.setter
    def H_labels(self, labels):
        self._H_labels = np.array(labels)

    @property
    def shielding_labels(self):
        return self._shielding_labels

    @shielding_labels.setter
    def shielding_labels(self, labels):
        self._shielding_labels = np.array(labels)

    def add_conformer_data(self, data):
        self.atoms = data.atoms
        self.conformers = data.conformers
        self.charge = data.charge
        self.energies = data.energies
        self._rdkit_mols = None

    def add_dft_data(self, data):
        attrs = (
            "atoms",
            "conformers",
            "charge",
            "energies",
            "conformer_C_pred",
            "C_labels",
            "conformer_H_pred",
            "H_labels",
            "shielding_labels",
        )
        for attr in attrs:
            if hasattr(data, attr):
                setattr(self, attr, getattr(data, attr))
        self._rdkit_mols = None

    def add_nn_shifts(self, shifts_labels):
        self.conformer_C_pred, self.C_labels, self.conformer_H_pred, self.H_labels = (
            shifts_labels
        )

    def copy(self):
        """Prevents accidental attribute override."""
        new_mol = type(self).__new__(self.__class__)
        new_mol.__dict__.update(self.__dict__)

        return new_mol

    @property
    def populations(self):
        """Assumeds kJ/mol as energy unit"""
        if self._populations is None:
            energies = self.energies
            scaling = 1000 / 8.3415 / 298.15
            energies = energies - np.min(energies)
            exp_energies = np.exp(-energies * scaling)
            self._populations = exp_energies / exp_energies.sum()
        return self._populations

    def boltzmann_weighting(self, attr: str):
        # recomputes populations just in case
        data = getattr(self, attr)
        data = np.array(data, dtype=np.float32)
        return (self.populations[:, np.newaxis] * data).sum(axis=0)

    @property
    def H_shifts(self):
        return self.boltzmann_weighting("conformer_H_pred")

    @property
    def C_shifts(self):
        return self.boltzmann_weighting("conformer_C_pred")

    def assign_nmr(self, C_exp, H_exp):
        self.C_exp, self.H_exp = np.array(C_exp, dtype=np.float32), np.array(
            H_exp, dtype=np.float32
        )

    def add_dp4_data(self, dp4_data):
        self.dp4_data = dp4_data

    def add_dp5_data(self, dp5_data):
        self.dp5_data = dp5_data


class Molecules:
    """Class that handles all the calculations. Should keep the molecular data in itself"""

    def __init__(self, config):
        self.config = config
        self.mols = [Molecule(mol) for mol in self.config["structure"]]

    def __iter__(self):
        return (mol.copy() for mol in self.mols)

    def __getitem__(self, idx):
        return self.mols[idx]

    def get_conformers(self):
        """Runs conformational search."""
        mm_data = conf_search(self.mols, self.config["conformer_search"])
        for mol, data in zip(self.mols, mm_data):
            mol.add_conformer_data(data)

    def get_dft_data(self):
        """Runs DFT calculations"""
        dft_mols = [mol for mol in self.mols]
        dft_data = dft_calculations(
            dft_mols, self.config["workflow"], self.config["dft"]
        )
        for mol, data in zip(self.mols, dft_data):
            mol.add_dft_data(data)

    def get_nn_nmr_shifts(self):
        """Should get C and H shifts"""
        mols = [mol.rdkit_mols for mol in self.mols]
        cascade_shifts_labels = get_nn_shifts(mols)
        for mol, *m_shift_label in zip(self.mols, *cascade_shifts_labels):
            mol.add_nn_shifts(m_shift_label)

    def assign_nmr_spectra(self, nmrdata):
        for mol in self.mols:
            C_exp, H_exp = nmrdata.assign(mol)
            mol.assign_nmr(C_exp, H_exp)

    def dp5_analysis(self):
        dp5 = DP5(self.config["output_folder"], self.config["workflow"]["dft_nmr"])
        self.dp5_output = dp5(self.mols)

    def dp4_analysis(self):
        dp4 = DP4(self.config["output_folder"], self.config["dp4"])
        self.dp4_output = dp4(self.mols)

    def print_results(self):
        output = "Workflow summary\n\n"
        output += f"Solvent = {self.config['solvent']}\n"
        if self.config["workflow"]["dft_opt"]:
            output += (
                f"DFT optimisation functional: {self.config['dft']['o_functional']}\n"
                f"DFT optimisation basis set: {self.config['dft']['o_basis_set']}\n"
            )
        if self.config["workflow"]["dft_energies"]:
            output += (
                f"DFT energy functional: {self.config['dft']['e_functional']}\n"
                f"DFT energy basis set: {self.config['dft']['e_basis_set']}\n"
            )
        if self.config["workflow"]["dft_nmr"]:
            output += (
                f"DFT NMR functional: {self.config['dft']['n_functional']}\n"
                f"DFT NMR basis set: {self.config['dft']['n_basis_set']}\n"
            )
        if self.config["dp4"]["param_file"] != "none":
            output += (
                f"\nDP4 statistical model file: {self.config['dp4']['param_file']}\n"
            )

        output += f"\nNumber of candidates: {len(self.mols)}\n"

        for i, mol in enumerate(self.mols):
            output += (
                f"Number of conformers for molecule {mol}: {len(mol.conformers)}\n"
            )

        if self.config["workflow"]["dp4"]:
            dp4_output = output + self.dp4_output
            with open((self.config["output_folder"]) / "output.dp4", "w") as f:
                f.write(dp4_output)

        if self.config["workflow"]["dp5"]:
            dp5_output = output + self.dp5_output
            with open((self.config["output_folder"]) / "output.dp5", "w") as f:
                f.write(dp5_output)
