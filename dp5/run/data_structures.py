import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdForceFieldHelpers

from dp5.run.run_cs import conf_search
from dp5.run.run_dft import dft_calculations
from dp5.run.run_nn import get_nn_shifts


class Molecule:
    def __init__(self, input_file:str):
        self.input_file = input_file
        self.base_name = input_file.rsplit('.',maxsplit=1)[0]
        mol = Chem.MolFromMolFile(input_file,removeHs=False)

        self.atoms = [at.GetSymbol() for at in mol.GetAtoms()]
        self.conformers = [mol.GetConformer(0).GetPositions().tolist()]
        self.charge = sum([at.GetFormalCharge() for at in mol.GetAtoms()])

        prop = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, prop)
        self.energies = [float(ff.CalcEnergy())*4.184]
        self.rdkit_mols = [mol]
        
    def __repr__(self) -> str:
        return self.base_name

    def create_rdkit_mols(self):
        mols = []
        for conformer in self.conformers:
            molecule = Chem.MolFromMolFile(self.input_file, removeHs=False)
            conf = molecule.GetConformer(0)
            for atom, atom_coord in enumerate(conformer):
                x, y, z = atom_coord
                conf.SetAtomPosition(atom, Point3D(x, y, z))
            mol = Chem.Mol(molecule, confId=0)
            mols.append(mol)
        self.rdkit_mols = mols
        return mols

    def add_conformer_data(self, data):
        self.atoms = data.atoms
        self.conformers = data.conformers
        self.charge = data.charge
        self.energies = data.energies

    def update_mol_data(self, data):
        self.__dict__.update(data.__dict__)

    def add_nn_shifts(self, shifts_labels):
        self.C_pred, self.C_labels, self.H_pred, self.H_labels = shifts_labels 


    def copy(self):
        """Prevents accidental attribute override.
        """
        new_mol = type(self).__new__(self.__class__)
        new_mol.__dict__.update(self.__dict__)

        return new_mol

    def calculate_populations(self):
        # kJ/mol required!
        scaling = 1000 / 8.3415 / 298.15
        energies = np.array(self.energies)
        energies = energies - np.min(energies)
        exp_energies = np.exp(-energies*scaling)
        self.populations = (exp_energies / exp_energies.sum())

    def boltzmann_weighting(self, attr:str):
        # recomputes populations just in case
        self.calculate_populations()
        data = getattr(self, attr)
        data = np.array(data, dtype=np.float32)
        return (self.populations[:,np.newaxis]*data).sum(axis=0)


    def predicted_c_shifts(self):
        return self.boltzmann_weighting('C_pred')
    
    def predicted_h_shifts(self):
        return self.boltzmann_weighting('H_pred')
    



class Molecules:
    """Class that handles all the calculations. Should keep the molecular data in itself"""
    def __init__(self, config):
        self.config = config
        self.mols = [Molecule(mol) for mol in self.config['structure']]

    def __iter__(self):
        return (mol for mol in self.mols)
    
    def __getitem__(self, idx):
        return self.mols[idx]
    
    def get_conformers(self):
        """Runs conformational search."""
        mm_data = conf_search(self.mols, self.config["conformer_search"])
        for mol, data in zip(self.mols, mm_data, strict=True):
            mol.add_conformer_data(data)

    def get_dft_data(self):
        """Runs DFT calculations"""
        dft_mols = [mol.copy() for mol in self.mols]
        dft_data = dft_calculations(dft_mols, self.config['workflow'], self.config['dft'])
        for mol, data in zip(self.mols, dft_data, strict=True):
            mol.update_mol_data(data)

    def get_nn_nmr_shifts(self):
        """Should get C and H shifts"""
        mols = [mol.create_rdkit_mols() for mol in self.mols]
        cascade_shifts_labels = get_nn_shifts(mols)
        for mol, m_shift_label in zip(self.mols, cascade_shifts_labels):
            mol.add_nn_shifts(m_shift_label)

    def assign_nmr_spectra(self, nmrdata):
        assignments = assign_nmr(self.mols, nmrdata)