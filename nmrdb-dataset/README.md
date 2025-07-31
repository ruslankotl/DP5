# nmrdb-dataset
This repository contains the nmrshiftdb2-derived model training dataset used for development of DFT-free DP5 analysis of NMR spectra.

Data is contained in form of compressed pickled Pandas dataframes. In order to open  the file, use the following command:

```
import numpy as np
import pandas as pd

from rdkit import Chem # useful for further manipulations

filename = 'your filename'
data = pd.read_pickle(filename, compression='gzip')
```
The columns contain the following data:

- `Mol`: contains RDKit Mol objects with 3D atomic coordinates optimised using MMFF
- `n_atoms`: number of carbon atoms in the molecule in `Mol`
- `atom_index`: numpy array containing 0-based indices of atoms in `Mol` for which shifts are recorded
- `Shift`: numpy array containing chemical shifts for atoms with indices specified in `atom_index`
