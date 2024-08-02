from pathlib import Path
import logging
from typing import List, Union, Dict

from rdkit import Chem
from rdkit.Chem import AllChem, EnumerateStereoisomers


logger = logging.getLogger(__name__)


def write_to_sdf(mol: Chem.rdchem.Mol, relative_path: Path):
    """
    Writes rdkit Mol object to a specified path

    arguments:
    - mol: RDKit Mol object
    - relative_path: path to write the file

    returns:
    - input_file: relative path to file from current working directory
    """
    path = Path.cwd() / relative_path
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    return relative_path


def cleanup_3d(mol):
    """
    Generates a 3D conformer of a molecule.

    arguments:
    - mol: RDKit mol object

    returns:
    - a reasonable conformer
    """
    mol = AllChem.AddHs(mol, addCoords=True)
    cid = AllChem.EmbedMolecule(mol)
    if cid == -1:
        raise ValueError("Molecule could not be sanitised")
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.rdmolops.AssignStereochemistryFrom3D(mol)

    return mol


def read_sdf(input_file: str):

    fullf = Path(input_file).resolve()
    mol = Chem.MolFromMolFile(str(fullf), removeHs=False, sanitize=True)

    return mol


def read_textfile(input_file: str, input_type: str):
    input_type = input_type.lower()

    input_readers = {
        "smiles": Chem.MolFromSmiles,
        "smarts": Chem.MolFromSmarts,
        "inchi": Chem.MolFromInchi,
    }

    if input_type not in input_readers:
        raise ValueError(f"Cannot parse {input_type}")

    mols = []

    with open(input_file) as f:
        for line in f:
            if line.strip() == "":
                continue
            try:
                mol = input_readers[input_type](line.strip(), sanitize=True)
                mol = Chem.AddHs(mol)
                mols.append(mol)
            except Exception as e:
                # to make an logging.error
                logging.error(f"Error reading line: {line} - {e}")

    return mols


def _generate_diastereomers(
    mol: Chem.rdchem.Mol,
    mutable_atoms: Union[List[int], None] = None,
    double_bonds: bool = False,
) -> List[Chem.rdchem.Mol]:
    """
    Enumerates all diastereomers and double bond isomers.

    Given an rdkit Mol object, generates all possible diastereomers and returns them as a list.
    Note that the function also checks if the resulting molecule can be embedded.

    Parameters:
    - mol (rdkit.Chem.rdchem.Mol): the molecule to have its stereocentres enumerated
    - protected_atoms (list[int], None): the list of atoms with a fixed stereochemistry, numbering starts with 1
    - double_bonds (bool): if double bond configuration is also altered. Defaults to False

    Returns:
    list of Mols
    """
    # copy the input to prevent accidental editing
    target_mol = Chem.Mol(mol)
    mutable_atoms = mutable_atoms.copy()

    enum_opts = EnumerateStereoisomers.StereoEnumerationOptions(
        tryEmbedding=False, onlyUnassigned=True, maxIsomers=0, unique=True
    )
    # extract chiral information if present in a molecule
    atom_chiral = [at.GetChiralTag() for at in target_mol.GetAtoms()]
    bond_geometry = [b.GetStereo() for b in target_mol.GetBonds()]

    # Creates list of mutable atoms. If none are specified, retains everything but the first stereocentre.
    atoms_with_stereo = [
        i
        for i, tag in enumerate(atom_chiral, start=1)
        if tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
    ]

    if not mutable_atoms:
        if len(atoms_with_stereo) > 0:
            atoms_with_stereo.pop(0)
        mutable_atoms = atoms_with_stereo

    mutable_atoms = [ind - 1 for ind in mutable_atoms]

    # all other atoms will have stereochemistry tag reset
    for atom in target_mol.GetAtoms():
        if atom.GetIdx() in mutable_atoms:
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)

    # scrambles all double bonds: if particular configuration was assigned, it is reset
    if double_bonds == True:
        [
            b.SetStereo(Chem.rdchem.BondStereo.STEREOANY)
            for b, g in zip(target_mol.GetBonds(), bond_geometry)
            if g != Chem.rdchem.BondStereo.STEREONONE
        ]

    result = []
    for isomer in EnumerateStereoisomers.EnumerateStereoisomers(
        target_mol, options=enum_opts
    ):
        isomer.RemoveAllConformers()
        cid = Chem.rdDistGeom.EmbedMolecule(isomer)
        if cid >= 0:
            AllChem.MMFFOptimizeMolecule(isomer)
            Chem.rdmolops.AssignStereochemistryFrom3D(isomer)
            result.append(isomer)

    return result


def prepare_inputs(
    input_files: List[str], input_type: str, stereocentres: List[int], workflow: Dict
) -> List[str]:
    """
    Reads files at the path specified by input config, prepares them as required by the user. Returns paths to the new files.

    Arguments:
    - input_file (list[str]): list of relative paths to structure input files. Contains one text file of several SD Files.
    - input_type (str): format of the input file. May be 'sdf', 'smiles', 'inchi', and 'smarts'.
    - stereocentres (list[int]): specifies mutable stereocentres. Defaults to empty list
    - workflow (dict): dictionary of booleans specifying the workflow.

    Returns:
    - mol_paths (list[str]): paths to the transformed files
    """
    # if not sdf, read text
    # generate diastereomers (cleans them) and clean inputs
    # returns paths to inputs
    # in principle, can create list of list of mols, use enumerate
    logger.info(f"Read structures from {input_files}")

    if input_type == "sdf":
        mols = [read_sdf(file) for file in input_files]
        logger.debug("read structures from SD File")
    else:
        mols = read_textfile(input_files[0], input_type)
        logger.debug(f"read structures from {input_type} file")
        input_files = [
            f"{input_type}_mol_{i:03}_.sdf"
            for i, mol in enumerate(range(len(mols)), start=1)
        ]

    if len(mols) < 1:
        raise ValueError("No molecules were provided!")

    logger.info(f"Structures read successfully")

    mols2 = []
    mutable_atoms = stereocentres if len(input_files) == 1 else []

    if workflow["generate"]:
        logger.info("Generating diastereomers")
        mols2 = [_generate_diastereomers(mol, mutable_atoms) for mol in mols]
    elif workflow["cleanup"] or (
        not workflow["conf_search"] and not workflow["dft_opt"]
    ):
        logger.info("Generating MMFF geometries for inputs")
        mols2 = [[cleanup_3d(mol)] for mol in mols]
    else:
        mols2 = [[mol] for mol in mols]

    logger.debug("Preparing to write structure files")
    filenames = []
    for filename, mol in zip(input_files, mols2):
        for i, isomer in enumerate(mol, start=1):
            if len(mol) == 1:
                fname = f"{filename[:-4]}.sdf"
            else:
                fname = f"{filename[:-4]}isomer{i:03}.sdf"
            filenames.append(fname)
            write_to_sdf(isomer, fname)

    return filenames
