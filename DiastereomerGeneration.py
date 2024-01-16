import os


from rdkit import Chem
from rdkit.Chem import EnumerateStereoisomers, rdForceFieldHelpers


def GenDiastereomers(structf:str, **generation_params):
    """
    Reads SD File and generates SD File of diastereomers for a given structure. Excludes original input.

    Parameters:
    - structf (str): path to the original sdf input
    - nS (int): number of stereocenters. Kept for reverse compatibility.
    - protected_atoms (list[int]): the list of atoms with a fixed stereochemistry, numbering starts with 1
    - double_bonds (bool): if double bond configuration is also altered. Defaults to False
    - tryEmbedding (bool): if set the process attempts to generate a standard RDKit distance geometry conformation for the stereisomer. If this fails, we assume that the stereoisomer is non-physical and don’t return it. NOTE that this is computationally expensive and is just a heuristic that could result in stereoisomers being lost.

    Returns:
    filenames as list of strings
    """

    f = structf

    if not f.endswith('.sdf'):
        f += '.sdf'

    if os.path.sep not in f:
        f = os.path.join(os.getcwd(), f)

    m = Chem.MolFromMolFile(f, removeHs=False)
    rdForceFieldHelpers.MMFFOptimizeMolecule(m)
    Chem.AssignStereochemistryFrom3D(m)
    mols_list = enumerate_diastereomers(m,**generation_params)
    filenames = []

    for i, mol in enumerate(mols_list,start=1):
        
        save3d = Chem.SDWriter(f'{f[:-4]}{i}.sdf')
        save3d.write(mol)
        filenames.append(f'{f[:-4]}{i}')
    
    return filenames
    

def enumerate_diastereomers(mol:Chem.rdchem.Mol, protected_atoms: list[int] | None = None,double_bonds: bool = False) -> list[Chem.rdchem.Mol]:
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
    if protected_atoms is None:
        protected_atoms = []
    else:
        protected_atoms = protected_atoms.copy()

    enum_opts = EnumerateStereoisomers.StereoEnumerationOptions(tryEmbedding=False,onlyUnassigned=True,maxIsomers=0,unique=True)
    # extract chiral information if present in a molecule
    atom_chiral = [at.GetChiralTag() for at in target_mol.GetAtoms()]
    bond_geometry = [b.GetStereo() for b in target_mol.GetBonds()]

    # Creates list of protected atoms. If none are specified, fixes the first stereocentre
    if protected_atoms==[]:
        atoms_with_stereo = [i for i,tag in enumerate(atom_chiral,start=1) if tag!=Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
        if len(atoms_with_stereo)>0:
            protected_atoms.append(atoms_with_stereo[0])
    protected_atoms = [ind-1 for ind in protected_atoms]

    # all other atoms will have stereochemistry tag reset
    for atom in target_mol.GetAtoms():
        if atom.GetIdx() not in protected_atoms:
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)

    # scrambles all double bonds: if particular configuration was assigned, it is reset
    if double_bonds == True:
        [b.SetStereo(Chem.rdchem.BondStereo.STEREOANY) for b,g in zip(target_mol.GetBonds(),bond_geometry) if g!=Chem.rdchem.BondStereo.STEREONONE]

    result = []
    for isomer in EnumerateStereoisomers.EnumerateStereoisomers(target_mol,options=enum_opts):
        isomer.RemoveAllConformers()
        cid = Chem.rdDistGeom.EmbedMolecule(isomer)
        if cid>=0:
            rdForceFieldHelpers.MMFFOptimizeMolecule(isomer)
            result.append(isomer)

    return result