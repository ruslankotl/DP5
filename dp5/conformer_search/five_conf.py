# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 15:20:18 2014

@author: ke291

Gets called by PyDP4.py if automatic 5-membered cycle corner-flipping is used.
"""

import logging
from math import pi, cos, sin
from scipy.optimize import minimize
from rdkit import Chem, Geometry
from rdkit.Geometry.rdGeometry import Point3D

logger = logging.getLogger(__name__)


def main(f, ring_atoms):
    """
    Find the axis atoms
    Find all the atoms to be rotated

    Rotate it and the substituents to the other side of the plane

    Saves the rotated files, returns nothing.
    Arguments:
    - f: SD File to be read
    - ring_atoms: a list of ring atoms to permute

    Returns:
    none
    """

    mol = Chem.MolFromMolFile(f, removeHs=False)
    conformer = mol.GetConformer()

    # find atoms making up the five-membered ring
    rings = Chem.GetSSSR(mol)
    ring_to_alter = []
    for ring in rings:
        if len(ring_atoms) == 5:
            if all((at in ring) for at in ring_atoms):
                ring_to_alter = ring
                break
        else:
            if len(ring) == 5 and not ring_is_aromatic(mol, ring):
                ring_to_alter = ring
                break

    if not ring_to_alter:
        logger.info("No five membered rings to rotate in %s. Skipping...", f)
        return ""

    # Find the plane of the 5-membered ring and the outlying atom
    norm, d, out_atom_index = find_five_membered_ring_plane(conformer, ring_to_alter)
    out_atom = mol.GetAtomWithIdx(out_atom_index)
    out_atom_pos = conformer.GetAtomPosition(out_atom_index)

    # Find the atoms connected to the outlying atom and sort them
    # as either part of the ring(axis atoms) or as atoms to be rotated
    backbone_atoms = []  # indices!
    rotate_atoms = []  # indices!

    # let's make everything index-centric!

    # this iteration is atom-centric

    for neighbor_atom in out_atom.GetNeighbors():
        nbr_index = neighbor_atom.GetIdx()
        if nbr_index in ring_to_alter:
            backbone_atoms.append(nbr_index)
        else:
            rotate_atoms.append(nbr_index)
            find_sub_atoms(mol, nbr_index, out_atom_index, rotate_atoms)

    # Simple switch to help detect if the atoms are rotated the right way
    angle = find_rot_angle(conformer, *backbone_atoms, out_atom_index, norm)
    rotation_angle = 2 * (0.5 * pi - angle)
    if angle > 0.5 * pi:
        was_obtuse = True
        rotation_angle = 2 * (angle - 0.5 * pi)
    else:
        was_obtuse = False
        rotation_angle = 2 * (0.5 * pi - angle)

    logger.info(
        "Atom "
        + str(out_atom_index)
        + " will be rotated by "
        + str(rotation_angle * 57.3)
        + " degrees"
    )

    # do not confuse positions, angles

    new_angle = find_rot_angle(conformer, *backbone_atoms, out_atom_index, norm)

    if ((new_angle > 0.5 * pi) and was_obtuse) or (
        (new_angle < 0.5 * pi) and not was_obtuse
    ):
        rotation_angle = -rotation_angle

    new_coords = rotate_atom_coords(conformer, out_atom_index, *backbone_atoms, angle)
    conformer.SetAtomPosition(out_atom_index, new_coords)

    rotated_atoms = []
    for atom_index in rotate_atoms:
        if atom_index not in rotated_atoms:
            new_coords = rotate_atom_coords(
                conformer, atom_index, *backbone_atoms, angle
            )
            conformer.SetAtomPosition(atom_index, new_coords)
            rotated_atoms.append(atom_index)
        else:
            logger.info("Atom already rotated, skipping")

    new_id = mol.AddConformer(conformer, assignId=True)
    Chem.MolToMolFile(mol, f[:-4] + "rot.sdf", confId=new_id)
    logger.info("Five membered rings processed in %s.", f)
    return f


def ring_is_aromatic(mol, ring_system):
    """iterates through the rings, confirms all bonds are aromatic
    Arguments:
    - mol(rdkit.Chem.Mol): rdkit Mol object
    - ring_system(tuple or list): a collection containing ring indices. Assumes the order of the atoms follows the ring
    Returns a boolean
    """
    ring_size = len(ring_system)

    for index in range(ring_size):
        i1, i2 = ring_system[index], ring_system[(index + 1) % ring_size]
        if not mol.GetAtomWithIdx(i1).GetIsAromatic():
            return False
        bond = mol.GetBondBetweenAtoms(i1, i2)
        if not bond.GetIsAromatic():
            return False

    return True


def find_plane(atom1: Point3D, atom2: Point3D, atom3: Point3D):

    vector1 = atom2 - atom1
    vector2 = atom3 - atom1

    normal_vector = vector1.CrossProduct(vector2)
    d = atom1.DotProduct(normal_vector)

    return normal_vector, d


def point_plane_dist(norm: Point3D, d: float, atom: Point3D):
    norm = Point3D(*norm)
    a = abs(norm.DotProduct(atom) - d)
    n_length = norm.Length()
    return a / n_length


def plane_error(atoms, a, b, c, d):
    dists = []
    for atom in atoms:
        dists.append(point_plane_dist([a, b, c], d, atom))
    return sum(dists) / len(dists)


def least_squares_plane(atom1, atom2, atom3, atom4):
    [a0, b0, c0], d0 = find_plane(atom1, atom2, atom3)

    f = lambda a: plane_error([atom1, atom2, atom3, atom4], a[0], a[1], a[2], a[3])
    res = minimize(f, (a0, b0, c0, d0), method="nelder-mead")
    plane = list(res.x)

    return Point3D(*plane[:3]), plane[3], f(plane)


def find_five_membered_ring_plane(conformer, atom_ids):
    """returns float, float, outlying atom index"""
    atoms = [conformer.GetAtomPosition(i) for i in atom_ids]
    min_error = 100.0

    for index, atom in enumerate(atom_ids):
        other_atoms = atoms[:index] + atoms[index + 1 :]
        norm, d, error = least_squares_plane(*other_atoms)
        if error < min_error:
            min_error = error
            max_norm = norm
            max_d = d
            out_atom = atom

    return max_norm, max_d, out_atom


# REDO!
def find_sub_atoms(mol, atom_index: int, out_index: int, rotate_atoms: list[int]):
    "Finds neighbourhood of an atom recursively"
    # tread lightly
    atom = mol.GetAtomWithIdx(atom_index)
    for neighbor_atom in atom.GetNeighbors():
        nbr_index = neighbor_atom.GetIdx()
        if (nbr_index not in rotate_atoms) and nbr_index != out_index:
            rotate_atoms.append(nbr_index)
            find_sub_atoms(mol, nbr_index, out_index, rotate_atoms)


def find_rot_angle(
    conformer,
    backbone_index1: int,
    backbone_index2: int,
    out_atom_index: int,
    norm: Point3D,
):
    backbone_atom1 = conformer.GetAtomPosition(backbone_index1)
    backbone_atom2 = conformer.GetAtomPosition(backbone_index2)
    out_atom = conformer.GetAtomPosition(out_atom_index)

    halfway_vec = (backbone_atom1 + backbone_atom2) / 2
    vector = out_atom - halfway_vec
    vangle = vector.AngleTo(norm)
    return vangle


def rotate_atom_coords(
    conformer, atom: int, backbone_index1: int, backbone_index2: int, angle
):
    backbone_atom1 = conformer.GetAtomPosition(backbone_index1)
    backbone_atom2 = conformer.GetAtomPosition(backbone_index2)
    k = backbone_atom1 - backbone_atom2
    k.Normalize()
    v = conformer.GetAtomPosition(atom)
    # Rodrigues formula
    v_rot = (
        v * cos(angle)
        + k.CrossProduct(v) * sin(angle)
        + k * k.DotProduct(v) * (1 - cos(angle))
    )
    return v_rot
