"""Tests for dp5.nmr_processing.helper_functions"""

import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AddHs

from dp5.nmr_processing.helper_functions import (
    generalised_lorentzian,
    lorentzian,
    normalise_intensities,
    methyl_protons,
    labile_protons,
    proton_count,
    preprocess,
    preprocess_proton,
    preprocess_carbon,
)


# ---------------------------------------------------------------------------
# generalised_lorentzian
# ---------------------------------------------------------------------------

class TestGeneralisedLorentzian:
    def test_peak_at_mu(self):
        """Amplitude at the peak centre should equal A."""
        result = generalised_lorentzian(x=5.0, mu=5.0, std=1.0, v=0.5, A=2.0)
        assert result == pytest.approx(2.0, rel=1e-6)

    def test_symmetry_around_mu(self):
        mu, std, v, A = 3.0, 1.0, 0.5, 1.0
        left = generalised_lorentzian(mu - 1.0, mu, std, v, A)
        right = generalised_lorentzian(mu + 1.0, mu, std, v, A)
        assert left == pytest.approx(right, rel=1e-6)

    def test_tails_smaller_than_peak(self):
        peak = generalised_lorentzian(0.0, 0.0, 1.0, 0.5, 1.0)
        tail = generalised_lorentzian(10.0, 0.0, 1.0, 0.5, 1.0)
        assert peak > tail

    def test_array_input(self):
        x = np.linspace(-5, 5, 1000)
        result = generalised_lorentzian(x, mu=0.0, std=1.0, v=0.0, A=1.0)
        assert result.shape == x.shape
        # Peak value should be close to A=1 (sampled array won't hit exactly 0)
        assert result.max() == pytest.approx(1.0, rel=1e-3)

    def test_amplitude_scales_output(self):
        r1 = generalised_lorentzian(0.0, 0.0, 1.0, 0.5, 1.0)
        r2 = generalised_lorentzian(0.0, 0.0, 1.0, 0.5, 3.0)
        assert r2 == pytest.approx(3.0 * r1, rel=1e-6)

    def test_v_zero_is_pure_lorentzian(self):
        """v=0 => pure Lorentzian shape; peak equals A."""
        result = generalised_lorentzian(0.0, 0.0, 1.0, 0.0, 1.0)
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_v_one_is_voigt_like(self):
        """v=1 => alternative functional form; peak still equals A."""
        result = generalised_lorentzian(0.0, 0.0, 1.0, 1.0, 1.0)
        assert result == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# lorentzian
# ---------------------------------------------------------------------------

class TestLorentzian:
    def test_peak_at_centre(self):
        result = lorentzian(p=5.0, w=1.0, p0=5.0, A=1.0)
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_symmetry(self):
        left = lorentzian(p=4.0, w=1.0, p0=5.0, A=1.0)
        right = lorentzian(p=6.0, w=1.0, p0=5.0, A=1.0)
        assert left == pytest.approx(right, rel=1e-6)

    def test_amplitude_scaling(self):
        r1 = lorentzian(5.5, 1.0, 5.0, 1.0)
        r2 = lorentzian(5.5, 1.0, 5.0, 2.0)
        assert r2 == pytest.approx(2.0 * r1, rel=1e-6)

    def test_array_input(self):
        p = np.linspace(0, 10, 200)
        result = lorentzian(p, w=1.0, p0=5.0, A=1.0)
        assert result.shape == p.shape


# ---------------------------------------------------------------------------
# normalise_intensities
# ---------------------------------------------------------------------------

class TestNormaliseIntensities:
    def test_max_modulus_is_one_for_real(self):
        data = np.array([1.0, 2.0, 3.0, -6.0])
        result = normalise_intensities(data)
        assert np.abs(result).max() == pytest.approx(1.0, rel=1e-6)

    def test_max_modulus_is_one_for_complex(self):
        data = np.array([1 + 0j, 0 + 2j, 3 + 4j])
        result = normalise_intensities(data)
        assert np.abs(result).max() == pytest.approx(1.0, rel=1e-6)

    def test_relative_magnitudes_preserved(self):
        data = np.array([1.0, 2.0, 4.0])
        result = normalise_intensities(data)
        assert result[2] == pytest.approx(2.0 * result[1], rel=1e-6)

    def test_returns_ndarray(self):
        data = np.array([1.0, 2.0])
        result = normalise_intensities(data)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# preprocess / preprocess_proton / preprocess_carbon
# ---------------------------------------------------------------------------

class TestPreprocess:
    def _make_fid(self, length=2048):
        """Minimal synthetic FID (real-valued, uniform noise)."""
        rng = np.random.default_rng(42)
        return rng.standard_normal(length).astype(complex)

    def test_preprocess_returns_ndarray(self):
        fid = self._make_fid()
        result = preprocess(fid, zero_filling=1)
        assert isinstance(result, np.ndarray)

    def test_preprocess_max_modulus_one(self):
        fid = self._make_fid()
        result = preprocess(fid, zero_filling=1)
        assert np.abs(result).max() == pytest.approx(1.0, rel=1e-4)

    def test_preprocess_proton_longer_than_input(self):
        fid = self._make_fid(512)
        result = preprocess_proton(fid)
        # zero_filling=4 => length 512 * 2**4 = 8192
        assert len(result) == 512 * (2 ** 4)

    def test_preprocess_carbon_longer_than_input(self):
        fid = self._make_fid(512)
        result = preprocess_carbon(fid)
        # zero_filling=2 => length 512 * 2**2 = 2048
        assert len(result) == 512 * (2 ** 2)


# ---------------------------------------------------------------------------
# methyl_protons
# ---------------------------------------------------------------------------

class TestMethylProtons:
    def _mol_from_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return AddHs(mol)

    def test_propane_has_two_methyls(self):
        # Propane: two terminal CH3 groups each with exactly 3 H neighbours
        mol = self._mol_from_smiles("CCC")
        groups = methyl_protons(mol)
        assert len(groups) == 2
        for g in groups:
            assert len(g) == 3

    def test_ethane_has_two_methyls(self):
        mol = self._mol_from_smiles("CC")
        groups = methyl_protons(mol)
        assert len(groups) == 2

    def test_benzene_has_no_methyls(self):
        mol = self._mol_from_smiles("c1ccccc1")
        groups = methyl_protons(mol)
        assert groups == []

    def test_methyl_labels_start_with_H(self):
        mol = self._mol_from_smiles("CCC")
        groups = methyl_protons(mol)
        for label in groups[0]:
            assert label.startswith("H")


# ---------------------------------------------------------------------------
# labile_protons
# ---------------------------------------------------------------------------

class TestLabileProtons:
    def _mol_from_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return AddHs(mol)

    def test_ethanol_has_one_labile(self):
        mol = self._mol_from_smiles("CCO")
        assert labile_protons(mol) == 1

    def test_water_has_two_labile(self):
        mol = self._mol_from_smiles("O")
        assert labile_protons(mol) == 2

    def test_ethane_has_no_labile(self):
        mol = self._mol_from_smiles("CC")
        assert labile_protons(mol) == 0

    def test_glycol_has_two_labile(self):
        mol = self._mol_from_smiles("OCCO")
        assert labile_protons(mol) == 2


# ---------------------------------------------------------------------------
# proton_count
# ---------------------------------------------------------------------------

class TestProtonCount:
    def _mol_from_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return AddHs(mol)

    def test_methane_four_protons(self):
        mol = self._mol_from_smiles("C")
        assert proton_count(mol) == 4

    def test_benzene_six_protons(self):
        mol = self._mol_from_smiles("c1ccccc1")
        assert proton_count(mol) == 6

    def test_ethane_six_protons(self):
        mol = self._mol_from_smiles("CC")
        assert proton_count(mol) == 6

    def test_ethanol_six_protons(self):
        mol = self._mol_from_smiles("CCO")
        assert proton_count(mol) == 6
