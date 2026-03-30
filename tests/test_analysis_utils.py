"""Tests for dp5.analysis.utils"""

import pickle
import pytest
import numpy as np

from dp5.analysis.utils import scale_nmr, _scale_nmr, AnalysisData


class TestScaleNMR:
    def test_single_shift_returns_unchanged(self):
        calc = np.array([100.0])
        exp = np.array([95.0])
        result = scale_nmr(calc, exp)
        np.testing.assert_array_almost_equal(result, calc)

    def test_perfect_linear_relationship(self):
        # calc = 2*exp + 5  =>  scaled should equal exp after correction
        exp = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        calc = 2.0 * exp + 5.0
        result = scale_nmr(calc, exp)
        np.testing.assert_array_almost_equal(result, exp, decimal=5)

    def test_identity_relationship(self):
        shifts = np.array([10.0, 20.0, 30.0, 40.0])
        result = scale_nmr(shifts, shifts)
        np.testing.assert_array_almost_equal(result, shifts, decimal=5)

    def test_accepts_lists(self):
        calc = [100.0, 110.0, 120.0, 130.0]
        exp = [95.0, 105.0, 115.0, 125.0]
        result = scale_nmr(calc, exp)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_multi_conformer_2d(self):
        # Two conformers, four shifts each
        exp = np.array([10.0, 20.0, 30.0, 40.0])
        calc = np.array([
            [11.0, 21.0, 31.0, 41.0],
            [12.0, 22.0, 32.0, 42.0],
        ])
        result = scale_nmr(calc, exp)
        assert result.shape == (2, 4)

    def test_output_shape_1d(self):
        calc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        exp = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = scale_nmr(calc, exp)
        assert result.shape == calc.shape

    def test_nan_intercept_handled(self):
        # When exp values have very low variance, slope/intercept may be degenerate.
        # _scale_nmr uses nan_to_num so the result must be finite.
        calc = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.0, 1.0001, 1.0002])  # near-constant but valid
        result = scale_nmr(calc, exp)
        assert not np.any(np.isnan(result))


class TestPrivateScaleNMR:
    def test_single_value_passthrough(self):
        calc = np.array([50.0])
        exp = np.array([50.0])
        result = _scale_nmr(calc, exp)
        np.testing.assert_array_equal(result, calc)

    def test_linear_correction_applied(self):
        exp = np.array([1.0, 2.0, 3.0, 4.0])
        calc = 3.0 * exp + 2.0
        result = _scale_nmr(calc, exp)
        np.testing.assert_array_almost_equal(result, exp, decimal=5)


class _MockMol:
    """Minimal stand-in for the Molecule objects AnalysisData expects."""
    def __init__(self, name):
        self.input_file = name


class TestAnalysisData:
    def _make_data(self, n=2):
        mols = [_MockMol(f"mol{i}") for i in range(n)]
        return mols

    def test_init_stores_mol_names(self, tmp_path):
        mols = self._make_data(3)
        ad = AnalysisData(mols, tmp_path / "data.p")
        assert ad.mols == ["mol0", "mol1", "mol2"]

    def test_exists_false_before_save(self, tmp_path):
        mols = self._make_data()
        ad = AnalysisData(mols, tmp_path / "data.p")
        assert ad.exists is False

    def test_save_and_load(self, tmp_path):
        mols = self._make_data(2)
        ad = AnalysisData(mols, tmp_path / "data.p")
        ad.extra_attr = [1, 2]
        ad.save()

        ad2 = AnalysisData(mols, tmp_path / "data.p")
        ad2.load()
        assert ad2.extra_attr == [1, 2]

    def test_exists_true_after_save(self, tmp_path):
        mols = self._make_data()
        ad = AnalysisData(mols, tmp_path / "data.p")
        ad.save()
        assert ad.exists is True

    def test_append_new_key(self, tmp_path):
        mols = self._make_data(2)
        ad = AnalysisData(mols, tmp_path / "data.p")
        ad.append({"score": 0.9})
        assert ad.score == [0.9]

    def test_append_existing_key(self, tmp_path):
        mols = self._make_data(2)
        ad = AnalysisData(mols, tmp_path / "data.p")
        ad.append({"score": 0.9})
        ad.append({"score": 0.7})
        assert ad.score == [0.9, 0.7]

    def test_values_dict_excludes_path_and_exists(self, tmp_path):
        mols = self._make_data()
        ad = AnalysisData(mols, tmp_path / "data.p")
        vd = ad.values_dict
        assert "path" not in vd
        assert "exists" not in vd

    def test_by_mol(self, tmp_path):
        mols = self._make_data(2)
        ad = AnalysisData(mols, tmp_path / "data.p")
        ad.mols = ["a", "b"]
        ad.score = [1, 2]
        by_mol = ad.by_mol
        assert len(by_mol) == 2
        assert by_mol[0] == {"mols": "a", "score": 1}

    def test_from_mol_dicts(self, tmp_path):
        mols = self._make_data(2)
        ad = AnalysisData(mols, tmp_path / "data.p")
        dicts = [{"val": 10}, {"val": 20}]
        ad.from_mol_dicts(dicts)
        assert ad.val == [10, 20]

    def test_iter(self, tmp_path):
        mols = self._make_data(2)
        ad = AnalysisData(mols, tmp_path / "data.p")
        ad.mols = ["a", "b"]
        ad.x = [1, 2]
        items = list(ad)
        assert items[0] == {"mols": "a", "x": 1}
        assert items[1] == {"mols": "b", "x": 2}
