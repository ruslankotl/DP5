"""Tests for dp5.nmr_processing.carbon.assign"""

import pytest
import numpy as np

from dp5.nmr_processing.carbon.assign import (
    external_scale_carbon_shifts,
    internal_scale_carbon_shifts,
    carbon_probabilities,
    removecrossassignments,
)


# ---------------------------------------------------------------------------
# external_scale_carbon_shifts
# ---------------------------------------------------------------------------

class TestExternalScaleCarbonShifts:
    def test_applies_linear_transform(self):
        calc = np.array([0.0, 100.0, 200.0])
        result = external_scale_carbon_shifts(calc)
        expected = calc * 0.9601578792266342 - 1.2625604390657088
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_shape_preserved(self):
        calc = np.arange(10, dtype=float)
        result = external_scale_carbon_shifts(calc)
        assert result.shape == calc.shape

    def test_returns_ndarray(self):
        calc = np.array([50.0])
        result = external_scale_carbon_shifts(calc)
        assert isinstance(result, np.ndarray)

    def test_zero_input(self):
        calc = np.array([0.0])
        result = external_scale_carbon_shifts(calc)
        assert result[0] == pytest.approx(-1.2625604390657088)


# ---------------------------------------------------------------------------
# internal_scale_carbon_shifts
# ---------------------------------------------------------------------------

class TestInternalScaleCarbonShifts:
    def test_returns_three_values(self):
        assigned = np.array([10.0, 20.0, 30.0, 40.0])
        peaks = np.array([11.0, 21.0, 31.0, 41.0])
        calc = np.array([10.0, 20.0, 30.0, 40.0])
        result = internal_scale_carbon_shifts(assigned, peaks, calc)
        assert len(result) == 3

    def test_output_shape_matches_calc(self):
        assigned = np.array([10.0, 20.0, 30.0])
        peaks = np.array([11.0, 21.0, 31.0])
        calc = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        scaled, slope, intercept = internal_scale_carbon_shifts(assigned, peaks, calc)
        assert scaled.shape == calc.shape

    def test_identity_relationship(self):
        """If assigned == peaks, scaling should be nearly identity."""
        vals = np.array([10.0, 20.0, 30.0, 40.0])
        scaled, slope, intercept = internal_scale_carbon_shifts(vals, vals, vals)
        np.testing.assert_array_almost_equal(scaled, vals, decimal=5)

    def test_slope_and_intercept_numeric(self):
        assigned = np.array([10.0, 20.0, 30.0])
        peaks = np.array([12.0, 22.0, 32.0])
        calc = np.array([10.0, 20.0, 30.0])
        _, slope, intercept = internal_scale_carbon_shifts(assigned, peaks, calc)
        assert np.isfinite(slope)
        assert np.isfinite(intercept)


# ---------------------------------------------------------------------------
# carbon_probabilities
# ---------------------------------------------------------------------------

class TestCarbonProbabilities:
    def test_output_shape(self):
        diff = np.array([[0.5, -0.5], [1.0, -1.0], [2.0, -2.0]])
        result = carbon_probabilities(diff, 0.0, 2.0)
        assert result.shape == diff.shape

    def test_rows_sum_to_one(self):
        diff = np.array([[0.5, -0.5, 1.0], [2.0, -1.0, 0.0]])
        result = carbon_probabilities(diff, 0.0, 2.0)
        np.testing.assert_array_almost_equal(result.sum(axis=1), np.ones(2))

    def test_non_negative(self):
        diff = np.random.default_rng(0).uniform(-5, 5, (4, 6))
        result = carbon_probabilities(diff, 0.0, 2.0)
        assert np.all(result >= 0)

    def test_zero_difference_gives_highest_probability(self):
        diff = np.array([[0.0, 2.0, -2.0]])
        result = carbon_probabilities(diff, 0.0, 2.0)
        assert result[0, 0] == result[0, :].max()


# ---------------------------------------------------------------------------
# removecrossassignments
# ---------------------------------------------------------------------------

class TestRemoveCrossAssignments:
    def test_returns_three_arrays(self):
        exp = np.array([10.0, 20.0, 30.0])
        calc = np.array([10.0, 20.0, 30.0])
        labels = np.array(["C1", "C2", "C3"])
        result = removecrossassignments(exp, calc, labels)
        assert len(result) == 3

    def test_no_cross_assignments_unchanged(self):
        """Descending calc with matching descending exp => no swaps needed."""
        calc = np.array([30.0, 20.0, 10.0])
        exp = np.array([31.0, 21.0, 11.0])
        labels = np.array(["C3", "C2", "C1"])
        ret_calc, ret_exp, ret_labels = removecrossassignments(exp, calc, labels)
        # After sorting by calc desc, order should still be monotone
        assert np.all(np.diff(ret_calc) <= 0)
        assert np.all(np.diff(ret_exp) <= 0)

    def test_cross_assignment_corrected(self):
        """If the exp assignments cross the calc order, they should be uncrossed."""
        calc = np.array([30.0, 10.0])   # calc desc order: 30, 10
        exp = np.array([5.0, 25.0])     # originally crossed
        labels = np.array(["C1", "C2"])
        ret_calc, ret_exp, ret_labels = removecrossassignments(exp, calc, labels)
        # exp should be in descending order matching calc
        assert np.all(np.diff(ret_exp) <= 0)

    def test_output_preserves_length(self):
        calc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        exp = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        labels = np.array(["C1", "C2", "C3", "C4", "C5"])
        ret_calc, ret_exp, ret_labels = removecrossassignments(exp, calc, labels)
        assert len(ret_calc) == len(calc)
        assert len(ret_exp) == len(exp)
        assert len(ret_labels) == len(labels)
