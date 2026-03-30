"""Tests for dp5.analysis.dp4"""

import math
import pickle
import pytest
import numpy as np

from dp5.analysis.dp4 import DP4ProbabilityCalculator, DP4, DP4Data
from dp5.analysis.utils import AnalysisData


# ---------------------------------------------------------------------------
# DP4ProbabilityCalculator
# ---------------------------------------------------------------------------

class TestDP4ProbabilityCalculatorInit:
    def test_scalar_float_inputs(self):
        calc = DP4ProbabilityCalculator(mean=0.0, stdev=2.0)
        assert callable(calc)

    def test_single_element_list_inputs(self):
        calc = DP4ProbabilityCalculator(mean=[0.0], stdev=[2.0])
        assert callable(calc)

    def test_multi_element_list_inputs(self):
        calc = DP4ProbabilityCalculator(mean=[0.0, 0.5], stdev=[2.0, 1.5])
        assert callable(calc)

    def test_mismatched_list_lengths_raises(self):
        with pytest.raises(ValueError):
            DP4ProbabilityCalculator(mean=[0.0, 1.0], stdev=[2.0])

    def test_mismatched_types_raises(self):
        with pytest.raises(TypeError):
            DP4ProbabilityCalculator(mean=[0.0], stdev=2.0)

    def test_mismatched_types_raises_reverse(self):
        with pytest.raises(TypeError):
            DP4ProbabilityCalculator(mean=0.0, stdev=[2.0])


class TestDP4ProbabilityCalculatorCall:
    def test_zero_error_gives_maximum_probability(self):
        calc = DP4ProbabilityCalculator(mean=0.0, stdev=2.0)
        # zero error should give p = 1 (two-tailed cdf at z=0)
        result = calc(np.array([0.0]))
        assert result.item() == pytest.approx(1.0, abs=1e-6)

    def test_probabilities_between_zero_and_one(self):
        calc = DP4ProbabilityCalculator(mean=0.0, stdev=2.0)
        errors = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        result = calc(errors)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_symmetric_probabilities(self):
        calc = DP4ProbabilityCalculator(mean=0.0, stdev=2.0)
        pos = calc(np.array([3.0])).item()
        neg = calc(np.array([-3.0])).item()
        assert pos == pytest.approx(neg, rel=1e-6)

    def test_multiple_gaussian_positive_output(self):
        calc = DP4ProbabilityCalculator(mean=[0.0, 1.0], stdev=[2.0, 1.5])
        errors = np.array([0.0, 1.0, -1.0])
        result = calc(errors)
        assert np.all(result > 0)

    def test_single_gaussian_static_method(self):
        val = DP4ProbabilityCalculator.single_gaussian(0.0, 0.0, 1.0)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_single_gaussian_larger_error_smaller_probability(self):
        p_small = DP4ProbabilityCalculator.single_gaussian(0.5, 0.0, 1.0)
        p_large = DP4ProbabilityCalculator.single_gaussian(5.0, 0.0, 1.0)
        assert p_small > p_large

    def test_multiple_gaussian_static_method(self):
        val = DP4ProbabilityCalculator.multiple_gaussian(0.0, [0.0, 0.0], [1.0, 1.0])
        assert val > 0.0


# ---------------------------------------------------------------------------
# DP4._dp4 (core scoring logic, tested indirectly via dp4_proton / dp4_carbon)
# ---------------------------------------------------------------------------

class TestDP4PrivateDp4:
    """Exercise DP4._dp4 through the public dp4_proton and dp4_carbon helpers."""

    def _make_dp4(self, tmp_path):
        config = {
            "stats_model": "g",
            "param_file": "",
        }
        return DP4(tmp_path, config)

    def test_dp4_proton_returns_7_values(self, tmp_path):
        dp4 = self._make_dp4(tmp_path)
        calc = np.array([1.0, 2.0, 3.0, 4.0])
        exp = np.array([1.1, 2.1, 3.1, 4.1])
        labels = np.array(["H1", "H2", "H3", "H4"])
        result = dp4.dp4_proton(calc, exp, labels)
        assert len(result) == 7

    def test_dp4_carbon_returns_7_values(self, tmp_path):
        dp4 = self._make_dp4(tmp_path)
        calc = np.array([10.0, 20.0, 30.0])
        exp = np.array([10.5, 20.5, 30.5])
        labels = np.array(["C1", "C2", "C3"])
        result = dp4.dp4_carbon(calc, exp, labels)
        assert len(result) == 7

    def test_dp4_proton_score_is_positive(self, tmp_path):
        dp4 = self._make_dp4(tmp_path)
        calc = np.array([1.0, 2.0, 3.0, 4.0])
        exp = np.array([1.1, 2.1, 3.1, 4.1])
        labels = np.array(["H1", "H2", "H3", "H4"])
        *_, score = dp4.dp4_proton(calc, exp, labels)
        assert score > 0.0

    def test_dp4_ignores_nan_experimental(self, tmp_path):
        dp4 = self._make_dp4(tmp_path)
        calc = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.0, np.nan, 3.0])
        labels = np.array(["H1", "H2", "H3"])
        ret_labels, *_, score = dp4.dp4_proton(calc, exp, labels)
        # H2 should be filtered out
        assert "H2" not in ret_labels
        assert math.isfinite(score)

    def test_all_nan_experimental_gives_score_one(self, tmp_path):
        dp4 = self._make_dp4(tmp_path)
        calc = np.array([1.0, 2.0])
        exp = np.array([np.nan, np.nan])
        labels = np.array(["H1", "H2"])
        *_, score = dp4.dp4_proton(calc, exp, labels)
        assert score == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# DP4.read_parameters
# ---------------------------------------------------------------------------

class TestDP4ReadParameters:
    def test_reads_single_gaussian_params(self, tmp_path):
        param_file = tmp_path / "params.txt"
        param_file.write_text(
            "# comment\n"
            "1.0\n"
            "2.5\n"
            "0.1\n"
            "0.2\n"
        )
        config = {"stats_model": "g", "param_file": str(param_file)}
        dp4 = DP4(tmp_path, config)
        # Just verify it constructed without error
        assert dp4 is not None

    def test_mismatched_params_raises(self, tmp_path):
        param_file = tmp_path / "params_bad.txt"
        param_file.write_text(
            "# comment\n"
            "1.0,2.0\n"        # 2 means
            "3.0\n"            # 1 stdev — mismatch
            "0.1\n"
            "0.2\n"
        )
        sub = tmp_path / "sub"
        sub.mkdir()
        config = {"stats_model": "g", "param_file": str(param_file)}
        with pytest.raises(ValueError):
            DP4(sub, config)


# ---------------------------------------------------------------------------
# DP4Data.print_assignment
# ---------------------------------------------------------------------------

class TestDP4DataPrintAssignment:
    def test_output_is_string(self):
        labels = np.array(["C1", "C2", "C3"])
        calc = np.array([10.0, 20.0, 30.0])
        scaled = np.array([10.1, 20.1, 30.1])
        exp = np.array([10.2, 20.2, 30.2])
        error = scaled - exp
        result = DP4Data.print_assignment(labels, calc, scaled, exp, error)
        assert isinstance(result, str)

    def test_output_contains_labels(self):
        labels = np.array(["C1", "C2"])
        calc = np.array([10.0, 20.0])
        scaled = np.array([10.1, 20.1])
        exp = np.array([10.2, 20.2])
        error = scaled - exp
        result = DP4Data.print_assignment(labels, calc, scaled, exp, error)
        assert "C1" in result
        assert "C2" in result
