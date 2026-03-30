"""Tests for dp5.nmr_processing.description_files"""

import os
import pytest
import tempfile

from dp5.nmr_processing.description_files import (
    _parse_description,
    pairwise_assignment,
    matching_assignment,
    process_description,
)


# ---------------------------------------------------------------------------
# _parse_description
# ---------------------------------------------------------------------------

class TestParseDescription:
    def test_empty_string_returns_empty_lists(self):
        labels, shifts = _parse_description("")
        assert labels == []
        assert shifts == []

    def test_single_shift_no_label(self):
        labels, shifts = _parse_description("100.0")
        assert shifts == [100.0]

    def test_multiple_shifts(self):
        labels, shifts = _parse_description("10.5,20.3,30.1")
        assert shifts == pytest.approx([10.5, 20.3, 30.1])

    def test_label_in_parentheses(self):
        labels, shifts = _parse_description("100.0(C1),200.0(C2)")
        assert labels[0] == ["C1"]
        assert labels[1] == ["C2"]
        assert shifts == pytest.approx([100.0, 200.0])

    def test_multi_label_with_or(self):
        labels, shifts = _parse_description("100.0(C1 or C2)")
        # 'or' is replaced by ',', so both C1 and C2 go in one group
        assert set(labels[0]) == {"C1", "C2"}

    def test_shift_count_matches_label_count(self):
        exp = "10.0(H1),20.0(H2),30.0(H3)"
        labels, shifts = _parse_description(exp)
        assert len(labels) == len(shifts) == 3


# ---------------------------------------------------------------------------
# pairwise_assignment
# ---------------------------------------------------------------------------

class TestPairwiseAssignment:
    def test_same_length_sorted_match(self):
        calc = [1.0, 2.0, 3.0]
        exp = [1.1, 2.1, 3.1]
        result = pairwise_assignment(calc, exp)
        assert len(result) == 3
        assert all(r is not None for r in result)

    def test_largest_calc_gets_largest_exp(self):
        calc = [5.0, 1.0]
        exp = [4.5, 0.5]
        result = pairwise_assignment(calc, exp)
        # calc[0]=5 is the largest => should get 4.5
        assert result[0] == pytest.approx(4.5)

    def test_output_length_equals_calc_length(self):
        calc = [1.0, 2.0, 3.0, 4.0]
        exp = [1.0, 2.0, 3.0, 4.0]
        result = pairwise_assignment(calc, exp)
        assert len(result) == 4

    def test_single_value(self):
        result = pairwise_assignment([5.0], [4.9])
        assert result[0] == pytest.approx(4.9)


# ---------------------------------------------------------------------------
# matching_assignment
# ---------------------------------------------------------------------------

class TestMatchingAssignment:
    def test_within_threshold_assigned(self):
        calc = [10.0, 20.0, 30.0]
        exp = [10.5, 20.5, 30.5]
        result = matching_assignment(calc, exp, threshold=5)
        assert len(result) == 3
        assert all(r is not None for r in result)

    def test_beyond_threshold_not_assigned(self):
        calc = [10.0]
        exp = [100.0]
        result = matching_assignment(calc, exp, threshold=5)
        assert result[0] is None

    def test_output_length_equals_calc_length(self):
        calc = [1.0, 2.0, 3.0]
        exp = [1.1, 2.1, 3.1]
        result = matching_assignment(calc, exp, threshold=10)
        assert len(result) == len(calc)

    def test_returns_list(self):
        result = matching_assignment([1.0], [1.1])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# process_description (reads from file)
# ---------------------------------------------------------------------------

class TestProcessDescription:
    def _write_description_file(self, tmp_path, content):
        p = tmp_path / "nmr.txt"
        p.write_text(content)
        return str(p)

    def test_basic_parsing(self, tmp_path):
        content = (
            "100.0(C1),200.0(C2)\n"
            "\n"
            "1.5(H1),2.5(H2)\n"
            "\n"
        )
        path = self._write_description_file(tmp_path, content)
        C_labels, C_exp, H_labels, H_exp, equivalents, omits = process_description(path)
        assert C_exp == pytest.approx([100.0, 200.0])
        assert H_exp == pytest.approx([1.5, 2.5])

    def test_omit_parsed(self, tmp_path):
        content = (
            "100.0(C1),200.0(C2)\n"
            "\n"
            "1.5(H1)\n"
            "\n"
            "OMIT C2\n"
        )
        path = self._write_description_file(tmp_path, content)
        _, _, _, _, _, omits = process_description(path)
        assert "C2" in omits

    def test_equivalents_parsed(self, tmp_path):
        content = (
            "100.0(C1),200.0(C2)\n"
            "\n"
            "1.5(H1)\n"
            "\n"
            "C1,C3\n"
        )
        path = self._write_description_file(tmp_path, content)
        _, _, _, _, equivalents, _ = process_description(path)
        assert ["C1", "C3"] in equivalents

    def test_empty_nmr_sections(self, tmp_path):
        # process_description with no shift data raises ValueError (as expected from
        # _parse_description being given a bare newline with no numbers).
        content = "\n\n\n\n"
        path = self._write_description_file(tmp_path, content)
        with pytest.raises(ValueError):
            process_description(path)
