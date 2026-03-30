"""Tests for dp5.nmr_processing.carbon.peak_picking"""

import pytest
import numpy as np

from dp5.nmr_processing.carbon.peak_picking import edge_removal, iterative_peak_picking


# ---------------------------------------------------------------------------
# edge_removal
# ---------------------------------------------------------------------------

class TestEdgeRemoval:
    def test_positive_leading_edge_zeroed(self):
        data = np.array([0.5, 0.3, 0.1, -0.1, -0.2, 0.0, 0.0])
        result = edge_removal(data.copy())
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_negative_leading_edge_zeroed(self):
        data = np.array([-0.5, -0.3, 0.1, 0.2, 0.0, 0.0, 0.0])
        result = edge_removal(data.copy())
        assert result[0] == 0.0

    def test_positive_trailing_edge_zeroed(self):
        data = np.array([0.0, 0.0, -0.1, 0.3, 0.5])
        result = edge_removal(data.copy())
        assert result[-1] == 0.0
        assert result[-2] == 0.0

    def test_negative_trailing_edge_zeroed(self):
        data = np.array([0.0, 0.0, 0.1, -0.3, -0.5])
        result = edge_removal(data.copy())
        assert result[-1] == 0.0
        assert result[-2] == 0.0

    def test_returns_array_of_same_length(self):
        data = np.array([0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        result = edge_removal(data.copy())
        assert len(result) == len(data)

    def test_interior_unchanged(self):
        """Values not touching an edge run should not be modified."""
        data = np.array([0.5, -0.1, 0.8, 0.3, -0.4])
        result = edge_removal(data.copy())
        # index 2 is interior (not adjacent to a monotone run at the edge)
        assert result[2] == 0.8


# ---------------------------------------------------------------------------
# iterative_peak_picking
# ---------------------------------------------------------------------------

class TestIterativePeakPicking:
    def _spectrum_with_peaks(self, length=4096, peak_positions=None, noise_scale=0.01):
        """Create a synthetic spectrum with Gaussian peaks."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, noise_scale, length).astype(float)
        if peak_positions is None:
            peak_positions = [length // 4, length // 2, 3 * length // 4]
        for pos in peak_positions:
            x = np.arange(length)
            y += np.exp(-0.5 * ((x - pos) / 10) ** 2)
        return y

    def test_returns_tuple(self):
        y = self._spectrum_with_peaks()
        result = iterative_peak_picking(y, threshold=5, corr_distance=5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_picked_peaks_are_sorted(self):
        y = self._spectrum_with_peaks()
        peaks, _ = iterative_peak_picking(y, threshold=5, corr_distance=5)
        assert peaks == sorted(peaks)

    def test_detects_main_peaks(self):
        positions = [500, 1500, 2500]
        y = self._spectrum_with_peaks(peak_positions=positions)
        peaks, _ = iterative_peak_picking(y, threshold=3, corr_distance=5)
        # Each true peak should have at least one detected peak within 30 samples
        for pos in positions:
            assert any(abs(p - pos) < 30 for p in peaks), (
                f"Peak near {pos} not detected; found peaks: {peaks}"
            )

    def test_no_peaks_in_flat_spectrum(self):
        # A very high threshold leaves at most the single global maximum
        y = np.zeros(4096)
        rng = np.random.default_rng(0)
        y += rng.normal(0, 1e-6, len(y))
        peaks, _ = iterative_peak_picking(y, threshold=100, corr_distance=5)
        # The algorithm always picks the global maximum first; after fitting it
        # out, subsequent candidates fall below the extreme threshold.
        assert len(peaks) <= 1

    def test_fit_y_same_length_as_input(self):
        y = self._spectrum_with_peaks()
        _, fit_y = iterative_peak_picking(y, threshold=5, corr_distance=5)
        assert len(fit_y) == len(y)

    def test_peak_indices_within_bounds(self):
        y = self._spectrum_with_peaks()
        peaks, _ = iterative_peak_picking(y, threshold=5, corr_distance=5)
        assert all(0 <= p < len(y) for p in peaks)
