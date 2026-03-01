import unittest
from lgta.transformations.manipulate_data import ManipulateData
import numpy as np
from scipy.interpolate import CubicSpline


class TestCreateTransformedDatasets(unittest.TestCase):

    def setUp(self):
        self.dataset = "tourism_small"
        self.sigma_magnitude_warp = 0.2
        np.random.seed(0)
        self.data = np.random.randn(200, 100)

    def test_magnitude_warping_individual_series(self):
        res = ManipulateData(
            self.data, "magnitude_warp", [self.sigma_magnitude_warp]
        ).apply_transf()

        random_warp_1 = np.array(
            [0.999904, 0.76587626, 0.92712345, 0.69992099, 0.87777622, 1.12951142]
        )

        warp_steps = np.array([0.0, 39.8, 79.6, 119.4, 159.2, 199.0])
        ret_1 = CubicSpline(warp_steps, random_warp_1)(np.arange(self.data.shape[0]))
        res_1 = self.data[:, 1] * ret_1

        self.assertIsNone(np.testing.assert_almost_equal(res_1, res[:, 1]))

    def test_drift_shape_preserved(self):
        sigma = 0.5
        res = ManipulateData(self.data, "drift", [sigma]).apply_transf()
        self.assertEqual(res.shape, self.data.shape)

    def test_drift_cumulative_structure(self):
        """Drift residuals must be autocorrelated (cumulative noise)."""
        np.random.seed(42)
        sigma = 1.0
        res = ManipulateData(self.data, "drift", [sigma]).apply_transf()
        residual = res[:, 0] - self.data[:, 0]
        autocorr = np.corrcoef(residual[:-1], residual[1:])[0, 1]
        self.assertGreater(autocorr, 0.5)

    def test_trend_shape_preserved(self):
        sigma = 0.5
        res = ManipulateData(self.data, "trend", [sigma]).apply_transf()
        self.assertEqual(res.shape, self.data.shape)

    def test_trend_linearity(self):
        """Trend residuals should be approximately linear."""
        np.random.seed(42)
        sigma = 2.0
        res = ManipulateData(self.data, "trend", [sigma]).apply_transf()
        residual = res[:, 0] - self.data[:, 0]
        t = np.arange(len(residual), dtype=float)
        correlation = np.abs(np.corrcoef(t, residual)[0, 1])
        self.assertGreater(correlation, 0.99)
