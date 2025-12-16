# ================================================================
# 0. Section: Imports
# ================================================================
import unittest
from src.neuroframe.pipeline.extract_skull import *
from src.neuroframe.mouse_data import MicroCT





# ================================================================
# 1. Section: Test Cases
# ================================================================
class TestExtractSkull(unittest.TestCase):
    def test_cumsum_projection_values(self):
        micro_ct = MicroCT('tests/integration/fixtures/test_experiment/test_mouse_p324/p324_uCT.nii.gz').data

        skull = cumsum_projection(micro_ct, [30, 20, 4000])

        count_nonzero = np.count_nonzero(skull)
        self.assertGreater(count_nonzero, 0, "Cumsum projection should have non-zero values")

    def test_mean_projection_values(self):
        micro_ct = MicroCT('tests/integration/fixtures/test_experiment/test_mouse_p324/p324_uCT.nii.gz').data

        skull = mean_projection(micro_ct)

        count_nonzero = np.count_nonzero(skull)
        self.assertGreater(count_nonzero, 0, "Mean projection should have non-zero values")

    def test_view_projection_values(self):
        micro_ct = MicroCT('tests/integration/fixtures/test_experiment/test_mouse_p324/p324_uCT.nii.gz').data

        skull, depth = view_projection(micro_ct)

        count_nonzero_skull = np.count_nonzero(skull)
        count_nonzero_depth = np.count_nonzero(depth)
        self.assertGreater(count_nonzero_skull, 0, "View projection should have non-zero skull values")
        self.assertGreater(count_nonzero_depth, 0, "View projection should have non-zero depth values")