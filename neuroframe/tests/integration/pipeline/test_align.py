# ================================================================
# 0. Section: Imports
# ================================================================
import unittest
from src.neuroframe.pipeline.align import *



# ================================================================
# 1. Section: Test Cases
# ================================================================
class TestAlign(unittest.TestCase):
    def test_align_to_allen_changes_mouse(self):
        # Load the mouse and align
        mouse = Mouse.from_folder('P874', 'tests/integration/fixtures/test_experiment/test_mouse_p874')
        align_to_allen(mouse)

        original_mouse = Mouse.from_folder('P874', 'tests/integration/fixtures/test_experiment/test_mouse_p874')
        # These condition should all be false after alignment
        condition_vol = np.array_equal(original_mouse.segmentation.volume, mouse.segmentation.volume)
        condition_seg = np.array_equal(original_mouse.segmentation.data, mouse.segmentation.data)
        condition_mri = np.array_equal(original_mouse.mri.data, mouse.mri.data)
        condition_ct = np.array_equal(original_mouse.micro_ct.data, mouse.micro_ct.data)

        # Assert that the mouse has changed
        self.assertFalse(condition_vol or condition_seg or condition_mri or condition_ct, "Mouse segmentation volume should change after alignment")

    def test_adapt_template_returns_same_shapes(self):
        mouse = Mouse.from_folder('P874', 'tests/integration/fixtures/test_experiment/test_mouse_p874')
        template_vol = adapt_template(mouse, ALLEN_TEMPLATE)

        # Assert that the adapted template volume has the same shape as the mouse segmentation volume
        self.assertEqual(template_vol.shape, mouse.segmentation.volume.shape, "Adapted template volume should have the same shape as mouse segmentation volume")

    def test_adapt_template_returns_similar_volume_sizes(self):
        mouse = Mouse.from_folder('P874', 'tests/integration/fixtures/test_experiment/test_mouse_p874')
        template_vol = adapt_template(mouse, ALLEN_TEMPLATE)

        # Calculate the size of non-zero elements in both volumes
        mouse_nonzero_size = np.count_nonzero(mouse.segmentation.volume)
        template_nonzero_size = np.count_nonzero(template_vol)

        # Assert that the sizes are similar (within 15% tolerance)
        size_difference = abs(mouse_nonzero_size - template_nonzero_size)
        tolerance = 0.15 * mouse_nonzero_size
        self.assertLessEqual(size_difference, tolerance, "Adapted template volume size should be similar to mouse segmentation volume size")


if __name__ == "__main_":
    unittest.main()