# ================================================================
# 0. Section: Imports
# ================================================================
import unittest
from src.neuroframe.pipeline.align import *



# ================================================================
# 0. Section: Imports
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


if __name__ == "__main_":
    unittest.main()