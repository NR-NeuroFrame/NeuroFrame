# ================================================================
# 0. Section: Imports
# ================================================================
import unittest
import numpy as np

from neuroframe import Mouse
from neuroframe.pipeline.align_brain import align_to_bl
from neuroframe.utils import save_object, load_object, TEMP_FOLDER



# ================================================================
# 1. Section: Test Cases
# ================================================================
class Test04AlignBL(unittest.TestCase):
    def test_align_bl_changes_bl(self):
        mouse = load_object(TEMP_FOLDER + '01_align_mouse.pkl')
        bregma = load_object(TEMP_FOLDER + '04_extract_bl_bregma.pkl')
        lambda_ = load_object(TEMP_FOLDER + '04_extract_bl_lambda.pkl')

        new_bregma, new_lambda = align_to_bl(mouse, bregma, lambda_, deviation=40)
        save_object(new_bregma, TEMP_FOLDER + '05_align_bl_bregma.pkl')
        save_object(new_lambda, TEMP_FOLDER + '05_align_bl_lambda.pkl')
        save_object(mouse, TEMP_FOLDER + '05_align_bl_mouse.pkl')

        self.assertFalse(all(bregma == new_bregma), "Bregma coordinates should change after alignment")
        self.assertFalse(all(lambda_ == new_lambda), "Lambda coordinates should change after alignment")

    def test_align_bl_changes_mouse(self):
        mouse_original: Mouse = load_object(TEMP_FOLDER + '01_align_mouse.pkl')
        mouse: Mouse = load_object(TEMP_FOLDER + '05_align_bl_mouse.pkl')

        self.assertFalse(np.array_equal(mouse_original.micro_ct.data, mouse.micro_ct.data), "Mouse micro-CT data should change after alignment")
        self.assertFalse(np.array_equal(mouse_original.mri.data, mouse.mri.data), "Mouse MRI data should change after alignment")
        self.assertFalse(np.array_equal(mouse_original.segmentation.data, mouse.segmentation.data), "Mouse segmentation data should change after alignment")
        self.assertFalse(np.array_equal(mouse_original.segmentation.volume, mouse.segmentation.volume), "Mouse segmentation volume should change after alignment")

    def test_align_bl_is_in_space(self):
        mouse = load_object(TEMP_FOLDER + '05_align_bl_mouse.pkl')
        new_bregma = load_object(TEMP_FOLDER + '05_align_bl_bregma.pkl')
        new_lambda = load_object(TEMP_FOLDER + '05_align_bl_lambda.pkl')

        self.assertTrue(all(0 <= new_bregma[i] < mouse.data_shape[i] for i in range(3)), "New bregma coordinates should be within mouse data shape")
        self.assertTrue(all(0 <= new_lambda[i] < mouse.data_shape[i] for i in range(3)), "New lambda coordinates should be within mouse data shape")

    def test_align_bl_z_alignment(self):
        new_bregma = load_object(TEMP_FOLDER + '05_align_bl_bregma.pkl')
        new_lambda = load_object(TEMP_FOLDER + '05_align_bl_lambda.pkl')

        self.assertEqual(new_bregma[0], new_lambda[0], "Bregma and lambda z-coordinates should be equal after alignment")

    def test_align_bl_x_alignment(self):
        new_bregma = load_object(TEMP_FOLDER + '05_align_bl_bregma.pkl')
        new_lambda = load_object(TEMP_FOLDER + '05_align_bl_lambda.pkl')

        mid_x = (new_bregma[2] + new_lambda[2]) / 2
        self.assertAlmostEqual(new_bregma[2], mid_x, delta=1, msg="Bregma x-coordinate should be approximately equal to midpoint after alignment")
        self.assertAlmostEqual(new_lambda[2], mid_x, delta=1, msg="Lambda x-coordinate should be approximately equal to midpoint after alignment")


if __name__ == "__main_":
    unittest.main()
