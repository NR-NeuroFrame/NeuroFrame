# ================================================================
# 0. Section: Imports
# ================================================================
import unittest
import numpy as np

from neuroframe.pipeline import extract_skull, get_bregma_lambda
from neuroframe.pipeline.bregma_lambda.extract_bl import extract_deformation_map
from neuroframe.registrator import convert_input
from neuroframe.templates import REF_TEMPLATES
from neuroframe.pipeline.bregma_lambda.extract_bl import (
    get_reference_point,
    apply_shape,
    compute_deviation
)
from neuroframe.utils import (
    save_object,
    load_object,
    TEMP_FOLDER,
    get_z_coord,
)



# ================================================================
# 1. Section: Test Cases
# ================================================================
class Test03ExtractBL(unittest.TestCase):
    def test_00_extract_deformation_map_transformation_not_identity(self):
        mouse = load_object(TEMP_FOLDER + '01_align_mouse.pkl')
        skull = extract_skull(mouse)
        save_object(skull, TEMP_FOLDER + '02_extract_skull_skull.pkl')

        transform = extract_deformation_map(skull)
        save_object(transform, TEMP_FOLDER + '03_extract_bl_transform.pkl')

        # Assert that the transform parameters are not all zeros (identity)
        self.assertFalse(all(param == 0 for param in transform.GetParameters()), "Deformation map transformation should not be identity")

    def test_get_referece_points_are_within_bounds(self):
        mouse = load_object(TEMP_FOLDER + '01_align_mouse.pkl')
        skull = load_object(TEMP_FOLDER + '02_extract_skull_skull.pkl')
        transform = load_object(TEMP_FOLDER + '03_extract_bl_transform.pkl')

        bregma_template, lambda_template = REF_TEMPLATES
        reference_slide = convert_input(mouse.segmentation.volume[100,:,:])
        bregma_template = apply_shape(reference_slide, bregma_template)
        lambda_template = apply_shape(reference_slide, lambda_template)

        bregma_coords = np.round(get_reference_point(bregma_template, skull, transform)).astype(int)
        lambda_coords = np.round(get_reference_point(lambda_template, skull, transform)).astype(int)

        # Get the z coordinates
        bregma_z = get_z_coord(mouse.micro_ct.data, bregma_coords)
        lambda_z = get_z_coord(mouse.micro_ct.data, lambda_coords)

        # Get the coordinates (z, y, x)
        bregma_coords = (bregma_z, bregma_coords[0], bregma_coords[1])
        lambda_coords = (lambda_z, lambda_coords[0], lambda_coords[1])

        # Assert that the coordinates are within the bounds of the skull surface
        for coords in [bregma_coords, lambda_coords]:
            self.assertTrue(all(0 <= coords[i] < mouse.data_shape[i] for i in range(3)), "Reference point coordinates should be within skull surface bounds")

    def test_compute_deviation_is_within_reason(self):
        mouse = load_object(TEMP_FOLDER + '01_align_mouse.pkl')
        skull = load_object(TEMP_FOLDER + '02_extract_skull_skull.pkl')
        transform = load_object(TEMP_FOLDER + '03_extract_bl_transform.pkl')

        bregma_template, lambda_template = REF_TEMPLATES
        reference_slide = convert_input(mouse.segmentation.volume[100,:,:])
        bregma_template = apply_shape(reference_slide, bregma_template)
        lambda_template = apply_shape(reference_slide, lambda_template)

        bregma_coords = np.round(get_reference_point(bregma_template, skull, transform)).astype(int)
        lambda_coords = np.round(get_reference_point(lambda_template, skull, transform)).astype(int)

        # Get the z coordinates
        bregma_z = get_z_coord(mouse.micro_ct.data, bregma_coords)
        lambda_z = get_z_coord(mouse.micro_ct.data, lambda_coords)

        # Get the coordinates (z, y, x)
        bregma_coords = (bregma_z, bregma_coords[0], bregma_coords[1])
        lambda_coords = (lambda_z, lambda_coords[0], lambda_coords[1])

        deviations, angle = compute_deviation(mouse, (bregma_coords, lambda_coords))

        # Assert that deviations are within a reasonable range (e.g., less than 5 mm)
        self.assertTrue(all(dev < 2.0 for dev in deviations), "Deviations should be less than 2 mm")

        # Assert that angle is within a reasonable range (e.g., less than 30 degrees)
        self.assertTrue(angle < 10.0, "Angle should be less than 10 degrees")

    def test_get_bregma_lambda_returns_coordinates(self):
        mouse = load_object(TEMP_FOLDER + '01_align_mouse.pkl')
        skull = load_object(TEMP_FOLDER + '02_extract_skull_skull.pkl')

        bregma, lambda_ = get_bregma_lambda(mouse, skull)
        save_object(bregma, TEMP_FOLDER + '04_extract_bl_bregma.pkl')
        save_object(lambda_, TEMP_FOLDER + '04_extract_bl_lambda.pkl')

        # Assert that the returned coordinates are tuples of length 3
        self.assertIsInstance(bregma, tuple, "Bregma should be a tuple")
        self.assertIsInstance(lambda_, tuple, "Lambda should be a tuple")
        self.assertEqual(len(bregma), 3, "Bregma should have 3 coordinates (z, y, x)")
        self.assertEqual(len(lambda_), 3, "Lambda should have 3 coordinates (z, y, x)")
