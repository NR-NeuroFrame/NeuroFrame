# ================================================================
# 0. Section: Imports
# ================================================================
import unittest
from src.neuroframe.pipeline.extract_bl import *
from src.neuroframe.pipeline.extract_skull import *
from src.neuroframe.mouse import Mouse





# ================================================================
# 1. Section: Test Cases
# ================================================================
class TestExtractBL(unittest.TestCase):
    def test_extract_deformation_map_transformation_not_identity(self):
        mouse = Mouse.from_folder('P874', 'tests/integration/fixtures/test_experiment/test_mouse_p874')
        skull = extract_skull(mouse)

        transform = extract_deformation_map(skull).GetParameters()

        # Assert that the transform parameters are not all zeros (identity)
        self.assertFalse(all(param == 0 for param in transform), "Deformation map transformation should not be identity")

    def test_get_referece_points_are_within_bounds(self):
        mouse = Mouse.from_folder('P874', 'tests/integration/fixtures/test_experiment/test_mouse_p874')
        skull = extract_skull(mouse)
        transform = extract_deformation_map(skull)

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
        mouse = Mouse.from_folder('P874', 'tests/integration/fixtures/test_experiment/test_mouse_p874')
        skull = extract_skull(mouse)
        transform = extract_deformation_map(skull)

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



        