import cv2

from ..mouse_data import Segmentation
from ..registrator import convert_input



ALLEN_TEMPLATE: Segmentation = Segmentation("src/neuroframe/templates/allen_brain_25μm_ccf_2017.nii.gz")
SUTURE_TEMPLATE = cv2.imread("src/neuroframe/templates/suture_template_t14.png", cv2.IMREAD_GRAYSCALE)
BREGMA_TEMPLATE = convert_input(cv2.imread("src/neuroframe/templates/bregma_template_t14.png", cv2.IMREAD_GRAYSCALE))
LAMBDA_TEMPLATE = convert_input(cv2.imread("src/neuroframe/templates/lambda_template_t14.png", cv2.IMREAD_GRAYSCALE))
REF_TEMPLATES: tuple = (BREGMA_TEMPLATE, LAMBDA_TEMPLATE)
