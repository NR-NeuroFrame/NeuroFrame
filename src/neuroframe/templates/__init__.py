import cv2

from pathlib import Path

from ..mouse_data import Segmentation
from ..registrator import convert_input

TEMPLATES_DIR = Path(__file__).resolve().parent

ALLEN_TEMPLATE: Segmentation = Segmentation(str(TEMPLATES_DIR / "allen_brain_25μm_ccf_2017.nii.gz"))
SUTURE_TEMPLATE = cv2.imread(str(TEMPLATES_DIR / "suture_template_t14.png"), cv2.IMREAD_GRAYSCALE)
BREGMA_TEMPLATE = convert_input(cv2.imread(str(TEMPLATES_DIR / "bregma_template_t14.png"), cv2.IMREAD_GRAYSCALE))
LAMBDA_TEMPLATE = convert_input(cv2.imread(str(TEMPLATES_DIR / "lambda_template_t14.png"), cv2.IMREAD_GRAYSCALE))
REF_TEMPLATES: tuple = (BREGMA_TEMPLATE, LAMBDA_TEMPLATE)
