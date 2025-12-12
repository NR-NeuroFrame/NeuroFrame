# ================================================================
# 0. Section: Imports
# ================================================================
import os

from ..mouse_data import MicroCT, MRI, Segmentation
from ._dunders import Dunders
from ._properties import Properties
from ._plots import Plots
from ._assertions import assert_required_files, assert_no_extra_files


# ================================================================
# 1. Section: Mouse Classes
# ================================================================
class Mouse(Dunders, Properties, Plots):
    def __init__(self, id: str, mri_path: str, ct_path: str, segmentations_path: str) -> None:
        self.micro_ct = MicroCT(ct_path)
        self.mri = MRI(mri_path)
        self.segmentations = Segmentation(segmentations_path)

        self.paths = {
            'ct_path': ct_path,
            'mri_path': mri_path,
            'segmentations_path': segmentations_path
        }

        self.id = id

    @classmethod
    def from_folder(cls, id: str, folder_path: str) -> 'Mouse':
        # Makes sure is safe to proceed
        assert_required_files(folder_path)
        assert_no_extra_files(folder_path)

        files = os.listdir(folder_path)
        target_files = ['_mri', '_uCT', '_seg']

        for target in target_files:
            target_file = [file for file in files if target in file][0]

            file_path = os.path.join(folder_path, target_file)

            if target == '_mri': mri_path = file_path
            elif target == '_uCT': ct_path = file_path
            elif target == '_seg': segmentations_path = file_path

        return cls(id, mri_path, ct_path, segmentations_path)