# ================================================================
# 0. Section: Imports
# ================================================================
import os

from pathlib import Path
from typing import Type

from ..mouse_data import (
    MicroCT,
    MRI,
    Segmentation,
    Hemisphere,
    SegmentationEDT,
    SegmentationNEDT,
    FieldBL
)
from ._dunders import Dunders
from ._properties import Properties
from ._plots import Plots
from ._assertions import assert_required_files, assert_no_extra_files
from ._utils import get_attribute, get_path_key



# ================================================================
# 1. Section: Mouse Classes
# ================================================================
class Mouse(Dunders, Properties, Plots):
    def __init__(
        self,
        id: str,
        mri_path: str,
        ct_path: str,
        segmentations_path: str,
        hemisphere_path: str | None = None,
        segmentation_edt_path: str | None = None,
        segmentation_nedt_path: str | None = None,
        field_bl_path: str | None = None
    ) -> None:
        self.micro_ct = MicroCT(str(ct_path))
        self.mri = MRI(str(mri_path))
        self.segmentation = Segmentation(str(segmentations_path))

        self.paths = {
            'ct_path': str(ct_path),
            'mri_path': str(mri_path),
            'segmentations_path': str(segmentations_path),
        }

        # Only adds these if defined
        self.add_path(hemisphere_path, Hemisphere)
        self.add_path(segmentation_edt_path, SegmentationEDT)
        self.add_path(segmentation_nedt_path, SegmentationNEDT)
        self.add_path(field_bl_path, FieldBL)

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



    # ================================================================
    # 2. Section: Helper Class Functions
    # ================================================================
    def add_path(self, path: str | Path | None, cls: Type) -> None:
        attribute = get_attribute(cls)
        path_key = get_path_key(cls)

        self.paths[path_key] = str(path) if path is not None else None

        if(path is not None):
            setattr(self, attribute, cls(str(path)))
