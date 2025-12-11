# ================================================================
# 0. Section: Imports
# ================================================================
import os

import numpy as np



class Properties:
    # ================================================================
    # 1. Section: Properties
    # ================================================================
    @property
    def path(self): return self._path

    @property
    def voxel_size(self): return self.nib.header.get_zooms()

    @property
    def affine(self): return self.nib.affine

    @property
    def filename(self): return os.path.basename(self.path)

    @property
    def shape(self): return self.data.shape

    @property
    def folder(self):
        folder_name = os.path.dirname(self.path)
        if folder_name == '': folder_name = '.'
        
        return folder_name



    # ================================================================
    # 2. Section: Setters
    # ================================================================
    @path.setter
    def path(self, value: str):
        if not value.endswith(('.nii', '.nii.gz')): raise ValueError("Image path must be a .nii or .nii.gz file.")
        if value.endswith('.nii'): print("Warning: Using .nii files is discouraged due to larger file sizes compared to .nii.gz.")

        self._path = value

    @voxel_size.setter
    def voxel_size(self, value: list | tuple | np.ndarray):
        if len(value) != 3: raise ValueError("Voxel size must be a composed of three numerical values.")
        if not np.all(np.isclose(value, value[0])): print("Warning: Voxel sizes are not isotropic (i.e., not the same in all dimensions).")

        self._voxel_size = value

    @affine.setter
    def affine(self, value: np.ndarray):
        if value.shape != (4, 4): raise ValueError("Affine must be a 4x4 numpy array.")

        self._affine = value