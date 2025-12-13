# ================================================================
# 0. Section: Imports
# ================================================================
from ._assertions import *



class Properties:
    # ================================================================
    # 1. Section: Properties
    # ================================================================
    @property
    def folder(self) -> str:
        # Warns if the folder paths do not match (designed to do every call)
        assert_folder_consitency(self.paths)

        # Return the mice ID folder path
        folder = self.mri.folder
        return folder
    
    @property
    def data_shape(self) -> tuple[int, int, int]: 
        # Warns if the shapes do not match (designed to do every call)
        assert_shape_consitency([self.micro_ct.shape, self.mri.shape, self.segmentation.shape])

        return self.mri.shape
    
    @property
    def voxel_size(self) -> tuple[float, float, float]:
        # Warns if the voxel sizes do not match (designed to do every call)
        assert_voxel_size_consitency([self.micro_ct.voxel_size, self.mri.voxel_size, self.segmentation.voxel_size])

        return self.mri.voxel_size

    @property
    def id(self) -> str: return self._id

    @property
    def paths(self) -> dict[str, str]: return self._paths



    # ================================================================
    # 2. Section: Setters
    # ================================================================
    @id.setter
    def id(self, value: str) -> None:
        # Warns if the folder path does not contain the mice ID
        assert_id_folder_consitency(self.folder, value)

        self._id = value

    @paths.setter
    def paths(self, value: dict[str, str]) -> None:
        # Warns if the folder paths do not match
        assert_folder_consitency(value)

        self._paths = value