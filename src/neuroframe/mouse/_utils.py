# ================================================================
# 0. Section: IMPORTS
# ================================================================
from ..mouse_data import Hemisphere, SegmentationEDT, SegmentationNEDT, FieldBL
from ..logger import logger



# ================================================================
# 1. Section: Helper Functions for Channels
# ================================================================
def get_path_key(cls) -> str:
    if(cls is Hemisphere): return "hemisphere_path"
    elif(cls is SegmentationEDT): return "segmentation_edt_path"
    elif(cls is SegmentationNEDT): return "segmentation_nedt_path"
    elif(cls is FieldBL): return "field_bl_path"
    else:
        logger.warning(f"Something is not right, you added a class ({cls}) that is not defined here")
        raise TypeError(f"{cls} is not defined")

def get_attribute(cls) -> str:
    if(cls is Hemisphere): return "hemisphere"
    elif(cls is SegmentationEDT): return "segmentation_edt"
    elif(cls is SegmentationNEDT): return "segmentation_nedt"
    elif(cls is FieldBL): return "field_bl"
    else:
        logger.warning(f"Something is not right, you added a class ({cls}) that is not defined here")
        raise TypeError(f"{cls} is not defined")



# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
