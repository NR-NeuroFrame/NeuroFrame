# ================================================================
# 0. Section: Imports
# ================================================================
import os



# ================================================================
# 1. Section: Folders
# ================================================================
def get_folders(folder_path: str) -> list:
    folders = [
        f for f in os.listdir(folder_path)
        if f != ".DS_Store" and os.path.isdir(os.path.join(folder_path, f))
    ]
    print(folders)

    return folders
