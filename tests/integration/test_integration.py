# ================================================================
# 0. Section: Imports
# ================================================================
import unittest
import os
import shutil

from neuroframe.utils.save_utils import TEMP_FOLDER



# ================================================================
# 1. Section: Test Cases
# ================================================================
class Test00Start(unittest.TestCase):
    def test_start(self):
        # Deletes previous test output if exists
        if os.path.exists(TEMP_FOLDER):
            for filename in os.listdir(TEMP_FOLDER):
                file_path = os.path.join(TEMP_FOLDER, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        self.assertTrue(os.path.exists(TEMP_FOLDER), "Temporary folder should exist for test outputs")
        self.assertEqual(len(os.listdir(TEMP_FOLDER)), 0, "Temporary folder should be empty at the start of tests")

class Test99End(unittest.TestCase):
    def test_end(self):
        # Cleans up temporary test output
        if os.path.exists(TEMP_FOLDER):
            for filename in os.listdir(TEMP_FOLDER):
                file_path = os.path.join(TEMP_FOLDER, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        self.assertTrue(os.path.exists(TEMP_FOLDER), "Temporary folder should exist for test outputs")
        self.assertEqual(len(os.listdir(TEMP_FOLDER)), 0, "Temporary folder should be empty at the end of tests")
