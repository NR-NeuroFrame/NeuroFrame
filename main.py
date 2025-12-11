from neuroframe.src.neuroframe import *
import time
import numpy as np


start_time = time.time()
test = MicroCT("p324.nii.gz")
print(test)
print("Loaded in --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print(np.max(test.data), np.min(test.data), sep=", ")
print("Loaded in --- %s seconds ---" % (time.time() - start_time))