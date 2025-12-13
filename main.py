from neuroframe.src.neuroframe import *
import time


start_time = time.time()
mouse = Mouse.from_folder('P874', 'tests/integration/fixtures/test_experiment/test_mouse_p874')
print("Loaded in --- %s seconds ---" % (time.time() - start_time), end="\n\n")

start_time = time.time()
template_vol = adapt_template(mouse, ALLEN_TEMPLATE)
align_to_allen(mouse)
print("Adapted in --- %s seconds ---" % (time.time() - start_time), end="\n\n")

plot_mouse_template_overlay(template_vol, mouse.segmentation.volume)