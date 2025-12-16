from neuroframe import *
import time

mouse = Mouse.from_folder('P874', 'tests/integration/fixtures/test_experiment/test_mouse_p874')

template_vol = adapt_template(mouse, ALLEN_TEMPLATE)
align_to_allen(mouse)

#plot_mouse_template_overlay(template_vol, mouse.segmentation.volume)

skull = extract_skull(mouse)

#plot_skull(skull)

bregma, lambda_ = get_bregma_lambda(mouse, skull)

#plot_bl(skull, bregma, lambda_)

new_bregma, new_lambda = align_to_bl(mouse, bregma, lambda_, deviation=40)

plot_alignment(mouse)