from neuroframe import *

import pandas as pd

mouse = Mouse.from_folder('P874', 'tests/integration/fixtures/test_experiment/test_mouse_p874')
segment_dataframe = pd.read_csv('tests/integration/fixtures/test_segmentation_info.csv')

labels = layer_colapsing(mouse, segment_dataframe)
print(labels)

'''
template_vol = adapt_template(mouse, ALLEN_TEMPLATE)
align_to_allen(mouse)

plot_mouse_template_overlay(template_vol, mouse.segmentation.volume)

skull = extract_skull(mouse)

#plot_skull(skull)

bregma, lambda_ = get_bregma_lambda(mouse, skull)

#plot_bl(skull, bregma, lambda_)

new_bregma, new_lambda = align_to_bl(mouse, bregma, lambda_, deviation=40)

#plot_alignment(mouse)
'''



#dataframe_coords = nf.stereotaxic_coordinates(mice_p324, reference_df, (bregma_coords, lambda_coords), is_parallelized=True, verbose=2, mode='full_inner')