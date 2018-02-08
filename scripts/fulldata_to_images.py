from scripts import ml_functions as ml
import scipy.misc
import numpy as np

# This script takes the data from greyscale_data.mat and saves the labeled images.

x, y = ml.parse_full_data("C:\\Users\\Sage\\PycharmProjects\\Denise-ML\\resources\\greyscale_data.mat")
save_path = "C:\\Users\\Sage\\PycharmProjects\\Denise-ML\\out\\labeled_images_from_greyscale_training_data"

i = 0
for image, val in zip(x, y):
    type_string = ""
    if np.argmax(val) == 0:  # skeletal
        type_string = "skeletal"
    elif np.argmax(val) == 1:  # respiratory
        type_string = "respiratory"
    elif np.argmax(val) == 2:  # neural
        type_string = "neural"
    elif np.argmax(val) == 3:  # muscular
        type_string = "muscular"

    scipy.misc.toimage(np.reshape(image, (100, 100)), cmin=0, cmax=255)\
        .save(save_path + "\\[" + str(i) + "]" + " type_" + type_string + ".jpg")
    i += 1
