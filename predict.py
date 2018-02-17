import tensorflow as tf
import os
import numpy as np
import scipy.misc

image_directory = os.fsencode("out\\greyscale_200x200_labeled")

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('.\\models\\test_save-0.meta')
    saver.restore(sess, tf.train.latest_checkpoint('.\\models'))
    graph = tf.get_default_graph()

    Z = graph.get_tensor_by_name(name="Z:0")
    X = graph.get_tensor_by_name(name="X:0")
    i = 0
    for file in os.listdir(image_directory):
        filename = os.fsdecode(file)
        image = scipy.misc.imread("out\\greyscale_200x200_labeled\\" + filename)

        Z_arr = sess.run(Z, feed_dict={X: image.reshape((1, 200, 200, 1))})
        print(Z_arr)
        print(filename)
        prediction = np.argmax(Z_arr)
        prediction_str = ""
        if prediction == 0:  # skeletal
            prediction_str = "skeletal"
        elif prediction == 1:  # respiratory
            prediction_str = "respiratory"
        elif prediction == 2:  # neural
            prediction_str = "neural"
        elif prediction == 3:  # muscular
            prediction_str = "muscular"
        print(prediction_str)

        if prediction_str not in filename:
            print("OOPS<-----------------------------------------------------------------------------------------\n")
            scipy.misc.toimage(np.reshape(image, (200, 200)), cmin=0, cmax=255) \
                .save("out\\misclassified_greyscale_200x200\\[" + str(i) + "] predicted_" + prediction_str + ", actual_" + filename + ".jpg")
        else:
            print('\n')

        i += 1
