from scripts import ml_functions as ml
import tensorflow as tf
import os
import numpy as np
import scipy.misc
import random

print("Path: " + os.path.abspath("./"))

# Don't forget to change this path when using new datasets
x_total, y_total = ml.parse_full_data_greyscale(
    os.path.abspath("resources/training_mat_data/greyscale_with_artificial_200x200.mat"), 200, 200, 1)

random.seed(69)
c = list(zip(x_total, y_total))
random.shuffle(c)
x_total, y_total = zip(*c)

x_train = x_total[0:int(0.8 * len(x_total))]
y_train = y_total[0:int(0.8 * len(y_total))]

x_test = x_total[int(0.8 * len(x_total)):]
y_test = y_total[int(0.8 * len(y_total)):]

image_dir = os.fsencode(os.path.abspath("out/greyscale_with_artificial_200x200_labeled"))
# Change save_dir to point to a different model directory to load different models
save_dir = os.path.abspath("models/model-0")

with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(save_dir + "/model-10.meta")  # <-- As of now this line needs to be manually
    # changed to load the meta graph for each trained model.
    saver.restore(sess, tf.train.latest_checkpoint(save_dir))
    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name(name="X:0")
    Y = graph.get_tensor_by_name(name="Y:0")
    Z = graph.get_tensor_by_name(name="Z:0")
    cost = graph.get_tensor_by_name(name="cost:0")
    optimizer = graph.get_operation_by_name(name="optimizer")
    accuracy = graph.get_tensor_by_name(name="accuracy:0")

    print("Test accuracy: " + str(sess.run(accuracy, feed_dict={X: x_test, Y: y_test})))
    print("Train accuracy: " + str(sess.run(accuracy, feed_dict={X: x_train, Y: y_train})))

    i = 0
    for image, label in zip(x_test, y_test):
        Z_arr = sess.run(Z, feed_dict={X: image.reshape((1, 200, 200, 1))})
        print(Z_arr)

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

        actual_str = ml.prediction_num_to_string(np.argmax(label))
        print("Predicted: " + prediction_str)
        print("Actual: " + actual_str)

        if prediction_str == actual_str:
            print("OOPS<-----------------------------------------------------------------------------------------")

            scipy.misc.toimage(np.reshape(image, (200, 200)), cmin=0, cmax=255) \
                .save(os.path.abspath(image_dir + "/[" + str(
                    i) + "] predicted_" + prediction_str + ", actual_" + actual_str + ".jpg"))

        print('\n')
        i += 1
