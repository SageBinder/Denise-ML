from scripts import ml_functions as ml
import tensorflow as tf
import os
import numpy as np
import scipy.misc
import random

print("Path: " + os.path.abspath("./"))

x_total, y_total = ml.parse_full_data_greyscale(os.path.abspath("resources/training_mat_data/greyscale_200x200.mat"))

random.seed(69)
c = list(zip(x_total, y_total))
random.shuffle(c)
x_total, y_total = zip(*c)

x_train = x_total[0:350]
y_train = y_total[0:350]

x_test = x_total[350:]
y_test = y_total[350:]

image_dir = os.fsencode(os.path.abspath("out/greyscale_200x200_labeled"))
save_dir = os.path.abspath("models/model-0")

with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(save_dir + "/test_save-99.meta")
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
    for file in os.listdir(image_dir):
        filename = os.fsdecode(file)
        image = scipy.misc.imread(os.path.abspath("out/greyscale_200x200_labeled/" + filename))

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
                .save(os.path.abspath("out/misclassified_greyscale_200x200/[" + str(
                    i) + "] predicted_" + prediction_str + ", actual_" + filename + ".jpg"))
        else:
            print('\n')

        i += 1
