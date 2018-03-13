from scripts import ml_functions as ml
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import os

print(tf.__version__)

export_dir = './models/test_save'

if os.path.exists(os.path.abspath(export_dir)):
    shutil.rmtree(os.path.abspath(export_dir))

tf.set_random_seed(69)
learning_rate = 0.003
regularization_coefficient = 0.12
num_epochs = 1
kernel_shapes = {"L1": [15, 15, 1, 19],
                 "L2": [9, 9, 19, 21],
                 "L3": [5, 5, 21, 23]}

max_pool_shapes = {"L1": ([1, 5, 5, 1], "SAME"),
                   "L2": ([1, 5, 5, 1], "SAME"),
                   "L3": ([1, 3, 3, 1], "SAME")}

x_total, y_total = ml.parse_full_data_greyscale('resources/training_mat_data/greyscale_200x200.mat')

(m, n_H, n_W, n_C) = x_total.shape

c = list(zip(x_total, y_total))
random.shuffle(c)
x_total, y_total = zip(*c)

x_train = x_total[0:350]
y_train = y_total[0:350]

x_test = x_total[350:]
y_test = y_total[350:]

train_accuracies = []
test_accuracies = []

X, Y = ml.create_placeholders(n_H, n_W, n_C, 4)
parameters = ml.initialize_parameters(kernel_shapes, regularization_coefficient)
Z = ml.forward_propagation(X, parameters, max_pool_shapes)
cost = ml.get_cost(Z, Y)

predict_op = tf.argmax(Z, axis=1)
correct_prediction = tf.equal(predict_op, tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, name="optimizer")
init = tf.global_variables_initializer()

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        sess.run(optimizer, feed_dict={X: x_train, Y: y_train})

        train_accuracy = accuracy.eval({X: x_train, Y: y_train})
        test_accuracy = accuracy.eval({X: x_test, Y: y_test})

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print("Epoch " + str(epoch) + ", current train acc: " + str(train_accuracy) +
              ", current test acc: " + str(test_accuracy))

    plt.plot(np.squeeze(train_accuracies), label="train acc", color="#ff0000")
    plt.plot(np.squeeze(test_accuracies), label="test acc", color='#0000ff')
    plt.legend()

    plt.xticks(range(num_epochs))
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # As of now, these values are already calculated in the last iteration of the for loop
    # train_accuracy = accuracy.eval({X: x_train, Y: y_train})
    # print("Train Accuracy:", train_accuracy)
    # test_accuracy = accuracy.eval({X: x_test, Y: y_test})
    # print("Test Accuracy:", test_accuracy)

    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.TRAINING],
                                         signature_def_map={
                                             "model": tf.saved_model.signature_def_utils.predict_signature_def(
                                                 inputs={"X": X},
                                                 outputs={"Z": Z}
                                             )})
builder.save()
