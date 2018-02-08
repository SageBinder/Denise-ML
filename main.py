from scripts import ml_functions as ml
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
#SAGGI IS F I L T H Y
print(tf.__version__)

learning_rate = 0.004
regularization_coefficient = 0.009
num_epochs = 300
kernel_shapes = {"L1": [9, 9, 1, 15],
                 "L2": [7, 7, 15, 17],
                 "L3": [5, 5, 17, 19],
                 "L4": [3, 3, 19, 21]}

max_pool_shapes = {"L1": ([1, 5, 5, 1], "SAME"),
                   "L2": ([1, 5, 5, 1], "SAME"),
                   "L3": ([1, 1, 1, 1], "SAME"),
                   "L4": ([1, 2, 2, 1], "SAME")}

tf.logging.set_verbosity(tf.logging.INFO)

x_total, y_total = ml.parse_full_data('C:\\Users\\special023\\PycharmProjects\\Denise-ML\\resources\\training_mat_data\\training_mat_data\\greyscale_data.mat')
(m, n_H, n_W, n_C) = x_total.shape

c = list(zip(x_total, y_total))
random.shuffle(c)
x_total, y_total = zip(*c)

x_train = x_total[0:350]
y_train = y_total[0:350]

x_test = x_total[350:]
y_test = y_total[350:]

X, Y = ml.create_placeholders(n_H, n_W, n_C, 4)
parameters = ml.initialize_parameters(kernel_shapes, regularization_coefficient)
Z = ml.forward_propagation(X, parameters, max_pool_shapes)
cost = ml.get_cost(Z, Y)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

costs = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        _, temp_cost = sess.run([optimizer, cost], feed_dict={X: x_train, Y: y_train})

        costs.append(temp_cost)

        print("Epoch " + str(epoch) + ", cost is: " + str(temp_cost))

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    predict_op = tf.argmax(Z, axis=1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, axis=1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy)
    train_accuracy = accuracy.eval({X: x_train, Y: y_train})
    print("Train Accuracy:", train_accuracy)
    test_accuracy = accuracy.eval({X: x_test, Y: y_test})
    print("Test Accuracy:", test_accuracy)

