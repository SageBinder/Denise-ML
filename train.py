from scripts import ml_functions as ml
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

print(tf.__version__)

# Load dataset ---
x_total, y_total = ml.parse_full_data_greyscale('resources/training_mat_data/greyscale_with_artificial_200x200.mat'
                                                , 200, 200, 1)
(m, n_H, n_W, n_C) = x_total.shape
# ---

# Shuffle dataset and get train/test subsets ---
random.seed(69)
c = list(zip(x_total, y_total))
random.shuffle(c)
x_total, y_total = zip(*c)

x_train = x_total[0:int(0.8 * len(x_total))]
y_train = y_total[0:int(0.8 * len(y_total))]
# Train/test split is 80/20
x_test = x_total[int(0.8 * len(x_total)):]
y_test = y_total[int(0.8 * len(y_total)):]
# ---

# Hyperparameters ---
learning_rate = 0.003
regularization_coefficient = 0.12
num_epochs = 2  # num_epochs should be a multiple of save_period to
# ensure the distance between save points remains constant.
minibatch_size = 100

kernel_shapes = {"L1": [15, 15, 1, 19],
                 "L2": [9, 9, 19, 21],
                 "L3": [5, 5, 21, 23]}

max_pool_shapes = {"L1": ([1, 5, 5, 1], "SAME"),
                   "L2": ([1, 5, 5, 1], "SAME"),
                   "L3": ([1, 3, 3, 1], "SAME")}
# ---

# Variables for keeping track of training ---
train_accuracies = []
test_accuracies = []
save_period = 1
check_acc_period = 1
save_points = []
check_acc_points = []
num_saves_to_keep = 3
# ---

# Build the graph ---
X, Y = ml.create_placeholders(n_H, n_W, n_C, 4)
parameters = ml.initialize_parameters(kernel_shapes, regularization_coefficient)
Z = ml.forward_propagation(X, parameters, max_pool_shapes)
cost = ml.get_cost(Z, Y)

predict_op = tf.argmax(Z, axis=1)
correct_prediction = tf.equal(predict_op, tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, name="optimizer")
init = tf.global_variables_initializer()
# ---

# Change save_dir to ./models/{model_name}/model
save_dir = './models/model-0/model'
saver = tf.train.Saver(max_to_keep=num_saves_to_keep)

with tf.Session() as sess:
    tf.set_random_seed(69)
    sess.run(init)

    for epoch in range(1, num_epochs + 1):  # epoch starts from 1 cause that makes life easier
        epoch_train_accuracy = 0
        epoch_test_accuracy = 0
        epoch_train_accuracies = []

        for x_train_minibatch, y_train_minibatch in ml.get_minibatches(x_train, y_train, minibatch_size):
            sess.run(optimizer, feed_dict={X: x_train_minibatch, Y: y_train_minibatch})

            if epoch % check_acc_period == 0:
                epoch_train_accuracies.append(sess.run(accuracy, feed_dict={X: x_train_minibatch, Y: y_train_minibatch}))

        if epoch % save_period == 0 or epoch == num_epochs:
            print("Saving:")
            saver.save(sess, save_dir, global_step=epoch)
            save_points.append(epoch)
            if len(save_points) > num_saves_to_keep:
                save_points.pop(0)

        if epoch % check_acc_period == 0:
            train_accuracy = sess.run(accuracy, feed_dict={X: x_train, Y: y_train})
            test_accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})

            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            check_acc_points.append(epoch)

            print("Epoch " + str(epoch) + ", current train acc: " + str(train_accuracy) +
                  ", current test acc: " + str(test_accuracy))
        else:
            print("Epoch " + str(epoch))

    plt.plot(check_acc_points, np.squeeze(train_accuracies), label="train acc", color="#ff0000")
    plt.plot(check_acc_points, np.squeeze(test_accuracies), label="test acc", color='#0000ff')
    for c, i in enumerate(save_points):
        if c == 0:
            plt.axvline(x=i, y_min=0, y_max=0.1, label="saved", color="#000000")
        else:
            plt.axvline(x=i, y_min=0, y_max=0.1, color="#000000")
    plt.legend()

    plt.xticks(check_acc_points)
    plt.ylabel('accuracy')
    plt.xlabel('iterations')
    plt.title("Epochs: " + str(num_epochs)
              + "\nLearning rate: " + str(learning_rate)
              + "\nRegularization: " + str(regularization_coefficient))
    plt.show()
