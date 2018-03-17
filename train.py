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

print("\nTotal data:")
ml.print_num_of_each_class(y_total)

print("Train data:")
ml.print_num_of_each_class(y_train)

print("Test data:")
ml.print_num_of_each_class(y_test)
# ---

# Hyperparameters ---
learning_rate = 0.003
regularization_coefficient = 0.12
num_epochs = 3  # num_epochs should be a multiple of save_period to
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
train_scores = []
test_scores = []
save_period = 1
evaluate_model_period = 1
save_points = []
check_model_points = []
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
        minibatch_train_accuracies = []
        minibatch_test_accuracies = []

        # Gets train minibatches, runs the optimizer, and saves accuracy of minibatch
        for x_train_minibatch, y_train_minibatch in ml.get_minibatches(x_train, y_train, minibatch_size,
                                                                       shuffle=True, drop_extra_examples=True):
            sess.run(optimizer, feed_dict={X: x_train_minibatch, Y: y_train_minibatch})

            if epoch % evaluate_model_period == 0:
                minibatch_train_accuracies.append(sess.run(accuracy,
                                                           feed_dict={X: x_train_minibatch, Y: y_train_minibatch}))

        if epoch % save_period == 0 or epoch == num_epochs:
            print("Saving:")
            saver.save(sess, save_dir, global_step=epoch)
            save_points.append(epoch)
            if len(save_points) > num_saves_to_keep:
                save_points.pop(0)

        if epoch % evaluate_model_period == 0:
            for x_test_minibatch, y_test_minibatch in ml.get_minibatches(x_test, y_test, minibatch_size,
                                                                         shuffle=True, drop_extra_examples=True):
                minibatch_test_accuracies.append(sess.run(accuracy,
                                                          feed_dict={X: x_test_minibatch, Y: y_test_minibatch}))

            epoch_train_accuracy = np.sum(minibatch_train_accuracies) / len(minibatch_train_accuracies)
            train_scores.append(epoch_train_accuracy)

            epoch_test_accuracy = np.sum(minibatch_test_accuracies) / len(minibatch_test_accuracies)
            test_scores.append(epoch_test_accuracy)

            check_model_points.append(epoch)

            print("Epoch " + str(epoch) + ", current train accuracy: " + str(epoch_train_accuracy) +
                  ", current test accuracy: " + str(epoch_test_accuracy))
        else:
            print("Epoch " + str(epoch))

    plt.plot(check_model_points, np.squeeze(train_scores), label="train acc", color="#ff0000")
    plt.plot(check_model_points, np.squeeze(test_scores), label="test acc", color='#0000ff')
    for c, i in enumerate(save_points):
        if c == 0:
            plt.axvline(x=i, ymin=0, ymax=0.1, label="saved", color="#000000")
        else:
            plt.axvline(x=i, ymin=0, ymax=0.1, color="#000000")
    plt.legend()

    plt.xticks(check_model_points)
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title("Epochs: " + str(num_epochs)
              + "\nLearning rate: " + str(learning_rate)
              + "\nRegularization: " + str(regularization_coefficient))
    plt.show()
