import tensorflow as tf
import numpy as np


def create_placeholders(n_h, n_w, n_c, n_y):
    x = tf.placeholder(dtype=tf.float32, shape=(None, n_h, n_w, n_c), name="X")
    y = tf.placeholder(dtype=tf.float32, shape=(None, n_y), name="Y")
    return x, y


def initialize_parameters(shapes, regularization_coefficient):
    parameters = {}
    regularizer = tf.contrib.layers.l2_regularizer(regularization_coefficient)
    for layer, shape in shapes.items():
        parameters[layer] = tf.get_variable(layer, shape, initializer=tf.contrib.layers.xavier_initializer(),
                                            regularizer=regularizer)

    return parameters


def forward_propagation(x, parameters, max_pool_shapes):
    p = x

    for layer, w in parameters.items():
        z = tf.nn.conv2d(p, w, strides=[1, 1, 1, 1], padding='SAME')
        a = tf.nn.relu(z)
        p = tf.nn.max_pool(a,
                           ksize=max_pool_shapes[layer][0],
                           strides=max_pool_shapes[layer][0],
                           padding=max_pool_shapes[layer][1])

    p = tf.contrib.layers.flatten(p)

    z = tf.contrib.layers.fully_connected(p, 4, activation_fn=None)
    z = tf.identity(z, name="Z")
    return z


def get_cost(z, y_true):
    return tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y_true)),
                  sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), name="cost")


def parse_full_data_greyscale(file_path, height, width, channels):
    mat = np.loadtxt(file_path)
    x = []
    y = []
    for image in mat:
        x.append(np.reshape(image[0:-1], (height, width, channels)))
        y.append(int(image[-1]) - 1)
    y_one_hot = np.zeros((len(y), 4))
    y_one_hot[np.arange(len(y)), y] = 1

    return np.array(x), np.array(y_one_hot)


def prediction_num_to_string(y):
    if y == 0:  # skeletal
        return "skeletal"
    elif y == 1:  # respiratory
        return "respiratory"
    elif y == 2:  # neural
        return "neural"
    elif y == 3:  # muscular
        return "muscular"


def get_minibatches(x, y, batch_size):
    m = np.shape(x)[0]
    assert m == np.shape(y)[0]

    for i in range(0, m - batch_size + 1, batch_size):
        excerpt = slice(i, i + batch_size)
        yield x[excerpt], y[excerpt]
    yield x[m - (m % batch_size):], y[m - (m % batch_size):]
