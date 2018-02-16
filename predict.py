import tensorflow as tf
import scipy.misc

image = scipy.misc.imread("out\\labeled_images_from_greyscale_200x200_data\\[0] type_muscular.jpg")
image = image.reshape((1, 200, 200, 1))

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('.\\models\\test_save-0.meta')
    saver.restore(sess, tf.train.latest_checkpoint('.\\models'))
    graph = tf.get_default_graph()

    Z = graph.get_tensor_by_name(name="Z:0")
    X = graph.get_tensor_by_name(name="X:0")
    print(sess.run(Z, feed_dict={X: image}))
    prediction = tf.argmax(Z)

    if prediction == 0:  # skeletal
        print("skeletal")
    elif prediction == 1:  # respiratory
        print("respiratory")
    elif prediction == 2:  # neural
        print("neural")
    elif prediction == 3:  # muscular
        print("muscular")
