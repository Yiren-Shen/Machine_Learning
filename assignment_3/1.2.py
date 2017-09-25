import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = np.load("data2D.npy")
#trainingData = data[0:6666]
#validationData = data[6666:10000]

def index_update(training_set, centroids_matrix):
    s1 = tf.matmul(training_set, centroids_matrix, transpose_b=True)
    s2 = tf.reshape(tf.reduce_sum(tf.square(training_set), 1), [-1, 1])
    s3 = tf.reduce_sum(tf.square(centroids_matrix), 1)
    s4 = tf.sub(tf.add(s2, s3), 2 * s1)
    return s4


def buildGraph(learning_rate, k):
    xs = tf.placeholder(tf.float32, [None, 2])
    centroids = tf.Variable(tf.random_normal(shape=[k, 2], stddev=1.0, dtype=tf.float32))
    s4= index_update(xs, centroids)
    l = tf.reduce_min(s4, axis=1)
    label = tf.argmin(l, axis=1)
    loss = tf.reduce_sum(l)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.99,epsilon=1e-5).minimize(loss=loss)
    return xs, centroids, label, loss, train_step

def run(learning_rate, k, itrs):
    xs, centroids, label, loss, train_step = buildGraph(learning_rate, k)
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    loss_itrs =[]
    for i in range(0, itrs):
        train, Loss = sess.run([train_step, loss], feed_dict={xs: data})
        loss_itrs.append(Loss)
        print (Loss)
    return loss_itrs


if __name__ == '__main__':
    learning_rate = 0.07
    k = 3
    iteration = 300
    loss = run(learning_rate, k, iteration)
    plt.plot(loss)
    plt.title("learning rate:{}, k:{}, epoch:{}".format(learning_rate, k, iteration))
    plt.ylabel("loss")
    plt.xlabel("iteration number")
    plt.show()