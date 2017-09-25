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
    label = tf.argmin(s4, axis=1)
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
        train, Loss, n_cluster = sess.run([train_step, loss, label], feed_dict={xs: data})
        loss_itrs.append(Loss)
    return loss_itrs, n_cluster


if __name__ == '__main__':
    learning_rate = 0.07
    k = 5
    iteration = 300
    loss, cluster= run(learning_rate, k, iteration)
    a = np.zeros((k))
    for num in(cluster):
        a[num] += 1
    a = a/len(cluster)
    print(a)
    print(loss[-1])
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    x = data[:, 0]
    y = data[:, 1]
    points = cluster
    # color is the length of each vector in `points`
    color = np.sqrt((points ** 2) / np.sqrt(2.0))
    rgb = plt.get_cmap('jet')(color)
    ax.scatter(x, y, color=rgb)
    ax1 = fig.add_subplot(1,2,2)
    ax1.plot(loss)
    ax1.set_title("learning rate:{}, k:{}, epoch:{}".format(learning_rate, k, iteration))
    ax1.set_ylabel("loss")
    ax1.set_xlabel("iteration number")
    plt.show()

