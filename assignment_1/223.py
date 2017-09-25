import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def runMult(batch_size, lam, color):
    with np.load("tinymnist.npz") as data:
        trainData, trainTarget = data["x"], data["y"]
        validData, validTarget = data["x_valid"], data["y_valid"]
        testData, testTarget = data["x_test"], data["y_test"]

    n_train_batches = 700 / batch_size

    # Variable creation
    W = tf.Variable(tf.truncated_normal(
        shape=[64, 1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 64], name='input_x')
    y_target = tf.placeholder(tf.float32, [None, 1], name='target_y')

    # Graph definition
    y_predicted = tf.matmul(X, W) + b

    # Error definition
    cost = tf.add(tf.reduce_sum(tf.square(y_predicted - y_target)) /
                  (2 * batch_size), tf.mul(lam / 2., tf.reduce_sum(tf.square(W))))

    # Training mechanism
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=eta)
    train = optimizer.minimize(loss=cost)

    # Initialize session
    init = tf.initialize_all_variables()

    sess = tf.InteractiveSession()
    sess.run(init)

    initialW = sess.run(W)
    initialb = sess.run(b)

    # print("Initial weights: %s, initial bias: %.2f" % (initialW, initialb))
    # Training model

    epoch = 0

    err = []
    while (epoch <= 100):

        epoch = epoch + 1
        for minibatch_index in np.arange(n_train_batches):
            _, error, currentW, currentb, yhat = sess.run([train, cost, W, b, y_predicted], feed_dict={X: trainData[minibatch_index * batch_size: (
                minibatch_index + 1) * batch_size], y_target: trainTarget[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]})
        err.append(error)
    plt.figure(1)
    plt.plot(err, '%s-' % color, linewidth=2.0, label='$\lambda$ = %.4f' % lam)

    # Testing model
    errTest = sess.run(cost, feed_dict={X: testData, y_target: testTarget})
    print("Final testing penalized MSE with mini-batch size lambda = %.4f: %.2f" %
          (lam, errTest))

    validPredicted = np.matmul(validData, currentW) + currentb
    validPredicted = np.round(np.clip(validPredicted, 0, 1))
    error_rate = np.sum(np.abs(validPredicted - validTarget)
                        ) / np.size(validTarget)
    acc.append(1 - error_rate)

    testPredicted = np.matmul(testData, currentW) + currentb
    testPredicted = np.round(np.clip(testPredicted, 0, 1))
    error_rate_test = np.sum(np.abs(testPredicted - testTarget)
                        ) / np.size(testTarget)
    acc_test.append(1 - error_rate)

if __name__ == '__main__':
    lam = [0., 0.0001, 0.001, 0.01, 0.1, 1.]
    eta = 0.2
    color = ['b', 'g', 'k', 'r', 'y', 'c']
    batch_size = 50
    acc = []
    acc_test = []
    for i in range(len(lam)):
        np.set_printoptions(precision=2)
        runMult(batch_size, lam[i], color[i])

    plt.axis([0, 100, 0, 1])
    plt.xlabel("Number of Updates")
    plt.ylabel("Total Loss Function L")
    plt.legend()
    plt.title("Total Loss Function L vs. Number of Updates")

    plt.figure(2)
    plt.semilogx(lam, acc, 'bo-', linewidth=2.0)
    plt.grid()
    # plt.plot(lam, acc, 'bo-', linewidth=2.0)
    plt.xlabel("$\lambda$")
    plt.ylabel("Classification Accuracy on Validation Set")
    plt.title("$\lambda$ vs. Classification Accuracy on Validation Set")

    plt.figure(3)
    plt.semilogx(lam, acc_test, 'bo-', linewidth=2.0)
    plt.grid()
    # plt.plot(lam, acc, 'bo-', linewidth=2.0)
    plt.xlabel("$\lambda$")
    plt.ylabel("Classification Accuracy on Test Set")
    plt.title("$\lambda$ vs. Classification Accuracy on Test Set")



    plt.show()