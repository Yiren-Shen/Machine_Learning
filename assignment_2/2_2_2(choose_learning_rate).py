import tensorflow as tf
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


def build_block(input_X, hidden_units):
    a = np.int((input_X.get_shape()[1]))
    W = tf.Variable(tf.truncated_normal(shape=[a, hidden_units], stddev=sqrt(
        3.0 / (a + hidden_units)), dtype=tf.float32), name='weights')
    b = tf.Variable(0.0, name='biases')
    z = tf.add(tf.matmul(input_X, W), b)
    return z, W, b


def buildGraph():
    # Variable creation
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    X_valid = tf.placeholder(tf.float32, [None, 784], name='input_x_valid')
    X_test = tf.placeholder(tf.float32, [None, 784], name='input_x_test')
    y_target = tf.placeholder(tf.float32, [None, 10], name='train_target')
    y_valid = tf.placeholder(tf.float32, [None, 10], name='valid_target')
    y_test = tf.placeholder(tf.float32, [None, 10], name='test_target')
    Lambda = tf.placeholder("float32", name='Lambda')
    eta = tf.placeholder("float32", name='Lambda')

    # Graph definition
    h1, W1, b1 = build_block(X, 1000)
    x2 = tf.nn.relu(h1)
    h2, W2, b2 = build_block(x2, 10)
    y_predict = tf.nn.softmax(h2)

    y_predict_valid = tf.nn.softmax(
        tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(X_valid, W1), b1)), W2), b2))
    y_predict_test = tf.nn.softmax(
        tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(X_test, W1), b1)), W2), b2))

    logits = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=h2, labels=y_target, name=None))
    weight_loss = tf.reduce_sum(W1 * W1) * 0.5 * \
        Lambda + tf.reduce_sum(W2 * W2) * 0.5 * Lambda
    #loss = meanSquaredError + weight_loss
    loss_train = logits + weight_loss

    loss_valid = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.add(
        tf.matmul(tf.add(tf.matmul(X_valid, W1), b1), W2), b2), labels=y_valid, name=None))

    loss_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.add(
        tf.matmul(tf.add(tf.matmul(X_test, W1), b1), W2), b2), labels=y_test, name=None))

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate=eta)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
    train = optimizer.minimize(loss=loss_train)
    return X, y_target, X_valid, y_valid, X_test, y_test, Lambda, eta, train, y_predict, loss_train, y_predict_valid, y_predict_test, loss_valid, loss_test


def importData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
        trainData = np.reshape(trainData, (15000, 784))
        validData = np.reshape(validData, (1000, 784))
        testData = np.reshape(testData, (2724, 784))
        zero_train = np.zeros([len(trainTarget), 10])
        zero_valid = np.zeros([len(validTarget), 10])
        zero_test = np.zeros([len(testTarget), 10])
        for m in xrange(0, len(trainTarget)):
            zero_train[m][trainTarget[m]] = 1
        for m in xrange(0, len(validTarget)):
            zero_valid[m][validTarget[m]] = 1
        for m in xrange(0, len(testTarget)):
            zero_test[m][testTarget[m]] = 1

    return trainData, zero_train, validData, zero_valid, testData, zero_test

X, y_target, X_valid, y_valid, X_test, y_test, Lambda, eta, train, y_predict, loss_train, y_predict_valid, y_predict_test, loss_valid, loss_test = buildGraph()

trainData, trainTarget, validData, validTarget, testData, testTarget = importData()


# Initialize session
init = tf.initialize_all_variables()

with tf.Session() as sess:
    # Training hyper-parameters
    B = 500
    epho = 50
    wd_lambda = 0.005
    wd_eta = [0.001, 0.0003, 0.0001]
    color = ['r', 'g', 'y']
    for idx in range(len(wd_eta)):  # i -> idx ============================== #
        sess.run(init)  # ============================== #
        # np.set_printoptions(precision=2)
        # runMult(B, eta[i], color[i])
        wList = []
        trainLoss_list = []
        validLoss_list = []
        testLoss_list = []
        trainclasserr_list = []
        validclasserr_list = []
        testclasserr_list = []
        numBatches = np.floor(len(trainData) / B)
        i = 0
        for step in xrange(0, epho * np.int(numBatches)):

            # sample minibatch without replacement

            feeddict_train = {X: trainData[i * B:(i + 1) * B, :], y_target: trainTarget[
                i * B:(i + 1) * B, :], Lambda: wd_lambda, eta: wd_eta[idx]}
            _, err_train, y_trainhat = sess.run(
                [train, loss_train, y_predict], feed_dict=feeddict_train)

            los_valid, y_pred_valid = sess.run([loss_valid, y_predict_valid], feed_dict={
                                               X_valid: validData, y_valid: validTarget})
            # loss_valid, y_pred_valid = sess.run([loss_valid, y_predict_valid],feed_dict={X_valid:validData,y_valid:validTarget})
            # #
            los_test, y_pred_test = sess.run([loss_test, y_predict_test], feed_dict={
                                             X_test: testData, y_test: testTarget})

            classerror_train = 1. - np.mean(np.equal(np.argmax(
                trainTarget[i * B:(i + 1) * B], 1), np.argmax(y_trainhat, 1), dtype=np.float32))
            classerror_valid = 1. - \
                np.mean(np.equal(np.argmax(validTarget, 1),
                                 np.argmax(y_pred_valid, 1), dtype=np.float32))
            classerror_test = 1. - np.mean(
                np.equal(np.argmax(testTarget, 1), np.argmax(y_pred_test, 1), dtype=np.float32))
            i += 1
            #
            # trainLoss_list.append(err_train)
            # trainclasserr_list.append(classerror_train)
            # validLoss_list.append(los_valid)
            # validclasserr_list.append(classerror_valid)
            # testLoss_list.append(los_test)
            # testclasserr_list.append(classerror_test)
            if step % numBatches == 0:
                # randIdx = np.arange(len(trainData))
                # np.random.shuffle(randIdx)
                # trainData = trainData[randIdx]
                i = 0
                print("Iter :%3d, Cross-Entropy-loss-train: %4.2f,classification error: %4.2f " %
                      (step, err_train, classerror_train))
                trainLoss_list.append(err_train)
                trainclasserr_list.append(classerror_train)
                validLoss_list.append(los_valid)
                validclasserr_list.append(classerror_valid)
                testLoss_list.append(los_test)
                testclasserr_list.append(classerror_test)

        x_axis = np.arange(1, epho + 1)
        plt.plot(x_axis, trainLoss_list, '%so-' %
                 color[idx], linewidth=0.5,  label='$\eta$ = %f' % wd_eta[idx])
        plt.ylabel('cross-entropy loss')
        plt.xlabel('epho')
        plt.axis([0, epho + 1, None, None])
        plt.legend()
        plt.savefig('221c')
plt.show()

# plt.figure(0)
# plt.ylabel("cross-entropy loss")
# plt.semilogx(eta, trainLoss_list, 'bo-', linewidth=2.0)
# plt.legend()
# plt.savefig('221c')
# plt.show()