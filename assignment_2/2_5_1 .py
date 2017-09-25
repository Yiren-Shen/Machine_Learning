import tensorflow as tf
import numpy as np
from math import sqrt
import math
import matplotlib.pyplot as plt


def build_block(input_X, hidden_units):
    a = np.int((input_X.get_shape()[1]))
    W = tf.Variable(
        tf.truncated_normal(shape=[a, hidden_units], stddev=sqrt(3.0 / (a + hidden_units)), dtype=tf.float32),
        name='weights')
    b = tf.Variable(0.0, name='biases')
    z = tf.add(tf.matmul(input_X, W), b)
    return z, W, b



def buildGraph():
    # Variable creation

    np.random.seed(521)
    learning_rate = math.exp( 3 * np.random.random_sample() - 7.5)
    layer_number = np.random.randint(1, 6)
    number_of_hidden_unit = np.random.randint(100, 501)
    weight_decay = math.exp(3 * np.random.random_sample() - 9.)
    drop_out = np.random.randint(0, 2)

    if drop_out==1:
        keep_prob = 0.5
        print(
            "Learning Rate: %.4f, Number of Layers: %d, Number of Hidden Units: %d, Weight Decay Coefficient: %.4f, Dropout: Yes"
            % (learning_rate, layer_number, number_of_hidden_unit, weight_decay))
    else:
        keep_prob=1.0
        print(
            "Learning Rate: %.4f, Number of Layers: %d, Number of Hidden Units: %d, Weight Decay Coefficient: %.4f, Dropout: No"
            % (learning_rate, layer_number, number_of_hidden_unit, weight_decay))
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    X_valid = tf.placeholder(tf.float32, [None, 784], name='input_x_valid')
    X_test = tf.placeholder(tf.float32, [None, 784], name='input_x_test')
    y_target = tf.placeholder(tf.float32, [None, 10], name='train_target')
    y_valid = tf.placeholder(tf.float32, [None, 10], name='valid_target')
    y_test = tf.placeholder(tf.float32, [None, 10], name='test_target')
    #Lambda = tf.placeholder("float32", name='Lambda')

    # Graph definition
    if layer_number==1:
        h1, W1, b1 = build_block(X, 1000)
        x2 = tf.nn.relu(h1)

        x3 = tf.nn.dropout(x2, keep_prob)

        h2, W2, b2 = build_block(x3, 10)
        y_predict = tf.nn.softmax(h2)

        y_predict_valid = tf.nn.softmax(tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(X_valid, W1), b1)), W2), b2))
        y_predict_test = tf.nn.softmax(tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(X_test, W1), b1)), W2), b2))

        logits = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h2, labels=y_target, name=None))
        weight_loss = tf.reduce_sum(W1 * W1) * 0.5 * Lambda + tf.reduce_sum(W2 * W2) * 0.5 * Lambda
        # loss = meanSquaredError + weight_loss
        loss_train = logits + weight_loss

        loss_valid = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=tf.add(tf.matmul(tf.add(tf.matmul(X_valid, W1), b1), W2), b2),
                labels=y_valid, name=None))

        loss_test = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=tf.add(tf.matmul(tf.add(tf.matmul(X_test, W1), b1), W2), b2),
                                                    labels=y_test, name=None))
    elif layer_number==2:
        h1, W1, b1 = build_block(X, number_of_hidden_unit)
        x2 = tf.nn.dropout(tf.nn.relu(h1), keep_prob)
        x2_valid=tf.nn.relu(tf.add(tf.matmul(X_valid, W1), b1))
        x2_test = tf.nn.relu(tf.add(tf.matmul(X_test, W1), b1))
        h2, W2, b2 = build_block(x2, number_of_hidden_unit)
        x3 = tf.nn.dropout(tf.nn.relu(h2), keep_prob)
        x3_valid = tf.nn.relu(tf.add(tf.matmul(x2_valid, W2), b2))
        x3_test = tf.nn.relu(tf.add(tf.matmul(x2_test, W2), b2))
        h3, W3, b3 = build_block(x3, 10)
        y_predict = tf.nn.softmax(h3)
        y_predict_valid=tf.nn.softmax(tf.add(tf.matmul(x3_valid, W3), b3))
        y_predict_test = tf.nn.softmax(tf.add(tf.matmul(x3_test, W3), b3))



        logits = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h3, labels=y_target, name=None))
        weight_loss = 0.5*weight_decay*(tf.reduce_sum(W1 * W1)  + tf.reduce_sum(W2 * W2)+ tf.reduce_sum(W3 * W3) )
        # loss = meanSquaredError + weight_loss
        loss_train = logits + weight_loss

        loss_valid = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=tf.add(tf.matmul(x3_valid, W3), b3),
                labels=y_valid, name=None))

        loss_test = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=tf.add(tf.matmul(x3_test, W3), b3),
                                                    labels=y_test, name=None))
    elif layer_number == 3:
        h1, W1, b1 = build_block(X, number_of_hidden_unit)
        x2 = tf.nn.dropout(tf.nn.relu(h1), keep_prob)
        x2_valid = tf.nn.relu(tf.add(tf.matmul(X_valid, W1), b1))
        x2_test = tf.nn.relu(tf.add(tf.matmul(X_test, W1), b1))
        h2, W2, b2 = build_block(x2, number_of_hidden_unit)
        x3 = tf.nn.dropout(tf.nn.relu(h2), keep_prob)
        x3_valid = tf.nn.relu(tf.add(tf.matmul(x2_valid, W2), b2))
        x3_test = tf.nn.relu(tf.add(tf.matmul(x2_test, W2), b2))
        h3, W3, b3 = build_block(x3, number_of_hidden_unit)
        x4 = tf.nn.dropout(tf.nn.relu(h3), keep_prob)
        x4_valid = tf.nn.relu(tf.add(tf.matmul(x3_valid, W3), b3))
        x4_test = tf.nn.relu(tf.add(tf.matmul(x3_test, W3), b3))
        h4, W4, b4 = build_block(x4, 10)
        y_predict = tf.nn.softmax(h4)
        y_predict_valid = tf.nn.softmax(tf.add(tf.matmul(x4_valid, W4), b4))
        y_predict_test = tf.nn.softmax(tf.add(tf.matmul(x4_test, W4), b4))

        logits = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4, labels=y_target, name=None))
        weight_loss = 0.5 * weight_decay * (tf.reduce_sum(W1 * W1) + tf.reduce_sum(W2 * W2) + tf.reduce_sum(W3 * W3)+ tf.reduce_sum(W4 * W4))
        # loss = meanSquaredError + weight_loss
        loss_train = logits + weight_loss

        loss_valid = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=tf.add(tf.matmul(x4_valid, W4), b4),
                labels=y_valid, name=None))

        loss_test = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=tf.add(tf.matmul(x4_test, W4), b4),labels=y_test, name=None))
    elif layer_number == 4:
        h1, W1, b1 = build_block(X, number_of_hidden_unit)
        x2 = tf.nn.dropout(tf.nn.relu(h1), keep_prob)
        x2_valid = tf.nn.relu(tf.add(tf.matmul(X_valid, W1), b1))
        x2_test = tf.nn.relu(tf.add(tf.matmul(X_test, W1), b1))
        h2, W2, b2 = build_block(x2, number_of_hidden_unit)
        x3 = tf.nn.dropout(tf.nn.relu(h2), keep_prob)
        x3_valid = tf.nn.relu(tf.add(tf.matmul(x2_valid, W2), b2))
        x3_test = tf.nn.relu(tf.add(tf.matmul(x2_test, W2), b2))
        h3, W3, b3 = build_block(x3, number_of_hidden_unit)
        x4 = tf.nn.dropout(tf.nn.relu(h3), keep_prob)
        x4_valid = tf.nn.relu(tf.add(tf.matmul(x3_valid, W3), b3))
        x4_test = tf.nn.relu(tf.add(tf.matmul(x3_test, W3), b3))
        h4, W4, b4 = build_block(x4, number_of_hidden_unit)
        x5 = tf.nn.dropout(tf.nn.relu(h4), keep_prob)
        x5_valid = tf.nn.relu(tf.add(tf.matmul(x4_valid, W4), b4))
        x5_test = tf.nn.relu(tf.add(tf.matmul(x4_test, W4), b4))
        h5, W5, b5 = build_block(x5, 10)
        y_predict = tf.nn.softmax(h5)
        y_predict_valid = tf.nn.softmax(tf.add(tf.matmul(x5_valid, W5), b5))
        y_predict_test = tf.nn.softmax(tf.add(tf.matmul(x5_test, W5), b5))

        logits = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h5, labels=y_target, name=None))
        weight_loss = 0.5 * weight_decay * (
        tf.reduce_sum(W1 * W1) + tf.reduce_sum(W2 * W2) + tf.reduce_sum(W3 * W3) + tf.reduce_sum(W4 * W4) + tf.reduce_sum(W5 * W5))
        # loss = meanSquaredError + weight_loss
        loss_train = logits + weight_loss

        loss_valid = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=tf.add(tf.matmul(x5_valid, W5), b5),
                labels=y_valid, name=None))

        loss_test = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=tf.add(tf.matmul(x5_test, W5), b5),
                                                    labels=y_test, name=None))
    elif layer_number == 5:
        h1, W1, b1 = build_block(X, number_of_hidden_unit)
        x2 = tf.nn.dropout(tf.nn.relu(h1), keep_prob)
        x2_valid = tf.nn.relu(tf.add(tf.matmul(X_valid, W1), b1))
        x2_test = tf.nn.relu(tf.add(tf.matmul(X_test, W1), b1))
        h2, W2, b2 = build_block(x2, number_of_hidden_unit)
        x3 = tf.nn.dropout(tf.nn.relu(h2), keep_prob)
        x3_valid = tf.nn.relu(tf.add(tf.matmul(x2_valid, W2), b2))
        x3_test = tf.nn.relu(tf.add(tf.matmul(x2_test, W2), b2))
        h3, W3, b3 = build_block(x3, number_of_hidden_unit)
        x4 = tf.nn.dropout(tf.nn.relu(h3), keep_prob)
        x4_valid = tf.nn.relu(tf.add(tf.matmul(x3_valid, W3), b3))
        x4_test = tf.nn.relu(tf.add(tf.matmul(x3_test, W3), b3))
        h4, W4, b4 = build_block(x4, number_of_hidden_unit)
        x5 = tf.nn.dropout(tf.nn.relu(h4), keep_prob)
        x5_valid = tf.nn.relu(tf.add(tf.matmul(x4_valid, W4), b4))
        x5_test = tf.nn.relu(tf.add(tf.matmul(x4_test, W4), b4))
        h5, W5, b5 = build_block(x5, number_of_hidden_unit)
        x6 = tf.nn.dropout(tf.nn.relu(h5), keep_prob)
        x6_valid = tf.nn.relu(tf.add(tf.matmul(x5_valid, W5), b5))
        x6_test = tf.nn.relu(tf.add(tf.matmul(x5_test, W5), b5))
        h6, W6, b6 = build_block(x6, 10)
        y_predict = tf.nn.softmax(h6)
        y_predict_valid = tf.nn.softmax(tf.add(tf.matmul(x6_valid, W6), b6))
        y_predict_test = tf.nn.softmax(tf.add(tf.matmul(x6_test, W6), b6))

        logits = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h6, labels=y_target, name=None))
        weight_loss = 0.5 * weight_decay * (
        tf.reduce_sum(W1 * W1) + tf.reduce_sum(W2 * W2) + tf.reduce_sum(W3 * W3) + tf.reduce_sum(W4 * W4) + tf.reduce_sum(W5 * W5)+tf.reduce_sum(W6 * W6))
        # loss = meanSquaredError + weight_loss
        loss_train = logits + weight_loss

        loss_valid = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=tf.add(tf.matmul(x6_valid, W6), b6),
                labels=y_valid, name=None))

        loss_test = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=tf.add(tf.matmul(x6_test, W6), b6),
                                                    labels=y_test, name=None))

    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
    train = optimizer.minimize(loss=loss_train)
    return keep_prob, X, y_target, X_valid, y_valid, X_test, y_test,  train, y_predict, loss_train, y_predict_valid, y_predict_test, loss_valid, loss_test


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

#layer_number,learning_rate,number_of_hidden_unit,weight_decay,drop_out=2,0.001,1000,0.001,1
keep_prob, X, y_target, X_valid, y_valid, X_test, y_test, train, y_predict, loss_train, y_predict_valid, y_predict_test, loss_valid, loss_test= buildGraph()

trainData, trainTarget, validData, validTarget, testData, testTarget = importData()

# Initialize session
init = tf.initialize_all_variables()

with tf.Session() as sess:
    ## Training hyper-parameters
    sess.run(init)
    row = 10
    col = 10
    B = 500
    epho = 30

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

        ## sample minibatch without replacement

        feeddict_train = {X: trainData[i * B:(i + 1) * B, :], y_target: trainTarget[i * B:(i + 1) * B, :]}
        _, err_train, y_trainhat = sess.run([train, loss_train, y_predict], feed_dict=feeddict_train)

        los_valid, y_pred_valid = sess.run([loss_valid, y_predict_valid],
                                           feed_dict={X_valid: validData, y_valid: validTarget})

        los_test, y_pred_test = sess.run([loss_test, y_predict_test], feed_dict={X_test: testData, y_test: testTarget})

        classerror_train = 1. - np.mean(
            np.equal(np.argmax(trainTarget[i * B:(i + 1) * B], 1), np.argmax(y_trainhat, 1), dtype=np.float32))
        classerror_valid = 1. - np.mean(
            np.equal(np.argmax(validTarget, 1), np.argmax(y_pred_valid, 1), dtype=np.float32))
        classerror_test = 1. - np.mean(
            np.equal(np.argmax(testTarget, 1), np.argmax(y_pred_test, 1), dtype=np.float32))
        i += 1

        if step % numBatches == 0:
            i = 0
            print(
            "Iter :%3d, Cross-entropy loss: %4.2f,Classification error: %4.2f " % (step, err_train, classerror_train*100))
            print(
            "Iter :%3d, Validation Classification Error: %4.2f,Test Classification error: %4.2f " % (step, classerror_valid*100, classerror_test*100))
            trainLoss_list.append(err_train)
            trainclasserr_list.append(classerror_train)
            validLoss_list.append(los_valid)
            validclasserr_list.append(classerror_valid)
            testLoss_list.append(los_test)
            testclasserr_list.append(classerror_test)

    # plt.figure(0)
    # plt.ylabel("Cross-entropy loss")
    # plt.xlabel("Epoch")
    # plt.plot(trainLoss_list, 'r-', linewidth=2.0, label='trainLoss_list')
    # plt.plot(validLoss_list, 'b-', linewidth=2.0, label='validLoss_list')
    # plt.plot(testLoss_list, 'g-', linewidth=2.0, label='testLoss_list')
    # plt.legend()
    # plt.savefig('241')
    #
    # plt.figure(1)
    # plt.ylabel("Classification Error")
    # plt.xlabel("Epoch")
    # plt.plot(trainclasserr_list, 'r-', linewidth=2.0, label='trainAcc_list')
    # plt.plot(validclasserr_list, 'b-', linewidth=2.0, label='validAcc_list')
    # plt.plot(testclasserr_list, 'g-', linewidth=2.0, label='testAcc_list')
    # plt.legend()
    # plt.savefig('241b')
    # plt.show()
    #
    # plt.figure(2)
    # for i in xrange(0, 100):
    #     w_1 = np.reshape(w1[:, i], [28, 28])
    #     plt.subplot(row, col, i + 1)
    #     plt.imshow(w_1, interpolation="nearest", cmap="gray")
    # plt.show()