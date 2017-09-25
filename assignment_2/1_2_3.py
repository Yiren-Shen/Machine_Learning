import tensorflow as tf
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt



def buildGraph():
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784,10], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    
    # X_valid = tf.placeholder(tf.float32, [None, 784], name='input_x_valid')
    X_test=tf.placeholder(tf.float32, [None, 784], name='input_x_test')
    y_target=tf.placeholder(tf.float32,[None,10],name='train_target')
    # y_valid = tf.placeholder(tf.float32, [None, 10],name='valid_target')
    y_test = tf.placeholder(tf.float32, [None, 10],name='test_target')
    Lambda = tf.placeholder("float32", name='Lambda')

    # Graph definition
    # h1,W1,b1=build_block(X,1000)
    # x2=tf.nn.relu(h1)
    # h2, W2,b2 = build_block(x2, 10)
    Z = tf.add(tf.matmul(X,W) , b)
    y_predict=tf.nn.softmax(tf.add(tf.matmul(X,W) , b))

    y_predict_test=tf.nn.softmax(tf.add(tf.matmul(X_test,W) , b))
    # y_predict_valid=tf.nn.softmax(tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(X_valid,W1) , b1)),W2),b2))
    # y_predict_test = tf.nn.softmax(tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(X_test, W1), b1)), W2), b2))

    logits=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z,labels=y_target,name=None))
    weight_loss = tf.reduce_sum(W*W) * 0.5 * Lambda
    #loss = meanSquaredError + weight_loss
    loss_train=logits+weight_loss

    # loss_valid=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.add(tf.matmul(tf.add(tf.matmul(X_valid,W1) , b1),W2),b2),labels=y_valid,name=None))

    loss_test=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= tf.add(tf.matmul(X_test,W) , b),labels=y_test,name=None))


    # Training mechanism
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
    train = optimizer.minimize(loss=loss_train)
    # return X,y_target, X_valid,y_valid,X_test,y_test,Lambda, train, y_predict,loss_train,y_predict_valid,y_predict_test,loss_valid,loss_test
    return X,y_target, X_test,y_test,Lambda, train, y_predict,loss_train, y_predict_test,loss_test

def importData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
        trainData = np.reshape(trainData, (15000,784))
        validData = np.reshape(validData, (1000,784))
        testData = np.reshape(testData, (2724,784))
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

# X, y_target, X_valid,y_valid,X_test,y_test,Lambda,train,y_predict,loss_train,y_predict_valid,y_predict_test,loss_valid,loss_test = buildGraph()
X, y_target, X_test,y_test,Lambda,train,y_predict,loss_train,y_predict_test,loss_test = buildGraph()

trainData, trainTarget, validData, validTarget, testData, testTarget=importData()


# Initialize session
init = tf.initialize_all_variables()

with tf.Session() as sess:
## Training hyper-parameters
    sess.run(init)
    B = 500
    epho=100
    wd_lambda = 0.01
    wList = []
    trainLoss_list = []
    validLoss_list = []
    testLoss_list = []
    trainAcc_list = []
    validAcc_list = []
    testAcc_list = []
    numBatches = np.floor(len(trainData)/B)
    i=0
    for step in xrange(0,epho*np.int(numBatches)):

        ## sample minibatch without replacement

        feeddict_train = {X: trainData[i * B:(i + 1) * B, :], y_target: trainTarget[i * B:(i + 1) * B, :],
                          Lambda: wd_lambda}
        _, err_train, y_trainhat = sess.run([train, loss_train, y_predict], feed_dict=feeddict_train)

        # los_valid, y_pred_valid = sess.run([loss_valid, y_predict_valid], feed_dict={X_valid:validData,y_valid:validTarget})
        # loss_valid, y_pred_valid = sess.run([loss_valid, y_predict_valid],feed_dict={X_valid:validData,y_valid:validTarget})
        # #
        los_test, y_pred_test = sess.run([loss_test, y_predict_test], feed_dict={X_test:testData,y_test:testTarget})

        acc_train=np.mean(np.equal(np.argmax(trainTarget[i*B:(i+1)*B], 1), np.argmax(y_trainhat, 1),dtype=np.float32))
        # acc_valid = np.mean(np.equal(np.argmax(validTarget, 1), np.argmax(y_pred_valid, 1), dtype=np.float32))
        acc_test = np.mean(
            np.equal(np.argmax(testTarget, 1), np.argmax(y_pred_test, 1), dtype=np.float32))
        i +=1
        #
        trainLoss_list.append(err_train)
        trainAcc_list.append(acc_train)
        # validLoss_list.append(los_valid)
        # validAcc_list.append(acc_valid)
        testLoss_list.append(los_test)
        testAcc_list.append(acc_test)
        if step % numBatches == 0:
           # randIdx = np.arange(len(trainData))
           #np.random.shuffle(randIdx)
           # trainData = trainData[randIdx]
           i=0
           print("Iter :%3d, trainloss: %4.2f,Acc: %4.2f " % (step, err_train, acc_train))
    print("testloss: %4.2f,testAcc: %4.2f " % (los_test, acc_test))
           

    plt.figure(0)
    plt.subplot(211)
    plt.ylabel("cross-entropy loss")
    plt.plot(trainLoss_list, 'r-', linewidth=2.0,label='trainLoss_list')
    # plt.plot(validLoss_list, 'b-', linewidth=2.0)
    plt.plot(testLoss_list, 'b-', linewidth=2.0,label='testLoss_list')
    plt.legend()
    plt.savefig('123loss.png')
    plt.subplot(212)
    plt.ylabel("accuracy")
    plt.plot(trainAcc_list, 'r-', linewidth=2.0,label='trainAcc_list')
    # plt.plot(validAcc_list, 'b-', linewidth=2.0)
    plt.plot(testAcc_list, 'b-', linewidth=2.0,label='testAcc_list')
    plt.legend()
    plt.savefig('123acc.png')
    plt.show()