import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(521)
Data = np.linspace(1.0, 10.0, num=100)[:, np. newaxis]
Target = np.sin(Data) + 0.1 * np.power(Data, 2) + 0.5 * np.random.randn(100, 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]


def GetDist(v1, v2):
    v1 = tf.constant(v1)
    v2 = tf.constant(v2)
    expanded_v1 = tf.expand_dims(v1, 0)  # shape: (B,N) -> (1, B, N)
    expanded_v2 = tf.expand_dims(v2, 1)  # shape: (C,N) -> (C, 1, N)
    diff = tf.sub(expanded_v1, expanded_v2)  # subtract with broadcasting
    sqr = tf.square(diff)
    dist = tf.reduce_sum(sqr, 2)  # sum over N
    return dist


def GetIdx(dist, k):
    indices = tf.nn.top_k(-dist, k).indices
    return indices


def GetHatY(trainData, testData, k):
    # get the distance matrix
    dist = GetDist(trainData, testData)
    # get the indices of the k nearest training points to the test points
    indices = GetIdx(dist, k)

    with tf.Session() as sess:
        indices = sess.run(indices)

    # initialize the responsibilities of all the training points
    r = np.zeros([testData.size, trainData.size])
    # set the responsibilities of the k nearest training points to be 1/k
    for i in range(0, testData.size):
        r[i, indices[i]] = 1. / k
    hatY = np.dot(np.transpose(trainTarget), np.transpose(r))
    hatY = hatY.reshape(-1, 1)
    return hatY


def GetLoss(hatY, testTarget):
    # get the MSE loss
    testLoss = ((hatY - testTarget) ** 2).sum() / (2 * testData.size)
    return testLoss


def Predict(trainData, testData, testTarget, k):
    # compute and return the predictions and the loss
    hatY = np.zeros([np.size(k), testData.size, 1])
    Loss = np.zeros(np.size(k))
    for n in range(0, np.size(k)):
        hatY[n] = GetHatY(trainData, testData, k[n])
        Loss[n] = GetLoss(hatY[n], testTarget)
    return Loss, hatY

# set the value of k
k = [1, 3, 5, 50]

trainLoss, trainHatY = Predict(trainData, trainData, trainTarget, k)
validLoss, validHatY = Predict(trainData, validData, validTarget, k)
testLoss, testHatY = Predict(trainData, testData, testTarget, k)

print "For k =", k, ":"
print "The predictions of training targets are:\n", trainHatY.reshape([np.size(k), -1])
print "The training MSE loss is\n", trainLoss
print "The predictions of validation targets are:\n", validHatY.reshape([np.size(k), -1])
print "The validation MSE loss is\n", validLoss
print "The predictions of test targets are:\n", testHatY.reshape([np.size(k), -1])
print "The test MSE loss is\n", testLoss

# Plot the prediction function for x = [0, 11]
X = np.linspace(0.0, 11.0, num=1000)[:, np.newaxis]
XTarget = np.sin(X) + 0.1 * np.power(X, 2) + 0.5 * np.random.randn(1000, 1)

XLoss, XHatY = Predict(trainData, X, XTarget, k)

for n in range(0, np.size(k)):
    plt.figure(n)
    plt.plot(X, XTarget, 'bo')
    plt.plot(X, XHatY[n], 'r-', linewidth=2.0)
    plt.axis([0, 10, -2, 10])
    plt.title("k-NN regression on X, k=%d" % k[n])
plt.show()
