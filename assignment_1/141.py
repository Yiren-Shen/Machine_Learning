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
    return tf.sqrt(dist)


def GetK(dist):
    # get the exponentiated negative pairwise distances
    K = tf.exp(-lam * dist)
    return K


def GetSoftR(K):
    # get R based on the soft k-NN model
    sumK = tf.reduce_sum(K, 1, keep_dims=True)
    r = tf.div(K, sumK)
    return r


def GetGausR(K, trainData):
    # get R based on the Gaussian process regression model
    DistXX = GetDist(trainData, trainData)
    KXX = GetK(DistXX)
    r = tf.transpose(tf.matmul(tf.matrix_inverse(KXX), tf.transpose(K)))
    return r


def GetHatY(trainTarget, r):
    # get the predictions
    hatY = tf.matmul(tf.transpose(trainTarget), tf.transpose(r))
    return hatY


def plot(predictions, data, target):
    plt.plot(data, target, 'bo')
    plt.plot(data, predictions, 'ro')
    # plt.axis([1, 10, -2, 11])
    return

lam = 100
dist = GetDist(trainData, Data)
K = GetK(dist)

# ======================= soft k-NN =======================
softR = GetSoftR(K)
softHatY = GetHatY(trainTarget, softR)
with tf.Session() as sess:
    softHatY = np.squeeze(sess.run(softHatY))
# print "Predictions with soft k-NN model:\n", softHatY
plt.figure(0)
plot(softHatY, Data, Target)
plt.title("soft k-NN, Lambda=%.2f" % lam)
plt.savefig('softkNN.png')
# ======================= Gaussian =======================
gausR = GetGausR(K, trainData)
gausHatY = GetHatY(trainTarget, gausR)
with tf.Session() as sess:
    gausHatY = np.squeeze(sess.run(gausHatY))
    # print np.dtype(trainTarget[0])
# print "Predictions with Gaussian progress regression model:\n", gausHatY
plt.figure(1)
plot(gausHatY, Data, Target)
plt.title("Gaussian progress regression, Lambda=%.2f" % lam)
plt.savefig('gaussian.png')
plt.show()