import numpy as np
import tensorflow as tf

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

# set the value of k
k = 5
# get the distance matrix
dist = GetDist(trainData, testData)
# get the indices of the k nearest training points to the test points
indices = GetIdx(dist, k)

with tf.Session() as sess:
    indices = sess.run(indices)

# initialize the responsibilities of all the training points
r = np.zeros([testData.size, trainData.size])
# set the responsibilities of the k nearest training points to be 1/k
# r[indices] = 1. / k
for i in range(0, testData.size):
    r[i, indices[i]] = 1. / k
print r
