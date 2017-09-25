import numpy as np
import tensorflow as tf


def GetDist(v1, v2):
    v1 = tf.constant(v1)
    v2 = tf.constant(v2)
    expanded_v1 = tf.expand_dims(v1, 0)  # shape: (1, B , N)
    expanded_v2 = tf.expand_dims(v2, 1)  # shape: (C, 1 , N)
    diff = tf.sub(expanded_v1, expanded_v2)  # subtract with broadcasting
    sqr = tf.square(diff)
    dist = tf.reduce_sum(sqr, 2)  # sum over N
    #dist = tf.sqrt(dist_sqr)
    return dist

B = 5  # length of X
C = 2  # length of Z
N = 3  # magnitude in a dataset
minvalue = 0
maxvalue = 11
X = np.random.uniform(minvalue, maxvalue, [B, N])
Z = np.random.uniform(minvalue, maxvalue, [C, N])
dist = GetDist(X, Z)

with tf.Session() as sess:
    print "X=", X
    print "Z=", Z
    print "DIST=", sess.run(dist)
#     X1    X2  .   .   XB
# Z1
# Z2
# .
# .
# ZC
