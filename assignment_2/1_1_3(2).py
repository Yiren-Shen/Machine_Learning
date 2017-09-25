import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def cross(y_hat,y_target):
    
    crossloss = -y_target*np.log(y_hat)-(1-y_target)*np.log(1-y_hat)
    squareloss = np.square(y_hat - y_target)
    return crossloss, squareloss

y_hat = np.linspace(0, 1, 100)
y_target = np.zeros(100)
cro= np.zeros(100)
sqr = np.zeros(100)
for i in xrange(100):
    cro[i],sqr[i]=cross(y_hat[i],y_target[i])

    print (cro[i], sqr[i])
    print ("\n")
plt.figure(0)
plt.subplot(211)
plt.ylabel("loss")
plt.xlabel("y_predict")
plt.plot(y_hat, cro, 'r-', linewidth=2.0,label='cross-entropy loss')
plt.plot(y_hat, sqr, 'b-', linewidth=2.0,label='squared-error loss')
plt.legend()
plt.savefig('cross_entropy_vs_squared.png')
plt.show()