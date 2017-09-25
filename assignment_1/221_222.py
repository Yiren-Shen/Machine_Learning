import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


lam = 0.001

def runMult(batch_size):
    with np.load ("tinymnist.npz") as data :
        trainData, trainTarget = data ["x"], data["y"]
        validData, validTarget = data ["x_valid"], data ["y_valid"]
        testData, testTarget = data ["x_test"], data ["y_test"]
        
    n_train_batches = 700

    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[64,1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 64], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')

    # Graph definition
    y_predicted = tf.matmul(X,W) + b

    # Error definition
    cost = tf.add(tf.reduce_sum(tf.square(y_predicted - y_target))/(2*batch_size),tf.mul(lam / 2., tf.reduce_sum(tf.square(W))))

        # Training mechanism
    #tune learning rate 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimize(loss=cost)

    # Initialize session
    init = tf.initialize_all_variables()

    sess = tf.InteractiveSession()
    sess.run(init)

    initialW = sess.run(W)  
    initialb = sess.run(b)

    print("Initial weights: %s, initial bias: %.2f"%(initialW, initialb))
    # Training model
  
    epoch = 0
    n_train_batches /= batch_size
    err = []
    
    while (epoch < 100):
        epoch = epoch + 1
        
        for minibatch_index in np.arange(n_train_batches):
            _, error, currentW, currentb, yhat = sess.run([train, cost, W, b, y_predicted], feed_dict={X: trainData[minibatch_index * batch_size: (minibatch_index + 1) * batch_size], y_target: trainTarget[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]})
            
        
        err.append(error)   
    
    
    print error           
          
    plt.plot(err, 'bo-', linewidth=2.0)
    plt.xlabel("Number of Updates")
    plt.ylabel("Total Loss Function L")
    plt.title("Total Loss Function L vs. Number of Updates")
    plt.show()
    

    # Testing model
    errTest = sess.run(cost, feed_dict={X: testData, y_target: testTarget})
    print("Final testing MSE: %.2f"%(errTest))      


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    runMult(50)#we can change batch size here