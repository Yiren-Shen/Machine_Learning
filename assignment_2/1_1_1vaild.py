import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+tf.exp(np.negative(z)))

def buildGraph():
    # Variable creation
    W = tf.Variable(tf.truncated_normal(shape=[784,1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y_target = tf.placeholder(tf.float32, [None,1], name='target_y')

    X_valid = tf.placeholder(tf.float32)
    y_valid = tf.placeholder(tf.float32, [None, 1])

    Lambda = tf.placeholder("float32", name='Lambda')
    # Graph definition
    y_train_predicted = tf.matmul(X,W) + b
    y_train_hat=sigmoid(y_train_predicted)

    y_valid_predicted = tf.matmul(X_valid, W) + b
    y_valid_hat = sigmoid(y_valid_predicted)

    #meanSquaredError = tf.reduce_mean(tf.reduce_mean(tf.square(y_predicted - y_target),reduction_indices=1,name='squared_error'),name='mean_squared_error')
    logistic_train=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_train_predicted,targets=y_target,name=None))
    logistic_valid = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_valid_predicted, targets=y_valid, name=None))
    weight_loss = tf.reduce_sum(W*W) * 0.5 * Lambda
    #loss = meanSquaredError + weight_loss

    loss_train=logistic_train+weight_loss
    loss_valid = logistic_valid + weight_loss
    # Training mechanism
    #optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    train = optimizer.minimize(loss=loss_train)
    return W, b, X,X_valid,y_target,  y_valid, Lambda, train, y_train_hat,y_valid_hat,loss_train,loss_valid

W, b, X,X_valid, y_target,  y_valid, Lambda, train, y_train_hat,y_valid_hat,loss_train,loss_valid = buildGraph()

# Initialize session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
# with np.load ("tinymnist (1).npz") as data:
#     trainData, trainTarget = data ["x"], data["y"]
#     validData, validTarget = data["x_valid"], data["y_valid"]
#     testData, testTarget = data["x_test"], data["y_test"]

with np.load("notMNIST.npz") as data:
    Data,Target=data["images"],data["labels"]
    posClass=2
    negClass=9
    dataIndx=(Target==posClass)+(Target==negClass)
    Data=Data[dataIndx]/255
    Target=Target[dataIndx].reshape(-1,1)
    Target[Target==posClass]=1
    Target[Target ==negClass]=0
    np.random.seed(521)
    randIdx=np.arange(len(Data))
    np.random.shuffle(randIdx)
    Data,Target=Data[randIdx],Target[randIdx]
    trainData,trainTarget=Data[:3500],Target[:3500]
    validData,validTarget=Data[3500:3600],Target[3500:3600]
    testData,testTarget=Data[3600:],Target[3600:]
    trainData=np.reshape(trainData,[3500,-1])
    validData=np.reshape(validData,[len(validTarget),-1])
    testData=np.reshape(testData,[len(testTarget),-1])

## Training hyper-parameters
B = 500
epho=2000
wd_lambda = 0.01
wList = []
trainLoss_list = []
validLoss_list = []
testLoss_list = []
trainAcc_list = []
validAcc_list = []
testAcc_list = []
numBatches = np.floor(len(trainData)/B)
validation_dict={"valid":(validData,validTarget),"test":(testData,testTarget)}
for step in xrange(0,epho*np.int(numBatches)):
    ## sample minibatch without replacement
    if step % numBatches == 0:
        # randIdx = np.arange(len(trainData))
        # np.random.shuffle(randIdx)
        # trainData = trainData[randIdx]
        i=0
    feeddict_train={X:trainData[i*B:(i+1)*B],y_target:trainTarget[i*B:(i+1)*B],Lambda:wd_lambda}
    feeddict_valid = {X_valid: validData, y_valid: validTarget, Lambda: wd_lambda}
    _,err_train,y_trainhat=sess.run([train,loss_train,y_train_hat],feed_dict=feeddict_train)
    err_vaild, y_validhat = sess.run([loss_valid, y_valid_hat], feed_dict=feeddict_valid)
    acc_vaild = np.mean((y_validhat > 0.5) == validTarget)
    acc_train=np.mean((y_trainhat>0.5)==trainTarget[i*B:(i+1)*B])

    i +=1
    #wList.append(currentW)
    trainLoss_list.append(err_train)
    validLoss_list.append(err_vaild)
    trainAcc_list.append(acc_train)
    validAcc_list.append(acc_vaild)
    if not (step % 1000):
        print("Iter :%3d, MSE-train: %4.2f"%(step,err_train))

plt.figure(0)
plt.subplot(211)
plt.ylabel("cross-entropy loss")
plt.plot(trainLoss_list, 'r-', linewidth=2.0)
plt.plot(validLoss_list, 'b-', linewidth=2.0)
plt.subplot(212)
plt.ylabel("accuracy")
plt.plot(trainAcc_list, 'r-', linewidth=2.0)
plt.plot(validAcc_list, 'b-', linewidth=2.0)
plt.show()
#
# for dataset in validation_dict:
#     data,target=validation_dict[dataset]
#     errTest=sess.run(loss_train,feed_dict={X:data,y_target:target,Lambda:wd_lambda})
#     acc_test=np.mean((y_valid_hat.eval(feed_dict={X:data,y_target:target })>0.5)==target)
#     print("Final %s MSE: %.2f,acc:%.2f"%(dataset,errTest,acc_test))