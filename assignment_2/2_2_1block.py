def build_block(input_X,hidden_units):
    a=np.int((input_X.get_shape()[1]))
    W = tf.Variable(tf.truncated_normal(shape=[a, hidden_units], stddev=sqrt(3.0/(a+ hidden_units)),dtype=tf.float32),name='weights')
    b = tf.Variable(0.0, name='biases')
    z=tf.add(tf.matmul(input_X,W) , b)
    return z,W,b