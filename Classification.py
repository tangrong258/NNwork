import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True )

def add_layer(inputs, in_size, out_size,activation_function=None):
    # add one more layer and return the output of this layer
    #layer_name = 'layer%s' % n_layer
    #with tf.name_scope(layer_name):
        #with tf.name_scope('weights'):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
    #tf.summary.histogram('W', Weights)
        #with tf.name_scope('biases'):
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
    #tf.summary.histogram('b', biases)
        #with tf.name_scope('Wx_plus_b'):
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
        #tf.summary.histogram('out', outputs)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs} )
    correct_prediction=tf.equal(tf.argmax (y_pre,1),tf.argmax(v_ys,1))#对比预测值中1的位置是不是对应于原来1的位置
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})#一个百分比，越高预测越准确
    return result


xs= tf.placeholder(tf.float32,[None,784])
ys= tf.placeholder(tf.float32,[None,10])

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(tf.reduce_sum(ys*tf.log(prediction),reduction_indices= [1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy )

sess=tf.Session()
sess.run(tf.initialize_all_variables ())

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)#从我下载好的xs,ys选择100个学习，再选择100个学习，不需要每一次都选择全部数据学习
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys })
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels) )