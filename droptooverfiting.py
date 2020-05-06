
import tensorflow  as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split#分配训练集和测试集
from sklearn.preprocessing import LabelBinarizer

digits = load_digits()
X=digits.data #加载0-9的数字图片data(和minst类似)
y=digits.target
y=LabelBinarizer ().fit_transform(y)#他是数字几就在第几位放1（0-9）
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)

def add_layer(inputs, in_size, out_size,layer_name,activation_function=None):
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
    Wx_plus_b = tf.nn.dropout (Wx_plus_b,keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs

keep_prob = tf.placeholder(tf.float32)#一直使得train保持多少的结果不被drop掉，用来减小test与train之间差距
xs= tf.placeholder(tf.float32,[None,64])
ys= tf.placeholder(tf.float32,[None,10])

# add hidden layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
# add output layer
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices= [1]))
tf.summary.scalar('loss', cross_entropy )

train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy )


sess=tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)
sess.run(tf.initialize_all_variables ())

for i in range(1000):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})#保持0.5不被drop掉
    if i % 50 == 0:
        print(sess.run(cross_entropy , feed_dict={xs: X_train , ys: y_train, keep_prob: 1}))
        print(sess.run(cross_entropy , feed_dict={xs:  X_test , ys: y_test, keep_prob: 1 }))#记录结果时，不需要drop任何东西
        train_result = sess.run(merged, feed_dict={xs: X_train , ys: y_train, keep_prob: 1 })
        test_result = sess.run(merged, feed_dict={xs: X_test , ys: y_test, keep_prob: 1 })
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
